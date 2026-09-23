"""Commander / tactical-officer policy: the network decides what and when, the
advisor decides where.

For cards whose placement head is worse than a random cell (see tactics.py),
placement is taken out of the network:

    commander (MicroRoyaleNet)  what to play and when, untouched
    tactical officer            (a) veto spends that would bankrupt us
                                (b) place the advisor's cards where it says

Letting the officer also initiate plays (`initiate=True`) is kept only for the
ablation: it crowded out the commander's offence and lost win rate. The
placement override is safe because it is passive, changing where a card lands
and never how often one is played.

Everything reads the observation, so the same object drives the simulator and
the live emulator loop. Thresholds come from card stats, never from win rate: a
Minions squad's HP (3 x 230) in the Fireball blast, a Musketeer (721) in the
Cannon's coverage.
"""
import numpy as np
import torch

from python_ai.advisors import tactics
from python_ai.models.policy_io import LSTM_HIDDEN

CANNON, FIREBALL = tactics.CANNON_ID, tactics.FIREBALL_ID

FIREBALL_MIN_CATCH = 690.0   # 3 x 230 HP: a Minions squad, a 3-cost trade
CANNON_MIN_COVER = 721.0     # one Musketeer approaching
CANNON_COOLDOWN_STEPS = 30   # the Cannon's 300-tick lifetime, in decisions


class HybridPolicy:
    """Stateful per-episode policy. Call reset() between matches."""

    # The advisor surface for each overridable card; __init__ rejects any card
    # without one rather than falling back to the building surface.
    ADVISOR_SURFACE = {
        CANNON: "building",
        FIREBALL: "spell",
    }

    def __init__(self, net, device=None, use_gate=True, use_advisor=True,
                 override_cards=(CANNON, FIREBALL), initiate=False,
                 use_opp_elixir=True,
                 reserve=4.0, fireball_min_catch=FIREBALL_MIN_CATCH,
                 cannon_min_cover=CANNON_MIN_COVER):
        self.net = net
        self.device = device or torch.device("cpu")
        self.use_gate = use_gate
        self.use_advisor = use_advisor
        self.override_cards = tuple(override_cards)
        unmapped = [c for c in self.override_cards if c not in self.ADVISOR_SURFACE]
        if unmapped:
            raise ValueError(
                f"override_cards contains cards with no advisor surface: {unmapped}. "
                f"Add them to HybridPolicy.ADVISOR_SURFACE or drop them -- silently "
                f"falling back to the building surface is what this check replaces.")
        self.initiate = initiate
        self.use_opp_elixir = use_opp_elixir
        self.gate = tactics.SolvencyGate(reserve=reserve)
        self.fireball_min_catch = fireball_min_catch
        self.cannon_min_cover = cannon_min_cover
        self._legal = {c: np.asarray(net._placement_legal[c]).astype(bool)
                       for c in self.ADVISOR_SURFACE}
        self.reset()

    def reset(self):
        h = torch.zeros(1, LSTM_HIDDEN, device=self.device)
        self.hidden = (h, h.clone())
        self._cannon_cd = 0
        self.stats = {"initiated_cannon": 0, "initiated_fireball": 0,
                      "overrode": 0, "gate_blocked": 0, "plays": 0}

    @torch.no_grad()
    def act(self, obs):
        """(slot, x, y) for env.step / the live actuator."""
        net = self.net
        o = np.asarray(obs, dtype=np.float32)
        t = torch.tensor(o, device=self.device).unsqueeze(0)

        mask = net.affordability_mask(t)
        costs = net.hand_costs_from_obs(t)[0].tolist()

        if self.use_gate:
            # No opponent-elixir estimate: the aux head that supplied one now
            # predicts the opponent's next card. The closed form (start +
            # rate*t - spent, from two observed scalars) is the right
            # replacement, but it is not wired in until verified against the
            # engine; `gate.mask` accepts None and skips that clause.
            opp = None

            allow = torch.tensor([self.gate.mask(o, costs, opp)],
                                 dtype=torch.bool, device=self.device)
            blocked = int((mask & ~allow).sum())
            self.stats["gate_blocked"] += blocked
            # The no-op is legal in both masks, so this can only remove card
            # slots.
            mask = mask & allow

        feats, embeds, sp = net.extract_features(t)
        logits, _, _, _, hidden = net.step_lstm_and_card(feats, self.hidden, mask)
        self.hidden = hidden
        hx = hidden[0]

        slot = int(logits.argmax(-1).item())
        hand = net.hand_card_ids(t)[0].tolist()
        affordable = mask[0].tolist()
        self._cannon_cd = max(0, self._cannon_cd - 1)

        # --- (b) tactical initiation ---
        # Only converts a no-op into a play, never overrides one: the commander
        # is better at timing.
        if self.initiate and self.use_advisor and slot == net.hand_size:
            fired = self._try_initiate(o, hand, affordable)
            if fired is not None:
                slot = fired

        # --- (c) placement ---
        card_id = hand[slot] if slot < net.hand_size else -1
        if self.use_advisor and card_id in self.override_cards:
            x, y = self._advisor_cell(o, card_id)
            self.stats["overrode"] += 1
        else:
            place = net.placement_given_card(
                hx, embeds, torch.tensor([slot], device=self.device), t, sp)
            cell = int(place.argmax(-1).item())
            x, y = float(cell % tactics.BOARD_W), float(cell // tactics.BOARD_W)

        if slot != net.hand_size:
            self.stats["plays"] += 1
        return slot, float(x), float(y)

    def _advisor_cell(self, o, card_id):
        # __init__ has rejected any override card without a surface.
        surface = self.ADVISOR_SURFACE[card_id]
        if surface == "spell":
            x, y, _ = tactics.best_spell_cell(o, legal=self._legal[card_id])
        else:
            x, y, _ = tactics.best_building_cell(o, legal=self._legal[card_id])
        return x, y

    def _try_initiate(self, o, hand, affordable):
        """A hand slot to play, or None. Solvency is already in `affordable`.
        """
        if FIREBALL in hand:
            s = hand.index(FIREBALL)
            if affordable[s]:
                _, _, catch = tactics.best_spell_cell(o, legal=self._legal[FIREBALL])
                if catch >= self.fireball_min_catch:
                    self.stats["initiated_fireball"] += 1
                    return s
        if CANNON in hand and self._cannon_cd == 0:
            s = hand.index(CANNON)
            if affordable[s]:
                _, _, cover = tactics.best_building_cell(o, legal=self._legal[CANNON])
                if cover >= self.cannon_min_cover:
                    self._cannon_cd = CANNON_COOLDOWN_STEPS
                    self.stats["initiated_cannon"] += 1
                    return s
        return None
