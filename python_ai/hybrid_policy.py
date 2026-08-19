"""Commander / tactical-officer policy: neural WHAT-and-WHEN, advisor WHERE.

THE ARCHITECTURE, AND WHY IT IS SHAPED THIS WAY
-----------------------------------------------
Measured on this codebase (see PLACEMENT_COLLAPSE.md), the neural policy's
placement head is a constant function for three of eight cards and is worth
LESS THAN A RANDOM CELL, while a deterministic advisor reading the same
observation is worth far more:

                    Cannon (tower HP preserved)   Fireball (elixir killed)
    trained policy            12.1                        0.000
    random legal cell        353.5                        0.276
    advisor                  564.1                        2.405

Three attempts to repair the head inside the network were measured and failed
(entropy coverage; advisor distillation, frozen and with an anchor). So the
placement decision is taken OUT of the network for exactly those cards.

Division of labour:

    commander (MicroRoyaleNet)  WHAT to play and WHEN -- untouched
    tactical officer            (a) veto spends that would bankrupt us
                                (b) place the dead cards where the advisor says

INITIATION WAS BUILT, MEASURED, AND IS OFF BY DEFAULT. The commander's take-up
of the dead cards is 0.06 (Cannon) and 0.00 (Fireball/Giant -- measured while
Giant was still in the deck; it left on 2026-08-16 and its path here was
removed on 2026-08-19, see the note above CANNON/FIREBALL below), so it seemed
obvious that overriding only "where" would fire too rarely to matter and that
the officer should also be allowed to START those plays. Measured over 30 paired
openings, that reasoning is wrong and the component is actively harmful:

    arm         win rate   tower HP DEALT/ep   cannons initiated/ep
    neural        0.600         7627                  --
    gate          0.633         6659                 0.00
    place         0.733         7655                 0.00
    initiate      0.467         5390                 4.03
    full          0.233         4216                 4.73

Initiating a Cannon four times a match crowds out the commander's own offence:
tower damage DEALT falls 2,236/episode with initiation alone and 3,410 with the
full stack (6 better / 24 worse, p = 0.0014), and win rate goes with it
(-0.367, p = 0.013). Defending better is worthless if it is paid for with the
attack. `initiate=True` is kept only so the ablation stays reproducible.

The placement override survives precisely because it is PASSIVE -- it changes
where a card lands and never how often one is played, so it cannot spend elixir
the commander did not already decide to spend. That is the whole reason it is
safe to bolt onto a policy whose timing is better than its geometry.

Everything is computed from the OBSERVATION, so this identical object drives
both the simulator evaluation and the live emulator loop -- the perception
encoder is pinned bit-equal to `getObservationForTeam(0)`
(`perception/tests/test_encoder_matches_engine.py`). What was measured is what
runs.

THRESHOLDS ARE SET A PRIORI FROM CARD STATS, NEVER TUNED ON WIN RATE. Fireball
fires when the advisor sees at least a 3-cost Minions squad's HP (3x230=690) in
the blast; the Cannon goes down when at least a Musketeer's worth (721) is
inside its future coverage. Tuning these against the outcome would be optional
stopping, which this project has already paid for once.
"""
import numpy as np
import torch

import tactics

# The GIANT path was removed on 2026-08-19, matching the cleanup advisor_target
# .py already applied to itself on 2026-08-17 and for the same reason: the deck
# became 2.6 Hog Cycle on 2026-08-16, Giant is not in DEFAULT_DECK, so every
# Giant branch here was unreachable. A dead entry is worse than none -- it made
# this policy look like it covered a win condition when it covered nothing, and
# it silently collapsed two of hybrid_ab.py's --per-card arms into duplicates of
# two others. tactics.GIANT_ID and tactics.best_giant_cell stay: prove_giant.py
# and train_selfplay.py's scenarios still use them.
CANNON, FIREBALL = tactics.CANNON_ID, tactics.FIREBALL_ID
LSTM_HIDDEN = 256

FIREBALL_MIN_CATCH = 690.0   # 3 x 230 HP: a Minions squad, i.e. a 3-cost trade
CANNON_MIN_COVER = 721.0     # one Musketeer approaching
CANNON_COOLDOWN_STEPS = 30   # the Cannon's own 300-tick lifetime, in decisions


class HybridPolicy:
    """Stateful per-episode policy. Call reset() between matches."""

    # Which advisor surface each overridable card gets. Explicit, because the
    # old dispatch fell through to the BUILDING surface for anything it did not
    # recognise -- and override_cards is a caller-supplied tuple, so an
    # unexpected card silently got a Cannon-shaped placement rather than an
    # error. Adding a card here without a surface is now a loud failure.
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
        scal = t[:, net.spatial_size:]
        # Costs live at [1 : 1+hand_size] of the scalar tail, already /10 by the
        # encoder -- the same offsets affordability_mask reads, so there is no
        # second copy of the observation layout here.
        costs = (scal[0, 1:1 + net.hand_size] * 10.0).tolist()

        if self.use_gate:
            # Opponent-elixir estimate from the PREVIOUS step's recurrent state.
            # One step stale by construction -- the mask has to be built before
            # this step's LSTM runs -- which is 1 second of a quantity that
            # regenerates at 0.35/decision, so the staleness is far inside the
            # head's own ~0.9 MAE.
            opp = float(self.net.predict_opp_elixir(self.hidden[0])[0])                 if self.use_opp_elixir else None
            allow = torch.tensor([self.gate.mask(o, costs, opp)],
                                 dtype=torch.bool, device=self.device)
            blocked = int((mask & ~allow).sum())
            self.stats["gate_blocked"] += blocked
            # The no-op column is legal in both, so this can only remove card
            # slots -- it can never produce an all-illegal row.
            mask = mask & allow

        feats, embeds, sp = net.extract_features(t)
        logits, _, _, _, hidden = net.step_lstm_and_card(feats, self.hidden, mask)
        self.hidden = hidden
        hx = hidden[0]

        slot = int(logits.argmax(-1).item())
        hand = net.hand_card_ids(t)[0].tolist()
        affordable = mask[0].tolist()
        self._cannon_cd = max(0, self._cannon_cd - 1)

        # --- (b) tactical INITIATION ---------------------------------------
        # Only ever converts a no-op into a play, never overrides a play the
        # commander actually wanted: the commander is better than the advisor at
        # knowing WHEN, and this exists solely to reach cards it has stopped
        # considering at all.
        if self.initiate and self.use_advisor and slot == net.hand_size:
            fired = self._try_initiate(o, hand, affordable)
            if fired is not None:
                slot = fired

        # --- (c) placement ---------------------------------------------------
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

    # ------------------------------------------------------------------ util
    def _advisor_cell(self, o, card_id):
        # Keyed on the card's own mapped surface rather than an if/elif chain
        # ending in a catch-all else -- see ADVISOR_SURFACE. __init__ has
        # already rejected any override card missing from it, so this cannot
        # KeyError for a card that actually reaches here.
        surface = self.ADVISOR_SURFACE[card_id]
        if surface == "spell":
            x, y, _ = tactics.best_spell_cell(o, legal=self._legal[card_id])
        else:
            x, y, _ = tactics.best_building_cell(o, legal=self._legal[card_id])
        return x, y

    def _try_initiate(self, o, hand, affordable):
        """Return a hand slot to play, or None. Solvency is already in `affordable`."""
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
        # A third "initiate the Giant last, once nothing more urgent wanted the
        # elixir" branch lived here behind an initiate_giant flag. Removed with
        # the rest of the Giant path on 2026-08-19 -- see the module-level note.
        return None
