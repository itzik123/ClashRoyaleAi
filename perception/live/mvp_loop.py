"""Minimal end-to-end loop: screen -> GameState -> decision -> tap.

Two policies. `--policy neural` runs the trained network: `perception_encoder`
turns a GameState into the observation and `NeuralPolicy` steps the net.
`--policy scripted` (the default) is a hand-written rule that isolates the
integration from the policy: capture, detector, adapter, tile->screen
conversion and tap are real in both modes, so a fault that reproduces under
`scripted` is in the pipeline, not the network.

The loop exists to surface integration faults invisible inside any one
component: a tap on the wrong tile yields a valid GameState showing a unit
nobody intended; the detector competes with capture for CPU; the whole chain
must fit the policy's 1 Hz.

Dry run is the default. Acting on a real match needs `--act`.
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# The engine binding lives in python_ai/ and is not importable by default;
# live/unit_to_card.py needs it for the card registry.
_ENGINE = _ROOT.parent / "python_ai"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
# ...and the repo root, so `python_ai.*` resolves. The python_ai/ entry stays:
# clash_royale_env is an unpackaged .pyd inside it.
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from capture.window import WindowSource  # noqa: E402
from clashroyalebuildabot.constants import (  # noqa: E402
    SCREENSHOT_HEIGHT,
    SCREENSHOT_WIDTH,
)
from clashroyalebuildabot.detectors.detector import Detector  # noqa: E402
from clashroyalebuildabot.namespaces.cards import Cards  # noqa: E402
from live.action_gate import MAX_STALENESS_MS, ActionGate  # noqa: E402
from live.actuator import AdbActuator  # noqa: E402
from live.adapter import build_game_state  # noqa: E402
from live.elixir_ledger import ElixirLedger  # noqa: E402
from live.hand_tracker import HandTracker  # noqa: E402
from live.match_state import MatchState  # noqa: E402
from live.pipeline import PerceptionWorker, Stages  # noqa: E402
from live.placement_confirm import PlacementConfirmer  # noqa: E402

def _training_deck_ids() -> list[int]:
    """`gym_wrapper.DEFAULT_DECK`, parsed from source rather than imported.

    Importing it would pull in gymnasium and torch, which perception's venv
    does not carry. Parsing keeps one source of truth without the dependency.
    Strict: a DEFAULT_DECK that is missing or not a plain list of int literals
    raises, since a guessed deck means playing cards the policy has never seen.
    """
    import ast  # noqa: PLC0415

    source_path = (Path(__file__).resolve().parent.parent.parent
                   / "python_ai" / "envs" / "gym_wrapper.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "DEFAULT_DECK"
                   for t in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
            raise RuntimeError(
                f"DEFAULT_DECK in {source_path} is not a list of int literals; "
                "it can no longer be read without importing gym_wrapper.")
        return value
    raise RuntimeError(f"no DEFAULT_DECK assignment found in {source_path}")


def _deck_from_training() -> list:
    """The live deck, derived from the deck the policy was trained on.

    The loop is internally consistent whatever deck it holds, so a mismatch is
    invisible at runtime: the affordability mask, hand one-hots and
    card-conditioned placement would all be computed for the wrong cards.
    Raises on an unmappable card rather than returning a short deck, which
    would leave a slot the agent believes it holds permanently unplayable.
    """
    from live.unit_to_card import hand_card_id_for  # noqa: PLC0415

    DEFAULT_DECK = _training_deck_ids()

    by_sim_id = {}
    for card in vars(Cards).values():
        name = getattr(card, "name", None)
        if not name or name == "blank":
            continue
        sim_id = hand_card_id_for(name)
        # First writer wins: Evolutions reuse their base card's name, and the
        # base entry is the one the registry hands back.
        by_sim_id.setdefault(sim_id, card)

    deck, missing = [], []
    for sim_id in DEFAULT_DECK:
        card = by_sim_id.get(sim_id)
        if card is None:
            missing.append(sim_id)
        else:
            deck.append(card)
    if missing:
        raise RuntimeError(
            f"no CRBAB card image for simulator id(s) {missing} in "
            f"DEFAULT_DECK. The live loop cannot read a hand it has no "
            f"template for -- add the icon under "
            f"clashroyalebuildabot/images/cards/ and a Cards entry for it.")
    return deck


DECK = _deck_from_training()

# A fixed defensive tile in our own half, in engine coordinates: the point is
# to exercise placement, not to play well.
DEFAULT_TILE = (9, 8)

# Below this the loop is acting on a board that has already changed.
DECISION_HZ = 1.0

# Imported from action_gate so the threshold that reports staleness and the one
# that refuses to act on it cannot drift apart.


# The longest the loop will hold a decision for a fresher board: 150 ms on a
# 1000 ms tick, and only while the phase lock converges.
FRESHNESS_WAIT_CAP_S = 0.15

# Slack on the predicted arrival, so a board landing slightly late is still
# caught.
FRESHNESS_WAIT_SLACK_S = 0.03


def wait_for_fresher(worker, snap):
    """Hold the decision briefly if a newer board is about to land.

    The producer publishes at its own rate and the loop samples at 1 Hz on an
    unrelated phase, so the board in hand has usually sat finished for part of
    a producer period, ageing the world model for nothing. Waiting for the next
    board shrinks the gap between capture and the card landing, which decides
    whether a counter meets a push or arrives behind it.

    Waiting delays the action, so it is only worth it when the board in hand is
    stale and the next is imminent: within FRESHNESS_WAIT_CAP_S.

    Returns (snapshot, waited_ms).
    """
    period = worker.period
    if period is None:
        # Not enough history to predict an arrival; act now.
        return snap, 0.0

    started = time.perf_counter()
    eta = period - snap.sat_ms(started) / 1000.0
    if not 0.0 < eta <= FRESHNESS_WAIT_CAP_S:
        return snap, 0.0

    deadline = started + min(eta + FRESHNESS_WAIT_SLACK_S, FRESHNESS_WAIT_CAP_S)
    while time.perf_counter() < deadline:
        newer = worker.latest()
        if newer is not None and newer.index != snap.index:
            return newer, (time.perf_counter() - started) * 1000.0
        time.sleep(0.004)
    return snap, (time.perf_counter() - started) * 1000.0


def deck_costs(deck) -> tuple[tuple[float, ...], list[str]]:
    """The distinct elixir costs in `deck`, and any warnings.

    The ledger explains an elixir drop by decomposing it into card costs, so
    its cost table must be the deck's: a cost missing from it makes drops that
    match no combination, and those placements vanish from the ledger silently.

    Costs come from the engine registry, which fills the observation's cost
    scalars and so is what `affordability_mask` gates on. CRBAB's own per-card
    cost is cross-checked and a disagreement reported.
    """
    import clash_royale_env as engine  # noqa: PLC0415

    from live.unit_to_card import (  # noqa: PLC0415
        UNKNOWN_CARD_SIM_ID,
        hand_card_id_for,
    )

    costs: set[float] = set()
    warnings: list[str] = []
    for card in deck:
        crbab = float(card.cost)
        sim_id = hand_card_id_for(card.name)
        engine_cost = None
        if sim_id != UNKNOWN_CARD_SIM_ID:
            try:
                engine_cost = float(engine.get_card_info(sim_id)["cost"])
            except Exception:                               # noqa: BLE001
                engine_cost = None
        if engine_cost is None:
            # Fall back rather than drop it: a missing cost removes a whole
            # card's worth of explanations from the table.
            costs.add(crbab)
            warnings.append(f"{card.name}: not in the engine registry, "
                            f"using CRBAB's cost {crbab:.0f}")
        else:
            costs.add(engine_cost)
            if engine_cost != crbab:
                warnings.append(f"{card.name}: engine says {engine_cost:.0f}, "
                                f"CRBAB says {crbab:.0f}")
    return tuple(sorted(costs)), warnings


def deck_cost_by_sim_id(deck) -> dict[int, float]:
    """simulator card id -> elixir cost, for HandTracker.

    The ledger decomposes a drop and needs the distinct costs; the tracker
    answers "which card in hand cost 4?" and needs them per card. Both read the
    engine registry, so they cannot disagree.
    """
    import clash_royale_env as engine  # noqa: PLC0415

    from live.unit_to_card import (  # noqa: PLC0415
        UNKNOWN_CARD_SIM_ID,
        hand_card_id_for,
    )

    out: dict[int, float] = {}
    for card in deck:
        sim_id = hand_card_id_for(card.name)
        if sim_id == UNKNOWN_CARD_SIM_ID:
            continue
        try:
            out[sim_id] = float(engine.get_card_info(sim_id)["cost"])
        except Exception:                                   # noqa: BLE001
            out[sim_id] = float(card.cost)
    return out


def hand_cost(gs, slot: int) -> float | None:
    """What the card in `slot` costs, from the engine registry. None for an
    unreadable slot, so an unknown card cannot debit the wrong amount.
    """
    try:
        import clash_royale_env as engine  # noqa: PLC0415

        card_id = gs.my_hand[slot]
        if card_id is None or card_id < 0:
            return None
        return float(engine.get_card_info(card_id)["cost"])
    except Exception:                                   # noqa: BLE001
        return None


@dataclass
class Decision:
    slot: int | None
    tile: tuple[int, int]
    why: str


class ScriptedPolicy:
    """Play the cheapest ready card, on a cooldown. A placeholder with a
    deliberately obvious name.
    """

    def __init__(self, cooldown_s: float = 3.0, tile=DEFAULT_TILE):
        self.cooldown_s = cooldown_s
        self.tile = tile
        self._last_play = -1e9

    def decide(self, gs, ready, now: float) -> Decision:
        if now - self._last_play < self.cooldown_s:
            return Decision(None, self.tile, "cooldown")
        if not ready:
            return Decision(None, self.tile, "nothing ready")
        slot = int(min(ready))
        self._last_play = now
        return Decision(slot, self.tile, f"cheapest ready slot {slot}")


class NeuralPolicy:
    """The trained agent, reading a real screen.

    `perception_encoder` builds the observation it was trained on, and the
    masks are the net's own methods reading that vector, so affordability and
    placement legality are computed exactly as in training.

    The LSTM carries the match's history: it must persist across frames and
    reset only when a new match begins. Resetting every frame would silently
    reduce the policy to a reflex one.
    """

    def __init__(self, checkpoint: Path, deck_ability_slots: int = 0,
                 tactical: bool = True, reserve: float = 4.0):
        import torch  # noqa: PLC0415

        from python_ai.models import perception_encoder  # noqa: PLC0415
        from python_ai.models.net import MicroRoyaleNet  # noqa: PLC0415

        self.torch = torch
        self.encoder = perception_encoder
        self.net = MicroRoyaleNet(num_ability_slots=deck_ability_slots)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = blob["model"] if "model" in blob else blob
        # Not strict: a checkpoint predating the high-resolution placement
        # branch (`place_hires`/`place_ctx_hi`) lacks those tensors, and their
        # final conv is zero-initialised, so the net computes exactly the
        # pre-branch function. strict=False alone would also swallow tensors
        # this net has no home for, so missing keys are checked against the one
        # allowed prefix and anything unexpected is fatal.
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        allowed = tuple(k for k in missing
                        if k.startswith(("place_hires", "place_ctx_hi")))
        if unexpected or set(missing) - set(allowed):
            raise RuntimeError(
                f"checkpoint does not match this network -- unexpected "
                f"{list(unexpected)}, missing {sorted(set(missing) - set(allowed))}")
        if missing:
            print(f"checkpoint predates the hi-res placement branch; "
                  f"{len(missing)} tensor(s) left at their zero init "
                  f"(exactly the pre-branch behaviour)")
        self.net.eval()
        self.episodes = int(blob.get("episodes_completed", -1))
        self._hx = None
        self._cx = None
        self._was_in_game = False
        # Which placement cells the actuator can reach (see
        # actuator.engine_row_is_tappable). A constant of the screen mapping,
        # asserted non-empty so a geometry change masking every cell fails
        # loudly instead of making the agent a permanent no-op.
        from live.actuator import engine_row_is_tappable  # noqa: PLC0415
        cells = self.net.placement_cells
        width = self.net.board_width
        self._tappable_cells = torch.tensor(
            [engine_row_is_tappable(c // width) for c in range(cells)],
            dtype=torch.bool).unsqueeze(0)
        n_ok = int(self._tappable_cells.sum())
        if n_ok == 0:
            raise RuntimeError(
                "no placement cell is tappable -- the tile geometry and the "
                "engine frame disagree completely; check TILE_Y_OFFSET.")
        print(f"placement: {n_ok}/{cells} cells reachable by the actuator "
              f"({cells - n_ok} engine rows have no detector row)")

        # Per-card legality from the engine's own predicate. The net's
        # placement_mask grants a troop every own-half cell; the engine also
        # rejects the back-row dead zone and tower footprints, and those are
        # placements the real game refuses. The predicate is independent of
        # board state, so this is a constant table per card.
        #
        # DECK holds CRBAB card objects, not simulator ids; hand_card_id_for
        # bridges the two (mapping/card_map.json).
        import clash_royale_env  # noqa: PLC0415
        from live.unit_to_card import (  # noqa: PLC0415
            UNKNOWN_CARD_SIM_ID,
            hand_card_id_for,
        )
        sim_ids = [hand_card_id_for(c.name) for c in DECK]
        sim_ids = [i for i in sim_ids if i != UNKNOWN_CARD_SIM_ID]
        oracle = clash_royale_env.ClashRoyaleEnv(sim_ids, sim_ids, 100)
        oracle.reset()
        width = self.net.board_width
        self._legal_by_card = {}
        for card_id in set(sim_ids):
            self._legal_by_card[card_id] = torch.tensor(
                [oracle.is_valid_placement(card_id, float(c % width), float(c // width), 0)
                 for c in range(cells)], dtype=torch.bool).unsqueeze(0)
        counts = sorted(int(m.sum()) for m in self._legal_by_card.values())
        print(f"placement: engine-legal cells per card {counts} "
              f"(of {cells}; unmasked would be {cells})")

        # --- tactical officer ---
        # For Cannon, Fireball and Giant the learned placement is worse than a
        # random cell, so tactics.py, scored against the engine's own
        # accounting, chooses the cell.
        #
        # The advisor gets the same masks the sampled path uses, intersected:
        # engine legality and actuator reachability. The engine mask alone
        # would let it propose engine row 0, whose tap lands below the arena
        # and is dropped.
        self._tactical = tactical
        self._advisor_legal = {
            cid: (m & self._tappable_cells)[0].numpy().astype(bool)
            for cid, m in self._legal_by_card.items()
        }
        if tactical:
            from python_ai.advisors import tactics  # noqa: PLC0415
            self._tactics = tactics
            self._gate = tactics.SolvencyGate(reserve=reserve)
            self._override_ids = {tactics.CANNON_ID, tactics.FIREBALL_ID,
                                  tactics.GIANT_ID}
            have = sorted(self._override_ids & set(self._advisor_legal))
            print(f"tactical: advisor ON for card ids {have}, "
                  f"solvency gate reserve {reserve}")
        else:
            self._tactics = None
            self._gate = None
            self._override_ids = set()

    def reset_hidden(self) -> None:
        self._hx = self.torch.zeros(1, 256)
        self._cx = self.torch.zeros(1, 256)

    @staticmethod
    def _card_id_for_slot(gs, slot: int):
        """The simulator card id in this hand slot, or None.

        Read from the perceived hand, not DECK order: the hand cycles, and a
        blank or misread slot must be representable.
        """
        hand = getattr(gs, "my_hand", ()) or ()
        if 0 <= slot < len(hand):
            card_id = hand[slot]
            return int(card_id) if card_id is not None and card_id >= 0 else None
        return None

    def decide(self, gs, ready, now: float) -> Decision:
        torch = self.torch
        if self._hx is None:
            self.reset_hidden()

        obs = torch.as_tensor(self.encoder.encode(gs)).unsqueeze(0)
        with torch.no_grad():
            feats, embeds, smap = self.net.extract_features(obs)
            card_mask = self.net.affordability_mask(obs)
            if self._gate is not None:
                # Veto spends that would leave us unable to answer, but only
                # while nothing is attacking and only up to what the opponent
                # could punish with.
                from python_ai.advisors import tactics  # noqa: PLC0415
                o = obs[0].numpy()
                # From the net, not sliced here:
                # MicroRoyaleNet.hand_costs_from_obs.
                costs = self.net.hand_costs_from_obs(obs)[0].tolist()
                # Opponent elixir is reconstructed as the teacher does it:
                # start + regen*t - their observed spend. It assumes a flat
                # regen rate, so it under-reads after 2:00 in the phased match.
                # The gate is off by default (`USE_SOLVENCY_GATE` ships False).
                opp = float(tactics.opp_elixir_estimate(o))
                allow = torch.tensor([self._gate.mask(o, costs, opp)],
                                     dtype=torch.bool)
                card_mask = card_mask & allow
            logits, _a1, _a2, _v, (self._hx, self._cx) = self.net.step_lstm_and_card(
                feats, (self._hx, self._cx), card_mask)
            card = torch.distributions.Categorical(logits=logits).sample()
            slot = int(card.item())
            if slot >= self.net.hand_size:
                # The last column is the always-legal no-op; "hold elixir" is a
                # real decision.
                return Decision(None, DEFAULT_TILE, "no-op")
            # --- tactical override ---
            # The advisor already received the intersection of the masks below,
            # which exist to filter the network's distribution, so this returns
            # immediately.
            card_id = self._card_id_for_slot(gs, slot)
            if card_id in self._override_ids:
                legal = self._advisor_legal.get(card_id)
                if legal is not None and legal.any():
                    o = obs[0].numpy()
                    if card_id == self._tactics.FIREBALL_ID:
                        ax, ay, _ = self._tactics.best_spell_cell(o, legal=legal)
                    elif card_id == self._tactics.GIANT_ID:
                        ax, ay, _ = self._tactics.best_giant_cell(o, legal=legal)
                    else:
                        ax, ay, _ = self._tactics.best_building_cell(o, legal=legal)
                    return Decision(slot, (int(ax), int(ay)),
                                    f"advisor slot {slot} card {card_id} "
                                    f"-> ({int(ax)},{int(ay)})")

            placement = self.net.placement_given_card(
                self._hx, embeds, card, obs, smap)
            placement = placement.masked_fill(
                ~self.net.placement_mask(obs, card), float("-inf"))
            # Second mask, live-only: the net's placement_mask follows the
            # engine's bounds, which include rows this screen mapping cannot
            # reach (engine row 0 taps below the arena and the game drops it).
            # It belongs to the actuator, not the game; narrowing the training
            # action space would change what the policy learns.
            placement = placement.masked_fill(
                ~self._tappable_cells.to(placement.device), float("-inf"))
            # Third mask: the engine's isValidPlacement for this card, keyed on
            # card id because slot contents cycle. An unrecognised id falls
            # through unmasked: perception being unsure is no reason to refuse
            # a placement the game may accept.
            legal = self._legal_by_card.get(self._card_id_for_slot(gs, slot))
            if legal is not None:
                placement = placement.masked_fill(
                    ~legal.to(placement.device), float("-inf"))
            if not torch.isfinite(placement).any():
                # Every cell masked: wait. Sampling an all -inf row would
                # return cell 0 and tap somewhere arbitrary.
                return Decision(None, DEFAULT_TILE, "no-op (no legal cell)")
            cell = torch.distributions.Categorical(logits=placement).sample()
            x, y = self.net.cell_to_xy(cell)
        # cell_to_xy returns engine coordinates, which engine_tile_centre
        # expects.
        return Decision(slot, (int(x.item()), int(y.item())),
                        f"net slot {slot} -> ({int(x.item())},{int(y.item())})")

    def on_screen_change(self, in_game: bool) -> None:
        """Reset the recurrent state when a match starts, never mid-match."""
        if in_game and not self._was_in_game:
            self.reset_hidden()
        self._was_in_game = in_game



def _sim_id_to_detector_name(sim_id: int) -> str | None:
    """Simulator card id -> the detector's own class name, or None if unknown.

    Slugged from the ENGINE's card name rather than hand-typed, and every deck
    card is checked against CRBAB's `Cards` namespace at first use, so a deck
    change that breaks the mapping raises here instead of silently handing the
    confirmation oracle an empty expectation (which reads as "the placement
    never landed").
    """
    if sim_id is None or int(sim_id) < 0:
        return None
    import clash_royale_env as _E  # noqa: PLC0415

    try:
        engine_name = _E.get_card_info(int(sim_id))["name"]
    except Exception:
        return None
    slug = engine_name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
    if slug not in _detector_card_names():
        raise RuntimeError(
            f"card {sim_id} ({engine_name!r}) slugs to {slug!r}, which the "
            f"detector does not know. The confirmation oracle would silently "
            f"expect no units for it.")
    return slug


@functools.lru_cache(maxsize=1)
def _detector_card_names() -> frozenset[str]:
    import dataclasses  # noqa: PLC0415

    from clashroyalebuildabot.namespaces.cards import Cards  # noqa: PLC0415

    return frozenset(getattr(Cards, f.name).name
                     for f in dataclasses.fields(Cards))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--act", action="store_true",
                    help="actually tap. Without it the loop only watches.")
    ap.add_argument("--window", default="BlueStacks App Player")
    ap.add_argument("--serial", action="store_true",
                    help="run perception inline instead of on its own thread. "
                         "The original design, kept for comparison: it pins "
                         "the decision RATE to the detector's LATENCY, "
                         "measured at 0.48 Hz on a quiet machine.")
    ap.add_argument("--ignore-staleness", action="store_true",
                    help="act on a board older than the staleness budget. "
                         "Idempotency still holds -- this only relaxes the age "
                         "check, which is the one that stops the agent playing "
                         "into a board the match has already moved past.")
    ap.add_argument("--policy", choices=("scripted", "neural"),
                    default="scripted",
                    help="scripted exercises the joints; neural is the agent")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).resolve().parents[2] / "python_ai"
                    / "model_weights_live.pth",
                    help="model_weights_selfplay.pth was deleted in the "
                         "2026-08-19 cleanup, so this default pointed at a "
                         "missing file and --policy neural raised "
                         "FileNotFoundError before a single frame was read")
    ap.add_argument("--no-tactical", action="store_true",
                    help="disable the advisor override and solvency gate, so the "
                         "network alone decides where -- the A/B control arm")
    ap.add_argument("--reserve", type=float, default=4.0,
                    help="elixir the solvency gate keeps in hand while nothing "
                         "is attacking")
    ap.add_argument("--confirm-log", type=Path, default=None,
                    help="write one JSON line per issued placement, with both "
                         "oracles' verdicts and the pixel actually tapped")
    ap.add_argument("--ensure-match", action="store_true",
                    help="navigate into a Training Camp match before starting. "
                         "The run still ends on --seconds, NOT when the match "
                         "does, so size --seconds to cover the match (~200 s "
                         "plus loading) or the loop stops mid-game")
    ap.add_argument("--frames", type=Path, default=None,
                    help="replay a tools/record_match.py directory instead of "
                         "capturing live. The rest of the chain is identical, "
                         "so this is the integration test the live path cannot "
                         "be: deterministic, repeatable, and runnable with no "
                         "emulator.")
    ap.add_argument("--frames-fps", type=float, default=None,
                    help="treat --frames as a CONSTANT-RATE dump at this fps "
                         "(no manifest.json). Only correct for stills "
                         "extracted from a CFR video.")
    ap.add_argument("--hand", choices=("auto", "deck_pool", "crbab"),
                    default="auto",
                    help="hand classifier: 'deck_pool' requires a pool "
                         "covering this deck and fails if absent; 'auto' "
                         "falls back to crbab with a warning; 'crbab' forces "
                         "the incumbent")
    ap.add_argument("--hand-pool", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "config" / "templates" / "deck_pool_hog26")
    args = ap.parse_args()

    if args.frames and args.act:
        raise SystemExit("--act makes no sense while replaying a recording")

    print(f"{'ACTING - will place real cards' if args.act else 'DRY RUN - watching only'}")
    if args.frames:
        if args.frames_fps:
            # A constant-rate dump, declared as one. RecordingSource carries
            # real per-frame stamps and suits tools/record_match.py output,
            # which is variable-rate. A directory extracted from a CFR video
            # has no manifest and is genuinely constant-rate, so synthesised
            # timestamps are correct, but only when stated: applied silently to
            # a variable-rate recording they would bake drift into the clock
            # templates. Hence a flag, not a fallback.
            from capture.frames import FrameDirSource  # noqa: PLC0415
            source = FrameDirSource(args.frames, fps=args.frames_fps)
            print(f"replaying {source.frame_count} stills from {args.frames} "
                  f"at a synthesised {args.frames_fps} fps")
        else:
            from capture.frames import RecordingSource  # noqa: PLC0415
            source = RecordingSource(args.frames)
            print(f"replaying {source.frame_count} frames from {args.frames}")
    else:
        source = WindowSource(args.window)
    detector = Detector(DECK)

    # Swap in the deck-restricted hand classifier at the composition root.
    # `Detector.run` calls `self.card_detector.run(image) -> (cards, ready)`
    # and DeckHandDetector answers in the same shape, so no fork of the
    # vendored detector is needed.
    #
    # The gain is fewer impossible cycle transitions: the stock reader confuses
    # cards within the deck consistently, which produces a hand history the
    # 8-card FIFO rules out. A pool built for another deck cannot be used (it
    # would map every unseen card onto its nearest eight), so that case falls
    # back to the stock reader, loudly.
    if args.hand != "crbab":
        try:
            from live.deck_hand import DeckHandDetector  # noqa: PLC0415
            detector.card_detector = DeckHandDetector(list(DECK), args.hand_pool)
            print(f"hand classifier: deck_pool ({args.hand_pool})")
        except Exception as exc:  # noqa: BLE001
            if args.hand == "deck_pool":
                raise
            print("*" * 70)
            print("WARNING: falling back to the CRBAB hand classifier.")
            print(f"  {exc}")
            print("  Identity will be materially worse -- see tools/bench_hand.py.")
            print("*" * 70)
    else:
        print("hand classifier: crbab (incumbent)")

    # Recorded in the run's output: a timing log that does not name its
    # execution provider cannot be compared with another.
    print(f"execution provider: "
          f"{detector.unit_detector.sess.get_providers()[0]}")
    if args.ensure_match:
        # tools/ is not on the path. Added here so a run without the flag does
        # not depend on the navigation code.
        if str(_ROOT / "tools") not in sys.path:
            sys.path.insert(0, str(_ROOT / "tools"))
        from match_nav import ensure_in_match  # noqa: PLC0415
        ensure_in_match(detector)

    actuator = AdbActuator(dry_run=not args.act)
    if args.act:
        print(f"actuator backend: {actuator.backend}")
    if args.policy == "neural":
        # Copied first: a live training run rewrites this file, and reading it
        # mid-write loads a truncated checkpoint.
        import shutil, tempfile  # noqa: PLC0415
        frozen = Path(tempfile.gettempdir()) / "mvp_policy_snapshot.pth"
        shutil.copy2(args.checkpoint, frozen)
        policy = NeuralPolicy(frozen, tactical=not args.no_tactical,
                              reserve=args.reserve)
        print(f"policy: TRAINED NET from {args.checkpoint.name} "
              f"(episode {policy.episodes})"
              f"{'' if args.no_tactical else ' + tactical officer'}")
    else:
        policy = ScriptedPolicy()
        print("policy: scripted placeholder")
    costs, cost_warnings = deck_costs(DECK)
    for warning in cost_warnings:
        print(f"  !! card cost: {warning}")
    print(f"deck costs: {', '.join(f'{c:.0f}' for c in costs)}")
    # Printed in full: the loop is consistent whatever deck it holds, so this
    # line is what makes "live matches training" auditable.
    print("deck (derived from gym_wrapper.DEFAULT_DECK): "
          + ", ".join(c.name for c in DECK))
    ledger = ElixirLedger(costs=costs)

    # The hand comes from the cycle, not the screen. The per-frame reading
    # changes far more often than cards are played, and most of its slot
    # changes have no elixir drop behind them; the cycle is a strict 8-slot
    # FIFO, so the play history, which is read well, determines the hand
    # exactly. See hand_tracker.py.
    tracker = HandTracker(deck=tuple(_training_deck_ids()),
                          costs=deck_cost_by_sim_id(DECK))
    gate = ActionGate(enforce_staleness=not args.ignore_staleness)
    # Independent of the ledger: the ledger says whether the elixir trace
    # reconciled, this says whether a unit appeared. Only both being silent
    # means the game refused it.
    confirmer = PlacementConfirmer()

    print(f"capture {source.size[0]}x{source.size[1]}   "
          f"decision rate {DECISION_HZ} Hz   for {args.seconds:.0f}s\n")
    print("  t     screen     units  elix  spent  hand                    "
          "decision                 ms")

    # Timed by stage, so a slow run says which stage grew: execution provider,
    # capture or adapter.
    stages = Stages()

    # The match gate. Updated only here, on the thread that owns perception, so
    # there is a single writer; the decision loop only reads `match.in_match`.
    # It is debounced, unlike the raw `screen.name == "in_game"`, which let one
    # misread frame admit lobby frames to the readers or end the match early.
    match = MatchState()
    # A new battle deals a fresh opening hand; the previous match's FIFO no
    # longer applies.
    match.on_change(lambda in_match: tracker.reset() if in_match else None)

    def perceive(frame):
        """capture-frame -> (State, GameState). Runs on the thread that owns
        perception: the worker when pipelined, the loop when --serial.
        """
        t = time.perf_counter()
        native = Image.fromarray(frame.image[:, :, ::-1])       # BGR -> RGB
        small = native.resize((SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT), Image.LANCZOS)
        t = stages.time("1 decode+resize", t)
        state = detector.run(small)
        t = stages.time("2 detector.run", t)
        if state is None:
            return None, None
        in_match = match.update(state.screen.name)
        plays_before = len(ledger.plays)
        if in_match:
            # The reading's own capture time, not now(): the ledger models
            # regeneration between samples, and a frame of latency is a quarter
            # of an elixir.
            ledger.update(state.numbers.elixir.number,
                          now=frame.wall_time_ms / 1000.0)
        gs, _report = build_game_state(
            state, np.array(native), np.array(small),
            frame_index=frame.index, wall_time_ms=frame.wall_time_ms,
            my_elixir_spent=ledger.spent)

        # The hand is replaced by the tracker's, not merged. The screen reading
        # still feeds the tracker as seed, tie-breaker between equal-cost cards
        # and desync detector, but the policy sees the FIFO's answer. Only in a
        # match: the tracker advances on elixir drops.
        if in_match:
            tracker.update(tuple(getattr(gs, "my_hand", ()) or ()),
                           ledger.plays[plays_before:])
            # Until the consensus window fills, `as_tuple()` is four UNKNOWNs,
            # which would mask every slot and freeze the agent for the first
            # ~25 frames; the screen reading stands until then.
            if tracker.seeded:
                gs = replace(gs, my_hand=tracker.as_tuple())
        # Optimistic debit. The bar being read is ~1 s old and the last tap
        # needs ~0.9 s more to land, so committed cards still look affordable
        # and the same elixir gets spent several times over. Subtracting what
        # is promised but not yet seen leave the bar is the better estimate,
        # and `my_elixir` is what affordability_mask gates on.
        owed = ledger.unconfirmed_cost
        if owed:
            gs = replace(gs, my_elixir=max(0.0, gs.my_elixir - owed))
        stages.time("3 build_game_state", t)
        return state, gs

    replay = iter(source.sample_every(DECISION_HZ)) if args.frames else None
    pipelined = replay is None and not args.serial
    worker = None
    if pipelined:
        # Replay is never pipelined: it is the deterministic integration test,
        # and a thread racing a finite recording would reproduce differently
        # each run.
        worker = PerceptionWorker(source, detector, perceive)
        worker.start()
        print("perception: BACKGROUND THREAD (decisions use the newest board)")
    else:
        print("perception: inline" + (" [replay]" if replay else " [--serial]"))
    print()
    print("  t     screen     units  elix  spent  age   hand"
          "                     decision               ms")

    t0 = time.perf_counter()
    next_due = t0
    last_observed = None
    n = slow = stale = 0
    ages = []
    waits: list[float] = []
    try:
        while time.perf_counter() - t0 < args.seconds:
            now = time.perf_counter()
            if replay is None:
                if now < next_due:
                    time.sleep(min(0.02, next_due - now))
                    continue

            step = time.perf_counter()
            age_ms = 0.0
            board_index = n
            if pipelined:
                snap = worker.latest()
                if snap is None:
                    if not worker.alive:
                        print(f"  perception thread DIED: {worker.last_error!r}")
                        break
                    continue
                snap, waited_ms = wait_for_fresher(worker, snap)
                if waited_ms:
                    waits.append(waited_ms)
                step = time.perf_counter()
                state, gs = snap.state, snap.game_state
                age_ms = snap.age_ms(step)
                board_index = snap.index
            else:
                frame = (next(replay, None) if replay is not None
                         else source.read_new(timeout_s=1.0))
                if frame is None:
                    print("  (no frame)")
                    if replay is not None:
                        break
                    continue
                state, gs = perceive(frame)
                if state is None:
                    continue
                board_index = frame.index

            # Phase-locked: the next tick is measured from the decision that
            # happened, so a wait shifts the cadence instead of being repaid by
            # a short interval. Once aligned to the producer the wait stops
            # triggering.
            if replay is None:
                next_due = time.perf_counter() + 1.0 / DECISION_HZ

            # Fold this board into every pending placement, once per board: the
            # loop can sample the same published board twice when the producer
            # is slower than 1 Hz, and counting it twice would inflate
            # `observations`.
            if board_index != last_observed:
                confirmer.observe(gs, time.perf_counter())
                last_observed = board_index

            # The debounced gate, not `state.screen.name`: this value starts
            # the policy's recurrent state and releases taps.
            in_game = match.in_match
            if hasattr(policy, "on_screen_change"):
                policy.on_screen_change(in_game)
            # The policy steps every tick regardless of the gate: it was
            # trained stepping once per second, and skipping steps would change
            # the LSTM's cadence. Only the tap is gated.
            decision = (policy.decide(gs, state.ready, now) if in_game
                        else Decision(None, DEFAULT_TILE, "not in game"))
            if decision.slot is not None:
                verdict = gate.check(board_index, age_ms)
                if verdict:
                    _card_tap, tile_tap = actuator.play(decision.slot,
                                                        *decision.tile)
                    # Recorded only after the tap returns, so an adb failure
                    # leaves the board available to retry.
                    gate.record(board_index)
                    # The name comes from the hand the policy read
                    # (`gs.my_hand`), not CRBAB's stock icon read of
                    # `state.cards[1:5]`, which disagrees with the tracked hand
                    # on about half of placements. The confirmation oracle
                    # builds its target from this name, so a wrong one has it
                    # hunting for a unit never played. The stock crop is used
                    # only where the tracked slot is unreadable.
                    hand_names = [c.name for c in state.cards[1:5]]
                    tracked_name = _sim_id_to_detector_name(
                        gs.my_hand[decision.slot]
                        if decision.slot < len(gs.my_hand) else -1)
                    rec = confirmer.issue(
                        gs, slot=decision.slot,
                        card_name=tracked_name or hand_names[decision.slot],
                        card_sim_id=(gs.my_hand[decision.slot]
                                     if decision.slot < len(gs.my_hand) else -1),
                        tile=tuple(decision.tile),
                        tap=(tile_tap.x, tile_tap.y),
                        now=time.perf_counter(),
                        owed=ledger.unconfirmed_cost)
                    # Ground truth: we know exactly which card and what it
                    # cost, which the bar cannot recover once 3 and 4 quantise
                    # to the same drop. Only a tap actually sent spends elixir:
                    # in dry run nothing reaches the game, and on a replay the
                    # elixir being read is a human's.
                    cost = (None if actuator.dry_run
                            else hand_cost(gs, decision.slot))
                    if cost is not None:
                        # No timestamp: the ledger stamps it on its own frame
                        # clock, and a wall clock would mix time bases so
                        # nothing ever expires. Tagged with the confirmer's
                        # sequence number so the two verdicts can be
                        # cross-tabulated.
                        ledger.record_play(cost, tag=rec.seq)
                else:
                    gate.refuse(verdict.reason)
                    decision = Decision(None, decision.tile,
                                        f"[{verdict.reason}] {decision.why}")

            ms = (time.perf_counter() - step) * 1000.0
            slow += ms > 1000.0
            stale += age_ms > MAX_STALENESS_MS
            ages.append(age_ms)
            # [1:5], not [:4]: cards[0] is the "Next" preview. See
            # adapter._hand_ids.
            hand = ",".join(c.name[:6] for c in state.cards[1:5])
            print(f"  {now - t0:5.1f} {state.screen.name[:10]:<10} "
                  f"{len(gs.units):>5}  {gs.my_elixir:>4.0f}  {ledger.spent:>5.0f}  "
                  f"{age_ms:>4.0f}  {hand:<30} {decision.why:<22} {ms:>5.0f}")
            n += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if worker is not None:
            worker.stop()
        actuator.close()
        source.close()

    elapsed = time.perf_counter() - t0
    print(f"\n" + f"{n} decisions in {elapsed:.0f}s   {n / max(elapsed, 1e-9):.2f} Hz"
          f"   decision over 1000ms: {slow}/{max(n, 1)}")
    if worker is not None:
        print(f"perception thread: {worker.frames} boards "
              f"({worker.frames / max(elapsed, 1e-9):.2f} Hz), errors {worker.errors}")
        if ages:
            a = np.array(ages)
            print(f"board age: mean {a.mean():.0f} ms  median {np.median(a):.0f}"
                  f"  p95 {np.percentile(a, 95):.0f}  max {a.max():.0f}"
                  f"   over {MAX_STALENESS_MS:.0f} ms on {stale}/{max(n, 1)}")
        p = worker.period
        print(f"producer period: {p * 1000:.0f} ms" if p else
              "producer period: not measured")
        if waits:
            w = np.array(waits)
            print(f"freshness waits: {len(waits)}/{max(n, 1)} decisions, "
                  f"mean {w.mean():.0f} ms  max {w.max():.0f} ms")
        else:
            print("freshness waits: none (boards already arriving on phase)")
        print("\nproducer period, by stage (median):")
        print(worker.stages.report())
    print("\nperceive, by stage (median):")
    print(stages.report(total_key=None))
    print(f"\n{gate.summary()}")
    print(f"taps issued: {len(actuator.taps)}"
          f"{'' if args.act else ' (dry run - none sent)'}")
    if actuator.dropped or actuator.errors:
        # Dropped means a placement was still being tapped when the next was
        # chosen: rare at 1 Hz, and a sign the actuator is the bottleneck if
        # not.
        print(f"actuator: {actuator.dropped} dropped (still tapping), "
              f"{actuator.errors} errors"
              + (f" - last {actuator.last_error!r}" if actuator.last_error else ""))
    print(f"our elixir spent: {ledger.spent:.0f} over {ledger.cards} cards, "
          f"residual {ledger.residual:+.0f}, "
          f"{ledger.rejected} issued plays never confirmed")

    # `ledger.rejected` conflates a refused placement with a reconciliation
    # failure; the cross-tab separates them, and its listed rows are ground
    # truth for what the real game rejects.
    confirmer.close_all()
    confirmer.apply_ledger(ledger.confirmed_tags, ledger.rejected_tags)
    print()
    print(confirmer.summary())
    if args.confirm_log:
        confirmer.write_log(args.confirm_log)
        print(f"\nplacement log: {args.confirm_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
