"""Driving the simulator as a state estimator from perception events.

The simulator holds the state; perception says what was played, where and when.

Opponent placements are injected directly for team 1, bypassing hand, elixir
and legality, which is correct: the real game already validated the play. Our
own placements are harder. step() would also run the built-in heuristic
opponent, so they go through step_self_play() and GameManager::playCard, which
requires the card to be in the simulator's team-0 hand, and that hand is dealt
by the reset shuffle from a queue that cannot be read.

The fix is search. reset() costs 0.135 ms and its shuffle is uniform over the
70 hand-sets, which get_hand() can read; only the order of the 4 queued cards
(24 options) is hidden. So: draw many resets, keep those whose hand matches the
real opening hand, and carry them all forward as candidates. Each play
eliminates candidates that could not have held the card or that deal a
different incoming card than vision saw. After four confirmed deals the
survivors share the exact opening queue order (see `cycle_identified`), and
from then on they agree forever: both FIFOs receive the same plays and playCard
is deterministic.

The pool never shrinks to one object: slot arrangement is a free permutation
nothing observable depends on, so up to 24 behavioural duplicates survive. What
gets pinned down is the cycle.

playCard also checks affordability and legality, and the engine's elixir runs
~2% slow (timebase.py), so an aligned play can still be refused. A refused play
leaves get_hand() unchanged, so refusals are counted, separately from cycle
desyncs, in DriverReport.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from contracts import UNKNOWN_CARD_SIM_ID, PlacementEvent
from geometry import load_geometry

_PYTHON_AI = Path(__file__).resolve().parent.parent.parent / "python_ai"
if str(_PYTHON_AI) not in sys.path:
    sys.path.insert(0, str(_PYTHON_AI))
# ...and the repo root, so `python_ai.*` resolves. The python_ai/ entry stays:
# clash_royale_env is an unpackaged .pyd inside it.
if str(_PYTHON_AI.parent) not in sys.path:
    sys.path.insert(0, str(_PYTHON_AI.parent))


class SimUnavailableError(RuntimeError):
    """clash_royale_env could not be imported. Fatal here, unlike elsewhere in the
    package: driving the engine is this module's whole job.
    """


def _engine():
    try:
        import clash_royale_env  # noqa: PLC0415
    except Exception as exc:
        raise SimUnavailableError(
            "could not import clash_royale_env. The .pyd is built for Python "
            "3.11 and lives in python_ai/; run this with "
            "perception/.venv/Scripts/python.exe."
        ) from exc
    return clash_royale_env


# Hand-matching candidates the opening pool aims for, and the draw budget to
# find them. See SimDriver._draw_candidates for why 200.
DEFAULT_POOL_SIZE = 200
DEFAULT_RESET_DRAWS = 20000

# Recovery draw budget when the pool empties. Only 1/70 draws reach the history
# replay, so cost is dominated by cheap rejections; each correct candidate has
# probability 1/1680, so 50000 draws expects ~30.
REBUILD_WANTED = 4
REBUILD_MAX_DRAWS = 50000

# ClashEnv::MAX_BUILDING_HP, the building channels' normaliser. Read from the
# engine at runtime; this is the type-checker fallback.
_FALLBACK_MAX_BUILDING_HP = 4008.0

# PlayerState's deckQueue length, deck (8) minus hand (4): the confirmed deals
# needed to pin the opening queue order.
_QUEUE_SIZE = 4


@dataclass
class ApplyResult:
    applied: bool
    reason: str = ""
    candidates_remaining: int = 0


@dataclass
class DriverReport:
    own_plays_attempted: int = 0
    own_plays_applied: int = 0
    own_plays_rejected_by_engine: int = 0
    """Card WAS in hand, engine still said no -- elixir, placement legality
    against a slightly wrong board, or the 20-tick hand-slot cooldown."""

    own_plays_cycle_desync: int = 0
    """Card was in no candidate's hand: the cycle model is wrong. Distinct
    from an engine refusal because the fix is completely different."""

    own_plays_after_game_over: int = 0
    ended_early_at_tick: int | None = None
    """The estimate destroyed a King Tower the real match still had standing.
    Once this fires every later number in the report is downstream of it, so
    it must be read first."""

    opp_plays_injected: int = 0
    unmapped_skipped: int = 0
    pool_rebuilds: int = 0
    """Times the candidate pool was exhausted and redrawn from history. A
    few is normal; many means reset_draws is too small for this deck."""

    candidates_initial: int = 0
    candidates_final: int = 0
    cycle_identified_after: int | None = None
    """Own plays needed before the opening cycle was provably pinned down --
    see SimDriver.cycle_identified. None means it never was, which is the
    normal state when our own hand is not being read (no incoming_card)."""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class SimDriver:
    """Runs a ClashEnv forward as a state estimator.

    Not thread-safe and not restartable: build one per match.
    """

    def __init__(
        self,
        my_deck: list[int],
        opp_deck: list[int] | None = None,
        max_ticks: int = 3600,
        reset_draws: int = DEFAULT_RESET_DRAWS,
        pool_size: int = DEFAULT_POOL_SIZE,
    ):
        self.engine = _engine()
        self.geom = load_geometry()
        self.my_deck = list(my_deck)
        # The opponent's deck only decides the hand the engine deals them, and
        # we never play from it (their cards are injected). Our own deck is a
        # safe, always slot-legal stand-in; a partially discovered opponent
        # deck is not.
        self.opp_deck = list(opp_deck) if opp_deck else list(my_deck)
        self.max_ticks = max_ticks
        self.reset_draws = reset_draws
        self.pool_size = pool_size

        self.max_building_hp = float(
            getattr(self.engine.ClashRoyaleEnv, "MAX_BUILDING_HP", _FALLBACK_MAX_BUILDING_HP)
        )
        self.board_width = self.engine.ClashRoyaleEnv.BOARD_WIDTH
        self.board_height = self.engine.ClashRoyaleEnv.BOARD_HEIGHT

        self._candidates: list = []
        self._history: list = []
        self._opening_hand: tuple[int, ...] | None = None
        self._confirmed_deals = 0
        self.tick = 0
        self._last_event_tick = -1
        self.report = DriverReport()
        self._costs: dict[int, float] = {}

    # --- lifecycle ---

    def start(self, my_opening_hand: tuple[int, ...] | None = None) -> int:
        """Build the candidate pool. Returns how many candidates survived.

        `my_opening_hand` is the hand vision read at match start. Without it a
        single arbitrary candidate is used and own-play fidelity drops to
        chance; supported so the opponent-only path works, but not the intended
        mode.
        """
        Env = self.engine.ClashRoyaleEnv
        self.tick = 0

        self._opening_hand = my_opening_hand
        self._history = []
        self._confirmed_deals = 0

        if my_opening_hand is None:
            env = Env(self.my_deck, self.opp_deck, self.max_ticks)
            env.reset()
            self._candidates = [env]
            self.report.candidates_initial = 1
            return 1

        pool = self._draw_candidates(self.pool_size, self.reset_draws)
        if not pool:
            raise RuntimeError(
                f"no reset produced the opening hand {sorted(my_opening_hand)} "
                f"from deck {sorted(self.my_deck)} in {self.reset_draws} draws. "
                "Either the hand is not a subset of the deck (a card-map error), "
                "or the deck is wrong."
            )

        self._candidates = pool
        self.report.candidates_initial = len(pool)
        return len(pool)

    def _draw_candidates(self, wanted: int, max_draws: int) -> list:
        """Fresh resets consistent with everything observed so far.

        Stops once `wanted` candidates are found or after `max_draws`. Each
        draw is checked against the opening hand first (one get_hand() call
        rejects 69 of 70); only survivors pay for a replay of the event
        history.

        The shuffle is uniform over 1680 cycle states (70 hand-sets x 24 queue
        orders). The hand filter pins the first factor, so the pool must cover
        24 orders drawn uniformly, a coupon-collector problem. At 200
        candidates the chance of missing a given order is (23/24)^200 ~ 2e-4.
        Stepping costs 3.9 us per environment-tick, so 200 environments through
        a match is ~1.4 s, and the pool collapses to ~8 after the first play.
        """
        Env = self.engine.ClashRoyaleEnv
        target = frozenset(self._opening_hand)
        pool: list = []
        for _ in range(max_draws):
            env = Env(self.my_deck, self.opp_deck, self.max_ticks)
            env.reset()
            if frozenset(env.get_hand()) != target:
                continue
            if self._replay_history(env):
                pool.append(env)
                if len(pool) >= wanted:
                    break
        return pool

    def _replay_history(self, env) -> bool:
        """Feed every applied event into a fresh env; False if it disagrees. A
        candidate that cannot reproduce the history (a card not in its hand, a
        refused play, a different card dealt) is not our cycle.
        """
        tick = 0
        for event, incoming in self._history:
            if event.tick > tick:
                env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, event.tick - tick)
                tick = event.tick
            if event.team == 1:
                raw_y = float((self.board_height - 1) - event.tile_y)
                env.inject_enemy(event.card_sim_id, float(event.tile_x), raw_y)
                continue

            hand = list(env.get_hand())
            if event.card_sim_id not in hand:
                return False
            index = hand.index(event.card_sim_id)
            env.step_self_play(index, float(event.tile_x), float(event.tile_y),
                               -1, 0.0, 0.0, 1)
            tick += 1
            new_hand = list(env.get_hand())
            if new_hand == hand:
                return False
            if incoming is not None and new_hand[index] != incoming:
                return False

        if self.tick > tick:
            env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, self.tick - tick)
        return True

    # --- driving ---

    def advance_to(self, tick: int) -> None:
        """Run every candidate forward to `tick` with no action.

        A no-op when already at or past `tick`: applying a placement consumes
        its tick, so advancing to that same tick afterwards is normal.
        Out-of-order events are the real failure, and apply() guards those.

        step_self_play, not step: step() drives the heuristic opponent, which
        would place cards we never saw.
        """
        delta = tick - self.tick
        if delta <= 0:
            return
        for env in self._candidates:
            env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, delta)
        self.tick = tick

    def apply(self, event: PlacementEvent, incoming_card: int | None = None) -> ApplyResult:
        """Apply one placement at its own tick.

        `incoming_card` is the card vision saw enter our hand as a result, used
        to eliminate candidates. Optional: elimination still works off play
        failure alone, more slowly.
        """
        if event.tick < self._last_event_tick:
            raise ValueError(
                f"events must arrive in tick order: got tick {event.tick} after "
                f"{self._last_event_tick}. An out-of-order event silently "
                "reorders cause and effect in the estimate."
            )
        self._last_event_tick = event.tick
        self.advance_to(event.tick)

        if event.card_sim_id == UNKNOWN_CARD_SIM_ID:
            # The card is real and localised but cannot be represented;
            # injecting some other card would corrupt the estimate far worse
            # than omitting it.
            self.report.unmapped_skipped += 1
            return ApplyResult(
                False,
                f"{event.card_real_name!r} has no simulator id -- not injected",
                len(self._candidates),
            )

        if event.team == 1:
            return self._apply_opponent(event)
        return self._apply_own(event, incoming_card)

    def _apply_opponent(self, event: PlacementEvent) -> ApplyResult:
        """Inject an opponent card directly onto the board.

        PlacementEvent.tile_y arrives already mirrored for team 1, and
        injectEnemy takes raw board coordinates, so the mirror is undone here.
        Backwards, every opponent card lands in their back rows:
        plausible-looking, and it destroys the estimate.
        """
        raw_y = float((self.board_height - 1) - event.tile_y)
        for env in self._candidates:
            env.inject_enemy(event.card_sim_id, float(event.tile_x), raw_y)
        # Recorded in the history like an own play. A pool rebuild replays the
        # history into fresh environments; without this the rebuilt candidates
        # fight with no opponent on the board, level the enemy towers, and end
        # the match minutes early.
        self._history.append((event, None))
        self.report.opp_plays_injected += 1
        return ApplyResult(True, "injected", len(self._candidates))

    def _apply_own(self, event: PlacementEvent, incoming_card: int | None) -> ApplyResult:
        """Play one of our cards, eliminating candidates that disagree."""
        self.report.own_plays_attempted += 1
        card = event.card_sim_id

        if self.primary.is_game_over():
            # The estimate's match ended before the real one: it has drifted
            # far enough to destroy a King Tower still standing in reality.
            # Nothing can be applied after that (playCard and step() return
            # immediately when gameOver), so every later play would otherwise
            # count as a refusal and hide the cause.
            self.report.ended_early_at_tick = self.report.ended_early_at_tick or event.tick
            self.report.own_plays_after_game_over += 1
            return ApplyResult(False, "simulated match already over", len(self._candidates))

        was_identified = self.cycle_identified
        survivors, outcome = self._play_on_pool(self._candidates, event, incoming_card)
        self.tick += 1

        if not survivors and self._opening_hand is not None:
            # Every candidate's queue order has been contradicted, so the
            # correct one was never drawn. Rebuild from the full history
            # including this play, then retry; cheaper than covering all 24
            # orders up front.
            self.report.pool_rebuilds += 1
            self._history.append((event, incoming_card))
            rebuilt = self._draw_candidates(REBUILD_WANTED, REBUILD_MAX_DRAWS)
            self._history.pop()  # re-appended below on success, once
            if rebuilt:
                survivors, outcome = rebuilt, "played"

        if not survivors:
            for env in self._candidates:
                env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, 1)
            if outcome == "refused":
                self.report.own_plays_rejected_by_engine += 1
                reason = (f"engine refused card {card} at tick {event.tick} "
                          "(elixir, placement legality, or slot cooldown)")
            else:
                self.report.own_plays_cycle_desync += 1
                reason = (f"no cycle consistent with playing card {card} at tick "
                          f"{event.tick} could be found")
            self.report.candidates_final = len(self._candidates)
            return ApplyResult(False, reason, len(self._candidates))

        self._candidates = survivors
        self._history.append((event, incoming_card))
        if incoming_card is not None:
            self._confirmed_deals += 1
        self.report.candidates_final = len(survivors)
        if not was_identified and self.cycle_identified and self.report.cycle_identified_after is None:
            self.report.cycle_identified_after = self.report.own_plays_attempted

        self.report.own_plays_applied += 1
        return ApplyResult(True, "played", len(survivors))

    def _play_on_pool(self, candidates, event, incoming_card):
        """Apply one own play to every candidate, keeping those that agree.

        Returns (survivors, outcome). The two ways to end up empty have
        different causes and fixes:

          "refused"  -- the card was in hand and the engine still said no
                        (elixir, legality, slot cooldown). The cycle model is fine.
          "desync"   -- no candidate could hold this card, or all dealt a different
                        card afterwards. The cycle model is wrong.
        """
        survivors: list = []
        refused = 0
        had_card = 0
        for env in candidates:
            hand = list(env.get_hand())
            if event.card_sim_id not in hand:
                continue
            had_card += 1
            index = hand.index(event.card_sim_id)
            env.step_self_play(index, float(event.tile_x), float(event.tile_y),
                               -1, 0.0, 0.0, 1)
            new_hand = list(env.get_hand())
            if new_hand == hand:
                refused += 1
                continue
            if incoming_card is not None and new_hand[index] != incoming_card:
                continue
            survivors.append(env)

        if survivors:
            return survivors, "played"
        return [], "refused" if (had_card and refused == had_card) else "desync"

    @property
    def cycle_identified(self) -> bool:
        """True once the opening cycle is provably pinned down: four own plays
        whose dealt card was confirmed against reality.

        Not "one candidate left", which is unreachable (up to 24 slot
        arrangements always survive together, see _cycle_agreed) and
        meaningless (the opening-hand filter makes every candidate agree on the
        hand set from tick 0 while queue orders are unconstrained). The queue
        holds four cards and each play deals one from the front, so after four
        confirmed deals every position has been matched, and the FIFO can never
        diverge again.
        """
        return (
            bool(self._candidates)
            and self._confirmed_deals >= _QUEUE_SIZE
            and self._cycle_agreed(self._candidates)
        )

    @staticmethod
    def _cycle_agreed(candidates) -> bool:
        """Whether every candidate now describes the same cycle.

        Compared as hand sets, not ordered hands. Slot arrangement is a free
        permutation: this driver finds a card by `hand.index(card)` and
        playCard refills the vacated slot, so the same four cards in any
        arrangement behave identically forever. Membership implies queue order:
        each play deals a candidate its own queue.front(), so different queues
        give different hand sets on the next play. Agreement after k plays is
        agreement on the first k queue entries.
        """
        if not candidates:
            return False
        return len({frozenset(env.get_hand()) for env in candidates}) == 1

    # --- readback ---

    @property
    def primary(self):
        """The candidate treated as the estimate: the first survivor, unambiguous
        once the pool has converged. Hence `candidates_remaining` beside every
        result.
        """
        if not self._candidates:
            raise RuntimeError("SimDriver.start() has not been called")
        return self._candidates[0]

    def tower_hp(self) -> dict[str, int]:
        """Predicted tower HP, read out of the observation.

        extractObservationForTeam writes hp / MAX_BUILDING_HP into the building
        channels (3 ours, 7 theirs) at each entity's tile; towers are the only
        thing at those tiles, so multiplying back recovers the engine's tower
        HP.
        """
        obs = self.primary.get_observation_for_team(0)
        width, height = self.board_width, self.board_height
        plane = width * height

        def read(channel: int, x: float, y: float) -> int:
            index = channel * plane + int(y) * width + int(x)
            return int(round(obs[index] * self.max_building_hp))

        g = self.geom
        return {
            "own_king": read(3, *g.own_king),
            "own_princess_left": read(3, *g.own_princess_left),
            "own_princess_right": read(3, *g.own_princess_right),
            "opp_king": read(7, *g.opp_king),
            "opp_princess_left": read(7, *g.opp_princess_left),
            "opp_princess_right": read(7, *g.opp_princess_right),
        }

    def divergence(self, observed_hp: dict[str, int]) -> float:
        """Mean absolute HP error against observed tower HP, per tower. Validation
        only; never reaches the policy.

        King towers are excluded. That dates from when the engine's King fired
        from tick 0 while the real one sleeps until activated; the engine's
        King now sleeps too, so the exclusion could be revisited.
        """
        predicted = self.tower_hp()
        keys = [k for k in observed_hp if k in predicted and "king" not in k]
        if not keys:
            return 0.0
        return sum(abs(predicted[k] - observed_hp[k]) for k in keys) / len(keys)

    def card_cost(self, card_sim_id: int) -> float | None:
        if card_sim_id == UNKNOWN_CARD_SIM_ID:
            return None
        if card_sim_id not in self._costs:
            try:
                self._costs[card_sim_id] = float(
                    self.engine.get_card_info(card_sim_id)["cost"]
                )
            except Exception:
                return None
        return self._costs[card_sim_id]

    def close(self) -> None:
        self._candidates = []
