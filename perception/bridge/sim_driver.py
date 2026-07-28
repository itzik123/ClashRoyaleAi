"""Driving the simulator as a state estimator from perception events.

The simulator holds the state; perception only says what was played, where,
and when. This module is the seam between them.

--------------------------------------------------------------------------
THE PROBLEM THIS MODULE HAD TO SOLVE, AND HOW
--------------------------------------------------------------------------
Feeding the OPPONENT's placements is easy: ClashEnv::injectEnemy spawns a
card directly onto the board for team 1, bypassing hand, elixir and placement
legality -- all of which is correct here, because the real game already
validated the play and we derive their elixir ourselves.

Feeding OUR OWN placements is not, and this was the one genuine blocker in
the whole design:

  * injectEnemy hardcodes team 1. There is no injectAlly.
  * step() would work, but it also runs opponentTurn(), which drives the
    built-in heuristic opponent -- it would place phantom cards on top of the
    real ones. Unusable for an estimator.
  * step_self_play() does NOT run opponentTurn (good), but our play goes
    through GameManager::playCard, which requires the card to be in the
    simulator's own team-0 hand at that moment.
  * That hand is SHUFFLED at reset by an internal, unseeded std::mt19937
    (PlayerState::initializeDeck's rng overload), cannot be set, and the
    queue behind it cannot even be read.

So the simulator starts every match with a random cycle state that is not
ours, and our plays fail against it.

--------------------------------------------------------------------------
THE FIX, IN PURE PYTHON, WITH NO ENGINE CHANGE
--------------------------------------------------------------------------
Two facts make this solvable without touching the core:

  1. reset() costs 0.135 ms (measured, 2000 iterations) and its shuffle is
     uniform over all 70 possible hand-sets (measured).
  2. get_hand() lets us READ the resulting team-0 hand.

The unknown state is therefore only: which 4 of the 8 are in hand (70
options, observable) and in what order the other 4 sit in the queue (24
options, NOT observable).

    resets are free  +  70 of the 1680 states are directly checkable
    =>  generate many resets, keep the ones whose hand-set matches the real
        opening hand, and carry them ALL forward as candidates.

Each subsequent play eliminates candidates: one whose hand does not contain
the card we really played is provably not us, and so is one that deals a
different incoming card than the one vision saw enter our hand. The queue
holds four cards, so after four confirmed deals every surviving candidate has
the exact opening queue order -- see `cycle_identified`.

The pool never shrinks to a single OBJECT, and should not be expected to.
Hand slot arrangement is a free permutation that nothing observable depends
on, so up to 24 candidates survive as exact behavioural duplicates. What gets
pinned down is the cycle, not the object.

Once hand-set and queue-order agree, they agree FOREVER: both FIFOs receive
the same plays in the same order, and PlayerState::playCard is deterministic.
No drift, no re-sync.

--------------------------------------------------------------------------
WHAT CAN STILL FAIL, MEASURED RATHER THAN ASSUMED
--------------------------------------------------------------------------
playCard also checks affordability and placement legality, and the
simulator's elixir economy runs about 2% slow (see timebase.py). So a play
made at exactly-affordable elixir, or aimed at a tile the estimate believes
is occupied, can be refused even with the cycle perfectly aligned.

Every one of these is detectable -- a refused play leaves get_hand()
unchanged -- so they are counted rather than assumed away, and counted
SEPARATELY from cycle desyncs because the two have unrelated causes.
DriverReport is the real number, and that number, not speculation, is what
should decide whether the engine needs changing here.

Measured so far, on the zero-error control: 0 refusals, 0 desyncs, and
divergence identically 0. On a training replay, where opponent placements
have to be reconstructed approximately, refusals appear -- but as a
consequence of the estimate drifting, not of the bridge.
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


class SimUnavailableError(RuntimeError):
    """clash_royale_env could not be imported.

    Fatal here specifically, unlike everywhere else in this package -- this
    module's entire job is to drive the engine.
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


# How many hand-matching candidates the opening pool aims for, and the draw
# budget allowed to find them. See SimDriver._draw_candidates for the
# coupon-collector argument behind 200 -- it is not a round number picked for
# comfort, and shrinking it reintroduces an intermittent failure.
DEFAULT_POOL_SIZE = 200
DEFAULT_RESET_DRAWS = 20000

# Recovery draw budget when the pool empties. Only 1/70 of these get as far
# as replaying the history, so the cost is dominated by cheap rejections.
# Sized so finding a few correct candidates is near-certain: each has
# probability 1/1680, so 50000 draws expects ~30.
REBUILD_WANTED = 4
REBUILD_MAX_DRAWS = 50000

# ClashEnv::MAX_BUILDING_HP, the normaliser the observation's building
# channels are divided by. Read from the engine at runtime; this is only the
# fallback for the type checker.
_FALLBACK_MAX_BUILDING_HP = 4008.0

# PlayerState's deckQueue length: deck (8) minus hand (4). The number of
# confirmed deals needed to pin the opening queue order -- see
# SimDriver.cycle_identified.
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
        # The opponent's deck only determines the hand the engine deals them,
        # and we never play from it -- their cards are injected directly. Our
        # own deck is a safe stand-in and is always slot-legal by
        # construction, whereas a partially discovered opponent deck is not.
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

    # -- lifecycle -------------------------------------------------------

    def start(self, my_opening_hand: tuple[int, ...] | None = None) -> int:
        """Build the candidate pool. Returns how many candidates survived.

        `my_opening_hand` is the real hand vision read at match start. Without
        it, a single arbitrary candidate is used and own-play fidelity drops
        to chance -- supported so the opponent-only path works before stage 2
        exists, but it is not the intended mode.
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

        Stops as soon as `wanted` candidates are found, or after `max_draws`
        attempts. Each draw is checked against the opening hand first -- one
        get_hand() call, which rejects 69 of every 70 draws for free -- and
        only the survivors pay for a replay of the event history.

        SIZING, WHICH IS NOT OPTIONAL
        -----------------------------
        The reset shuffle is uniform over 1680 distinct cycle states (70 hand
        sets x 24 queue orders). The hand filter pins the first factor, so the
        pool must cover 24 orders drawn uniformly -- a coupon-collector
        problem, not a "few samples will do" one.

        Undersizing this was a real bug here, and a quietly intermittent one:
        at ~31 hand-matching candidates, roughly two of the 24 orders were
        typically missing, so about one match in four eliminated every
        candidate at some point and could not recover. It looked like a
        nondeterministic bridge failure.

        200 hand-matching candidates puts the chance of missing any given
        order at (23/24)^200 ~ 2e-4. Affordable: stepping costs 3.9 us per
        environment-tick (measured), so 200 environments through a full
        three-minute match is ~1.4 s -- and the pool collapses to ~8 after
        the first play anyway, since only the quarter with the right queue
        front survives.
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
        """Feed every applied event into a fresh env. False if it disagrees.

        A candidate that cannot reproduce the observed history -- a card that
        was not in its hand, a play the engine refused, or a different card
        dealt afterwards -- is not our cycle, and is rejected.
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

    # -- driving ---------------------------------------------------------

    def advance_to(self, tick: int) -> None:
        """Run every candidate forward to `tick` with no action.

        A no-op when already at or past `tick`, rather than an error.
        Applying a placement necessarily consumes the tick it happens on, so
        a caller that applies an event and then advances its loop to the same
        tick is doing the normal thing, not rewinding. Out-of-order EVENTS
        are the failure actually worth catching, and apply() guards those
        directly.

        step_self_play, not step: step() calls opponentTurn(), which would
        drive the built-in heuristic opponent and place cards we never saw.
        """
        delta = tick - self.tick
        if delta <= 0:
            return
        for env in self._candidates:
            env.step_self_play(-1, 0.0, 0.0, -1, 0.0, 0.0, delta)
        self.tick = tick

    def apply(self, event: PlacementEvent, incoming_card: int | None = None) -> ApplyResult:
        """Apply one placement at its own tick.

        `incoming_card` is the card vision saw enter our hand as a result of
        this play, used to eliminate candidates. Optional: candidate
        elimination still works off play failure alone, just more slowly.
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
            # Defined behaviour, not a silent skip: the card is real, we
            # localised it, and we cannot represent it. Injecting some other
            # card would corrupt the estimate far worse than omitting it.
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

        PlacementEvent.tile_y arrives in ClashEnv convention -- already
        mirrored for team 1. injectEnemy takes RAW board coordinates, so the
        mirror has to be undone here. Getting this backwards puts every
        opponent card in their back rows instead of near the river, which
        looks superficially plausible and destroys the estimate.
        """
        raw_y = float((self.board_height - 1) - event.tile_y)
        for env in self._candidates:
            env.inject_enemy(event.card_sim_id, float(event.tile_x), raw_y)
        # Recorded in the history, exactly like an own play. Easy to omit,
        # and omitting it is silent and severe: a pool rebuild replays the
        # history into fresh environments, so without this the rebuilt
        # candidates fight a match with NO OPPONENT ON THE BOARD. Our own
        # units then go unopposed, level the enemy towers, and the estimate's
        # match ends minutes before the real one -- which surfaces much later
        # as an unexplained "simulated match already over".
        self._history.append((event, None))
        self.report.opp_plays_injected += 1
        return ApplyResult(True, "injected", len(self._candidates))

    def _apply_own(self, event: PlacementEvent, incoming_card: int | None) -> ApplyResult:
        """Play one of our cards, eliminating candidates that disagree."""
        self.report.own_plays_attempted += 1
        card = event.card_sim_id

        if self.primary.is_game_over():
            # The estimate's match ended before the real one did -- only
            # possible once the estimate has drifted far enough to destroy a
            # King Tower that is still standing in reality. Nothing can be
            # applied after that (GameManager::playCard and step() both
            # return immediately when gameOver), so every later play would
            # otherwise be counted as an engine refusal and hide the real
            # cause behind a pile of misleading failures.
            self.report.ended_early_at_tick = self.report.ended_early_at_tick or event.tick
            self.report.own_plays_after_game_over += 1
            return ApplyResult(False, "simulated match already over", len(self._candidates))

        was_identified = self.cycle_identified
        survivors, outcome = self._play_on_pool(self._candidates, event, incoming_card)
        self.tick += 1

        if not survivors and self._opening_hand is not None:
            # The pool was exhausted -- every candidate's queue order has now
            # been contradicted, which means the correct one was never drawn.
            # Rebuild from the full history INCLUDING this play, then retry.
            # See _draw_candidates for why this is cheaper than covering all
            # 24 orders up front.
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

        Returns (survivors, outcome). `outcome` distinguishes the two ways an
        empty result can happen, because they have completely different
        causes and completely different fixes:

          "refused"  -- the card WAS in hand and the engine still said no
                        (elixir, placement legality, slot cooldown). Nothing
                        about the cycle model is wrong.
          "desync"   -- no candidate could hold this card, or all of them
                        dealt a different card afterwards. The cycle model
                        is wrong.

        Conflating these was an actual bug here: eliminating every candidate
        on the incoming-card check reported "the card was in no hand" when
        the card had in fact been in every hand.
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
        """True once the opening cycle is provably pinned down.

        The condition is four own plays whose dealt card was confirmed
        against reality -- not "one candidate left", which is both
        unreachable and meaningless here.

        Unreachable, because slot ARRANGEMENT is a free permutation nothing
        observable depends on (see _cycle_agreed), so up to 24 genuine
        duplicates always survive together.

        Meaningless, because the pool is filtered on the opening hand at
        construction, so every candidate agrees on the hand set from tick 0
        while their queue ORDERS are still completely unconstrained.

        Four confirmed deals is the real bar: the queue holds exactly four
        cards, each play deals one from the front, so after four confirmed
        deals every position of the opening queue has been revealed and
        matched. A candidate that survived all four has the exact opening
        queue order, and from there its FIFO can never diverge from ours
        again -- both receive the same plays and PlayerState::playCard is
        deterministic.
        """
        return (
            bool(self._candidates)
            and self._confirmed_deals >= _QUEUE_SIZE
            and self._cycle_agreed(self._candidates)
        )

    @staticmethod
    def _cycle_agreed(candidates) -> bool:
        """Whether every candidate now describes the SAME cycle.

        Compared as hand SETS, not as ordered hands, and that distinction is
        the whole point.

        The pool never collapses to one object, because slot ARRANGEMENT is a
        free permutation: PlayerState::initializeDeck's shuffle decides which
        of the four slots each card sits in, and nothing observable depends
        on it -- this driver finds a card by `hand.index(card)`, and
        PlayerState::playCard refills the slot the card vacated, so the same
        four cards in any arrangement behave identically forever. Up to 24
        candidates therefore survive as genuine duplicates of one another.

        What actually has to be pinned down is hand MEMBERSHIP and queue
        ORDER, and membership implies the order: after each play a candidate
        receives its own queue.front(), so two candidates whose queues differ
        end up with different hand sets on the very next play. Agreement on
        the hand set after k plays is therefore agreement on the first k
        entries of the queue.
        """
        if not candidates:
            return False
        return len({frozenset(env.get_hand()) for env in candidates}) == 1

    # -- readback --------------------------------------------------------

    @property
    def primary(self):
        """The candidate treated as the estimate.

        Once the pool has collapsed this is unambiguous. Before then it is
        the first survivor, which is the best available answer and is exactly
        why `candidates_remaining` is reported alongside every result.
        """
        if not self._candidates:
            raise RuntimeError("SimDriver.start() has not been called")
        return self._candidates[0]

    def tower_hp(self) -> dict[str, int]:
        """Predicted tower HP, read out of the observation vector.

        No new binding needed. ClashEnv::extractObservationForTeam writes
        hp / MAX_BUILDING_HP into the building channels (3 for our side, 7 for
        theirs) at the tile the entity occupies, so multiplying back recovers
        the HP the engine believes each tower has.

        Towers are the only thing at those tiles, so the "last writer wins"
        collision in that encoding cannot affect this.
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
        """Mean absolute HP error against observed tower HP, per tower.

        King towers are EXCLUDED. The engine's King fires from tick 0 with no
        activation condition (Tower.h has none), while the real King is
        dormant until it is activated -- so King HP diverges systematically
        from the first second even with flawless perception, and including it
        would bury the signal this metric exists to measure. See
        perception/README.md, "Findings reported upstream".

        VALIDATION ONLY. This never enters the observation and never reaches
        the policy.
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
