"""Live strategy diagnostics: what the win rate cannot see.

These track the two degenerate behaviours measured in phase 1, both of which are
invisible in win/loss against a weak opponent and fatal against a competent one:

  * the greedy policy had abandoned 2 of its 8 cards entirely -- never the win
    condition, never the spell -- and won purely by cheap defence
  * it never accumulated elixir (mean 3.6/10 at the moment it acted), so it
    could never afford a real push

READ THESE IN PAIRS, WHICH IS THE ENTIRE POINT OF THE MODULE.

`Fwd` (forward placement rate) and the entropy series can BOTH rise from noise
alone: raising placement entropy spreads the marginal toward the middle of the
legal region, which lifts forward rate whether or not the policy got better.
`ROI` is the half that noise pushes the OTHER way. Spread up with ROI up or flat
is learning; spread up with ROI down is randomness.

TWO DEFINITIONS THAT ARE LOAD-BEARING:

  ROI is a ratio of SUMS over the window, never a mean of per-episode ratios.
      An episode where a card went unplayed contributes 0/0, and averaging those
      moves the number for reasons unrelated to how well the card was used.
  TwrDmg is per 1000 TICKS, not per episode. `AvgTicks` moved 24% in one hour
      after the placement-mask fix, and an unnormalized total would have read as
      more pressure when it was only longer matches.
"""
from collections import deque

import numpy as np

#: Placements at or beyond this row are "forward" -- the bridge-adjacent band
#: rather than the tower pocket. Reported as a rate so it does not move with
#: match length.
FORWARD_ROW_Y = 12

#: Engine ticks per bot decision, for the tick-normalized damage rate.
TICKS_PER_DECISION = 10.0


class StrategyMetrics:
    """Per-env accumulators plus the rolling windows they close out into."""

    def __init__(self, num_envs, deck, short=50):
        self.num_envs = num_envs
        self.deck = list(deck)
        self.cards_per_game = deque(maxlen=short)
        self.elixir_at_play = deque(maxlen=short)
        self.plays_per_game = deque(maxlen=short)
        self.forward_rate = deque(maxlen=short)
        self.tower_damage_rate = deque(maxlen=short)
        #: Realized per-card elixir economy, summed over recent episodes.
        self.econ_killed = deque(maxlen=short)
        self.econ_spent = deque(maxlen=short)

        self._card_ids = [set() for _ in range(num_envs)]
        self._elixir_at_play = [[] for _ in range(num_envs)]
        self._play_count = np.zeros(num_envs, dtype=np.int64)
        self._forward_placements = np.zeros(num_envs, dtype=np.int64)

    # -- per-step -----------------------------------------------------------
    def record_play(self, i, card_id, elixir, cell, board_width):
        """One card actually reached the board in env `i`.

        "The policy chose a card" is NOT the same as "a card was played": the
        engine silently refuses an illegal or unaffordable play, so only a rise
        in cumulative elixir_spent proves it. The caller checks that; this only
        records.

        Card identity and elixir must come from the observation the ACTION was
        taken on -- the hand rotates the instant a card is played, so reading it
        afterwards returns the wrong card.
        """
        if card_id is not None and card_id >= 0:
            self._card_ids[i].add(int(card_id))
        self._elixir_at_play[i].append(float(elixir))
        self._play_count[i] += 1
        # `cell` is row-major over board_width, the same layout cell_to_xy
        # inverts.
        if cell // board_width >= FORWARD_ROW_Y:
            self._forward_placements[i] += 1

    # -- per-episode --------------------------------------------------------
    def finish_episode(self, i, killed_by_card, spent_by_card, tower_damage,
                       steps):
        self.cards_per_game.append(len(self._card_ids[i]))
        self.plays_per_game.append(int(self._play_count[i]))
        if self._elixir_at_play[i]:
            self.elixir_at_play.append(float(np.mean(self._elixir_at_play[i])))
        if self._play_count[i] > 0:
            self.forward_rate.append(float(self._forward_placements[i])
                                     / float(self._play_count[i]))
        spent = np.asarray(spent_by_card, dtype=np.float64)
        if spent.sum() > 0.0:
            self.econ_killed.append(np.asarray(killed_by_card, dtype=np.float64))
            self.econ_spent.append(spent)
        ticks = float(steps) * TICKS_PER_DECISION
        if ticks > 0.0:
            self.tower_damage_rate.append(float(tower_damage) * 1000.0 / ticks)
        self.reset_env(i)

    def reset_env(self, i):
        self._card_ids[i] = set()
        self._elixir_at_play[i] = []
        self._play_count[i] = 0
        self._forward_placements[i] = 0

    # -- read-out -----------------------------------------------------------
    def roi(self):
        """(overall ROI, worst card index, worst ROI, per-card ROI array).

        `per_card` is None where nothing was spent on that slot in the window.
        """
        if not self.econ_spent:
            return float("nan"), None, float("nan"), None
        killed = np.sum(self.econ_killed, axis=0)
        spent = np.sum(self.econ_spent, axis=0)
        overall = float(killed.sum() / spent.sum()) if spent.sum() > 0 else float("nan")
        played = spent > 0
        if not played.any():
            return overall, None, float("nan"), None
        per_card = np.where(played, killed / np.maximum(spent, 1e-9), np.inf)
        worst = int(np.argmin(per_card))
        return overall, worst, float(per_card[worst]), np.where(played, per_card, np.nan)

    def summary(self):
        overall_roi, worst_idx, worst_roi, per_card = self.roi()
        return {
            "cards_per_game": _mean(self.cards_per_game),
            "plays_per_game": _mean(self.plays_per_game),
            "elixir_at_play": _mean(self.elixir_at_play),
            "forward_rate": _mean(self.forward_rate),
            "tower_damage_rate": _mean(self.tower_damage_rate),
            "roi": overall_roi,
            "roi_worst": worst_roi,
            "roi_worst_card": (self.deck[worst_idx]
                               if worst_idx is not None else None),
            "roi_per_card": per_card,
        }


class ScenarioMetrics:
    """Success rates for injected scenarios, split by whether the test means
    anything for that scenario.

    Only DEFENSIVE scenarios reach `defensive`. Success there means "the episode
    did not end in a tower/game loss", which is only a question worth asking
    when something was threatening us. `fireball_tower_value` spawns at the
    ENEMY tower, so it passes that test by default -- including when the agent
    does nothing at all. Counting it pushed ScenDef toward 1.0 and would have
    masked a genuine collapse in the very reflex the metric exists to watch.
    """

    def __init__(self, maxlen=200):
        self.defensive = deque(maxlen=maxlen)
        self.other = deque(maxlen=maxlen)

    def record(self, is_defensive, raw_reward):
        target = self.defensive if is_defensive else self.other
        target.append(1.0 if raw_reward > -0.5 else 0.0)

    def summary(self):
        return {
            "defensive_rate": _mean(self.defensive),
            "defensive_n": len(self.defensive),
            "other_rate": _mean(self.other),
            "other_n": len(self.other),
        }


def _mean(window):
    return float(np.mean(window)) if window else float("nan")
