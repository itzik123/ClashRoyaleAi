"""Live strategy diagnostics: what the win rate cannot see.

They track behaviours invisible in win/loss against a weak opponent: abandoning
cards (never playing the win condition or the spell) and never banking enough
elixir for a push.

Read them in pairs. `Fwd` (forward placement rate) and the entropy series both
rise from noise alone; `ROI` is the half that noise pushes the other way.
Spread up with ROI up or flat is learning; spread up with ROI down is
randomness.

  ROI     a ratio of sums over the window, never a mean of per-episode
          ratios (an unplayed card contributes 0/0)
  TwrDmg  per 1000 ticks, not per episode, so longer matches do not read as
          more pressure
"""
from collections import deque

import numpy as np

#: Placements at or beyond this row are "forward" (the bridge-adjacent band);
#: reported as a rate.
FORWARD_ROW_Y = 12

#: Engine ticks per decision, for the tick-normalized damage rate.
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
        #: Realized per-card elixir economy over recent episodes.
        self.econ_killed = deque(maxlen=short)
        self.econ_spent = deque(maxlen=short)

        self._card_ids = [set() for _ in range(num_envs)]
        self._elixir_at_play = [[] for _ in range(num_envs)]
        self._play_count = np.zeros(num_envs, dtype=np.int64)
        self._forward_placements = np.zeros(num_envs, dtype=np.int64)

    def record_play(self, i, card_id, elixir, cell, board_width):
        """One card actually reached the board in env `i`.

        The caller establishes that (the engine silently refuses bad plays, so
        only a rise in elixir_spent proves one). Card identity and elixir must
        come from the observation the action was taken on: the hand rotates as
        soon as a card is played.
        """
        if card_id is not None and card_id >= 0:
            self._card_ids[i].add(int(card_id))
        self._elixir_at_play[i].append(float(elixir))
        self._play_count[i] += 1
        # `cell` is row-major over board_width, as cell_to_xy inverts.
        if cell // board_width >= FORWARD_ROW_Y:
            self._forward_placements[i] += 1

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

    def roi(self):
        """(overall ROI, worst card index, worst ROI, per-card ROI array);
        per-card entries are NaN where nothing was spent in the window.
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
    anything.

    Only defensive scenarios reach `defensive`, where success ("no tower or
    game lost") is a real question. A scenario with no threat on our half
    passes by default and would mask a real collapse.
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
