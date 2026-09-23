"""Per-episode bookkeeping: the rolling windows both pipelines report from.

  100 outcomes  the curriculum gate's resolution (a +/-0.1 sampling band)
  500 outcomes  narrow enough bands to show a real trend
   50 episodes  continuous quantities (reward, shaping, length, building HP)
"""
from collections import deque

import numpy as np

WIN, LOSS, DRAW = 1, -1, 0


def outcome_of(raw_reward):
    """+1 / -1 / 0 from the raw engine reward: +/-1 for a decisive result, ~0 for
    a timeout.
    """
    if raw_reward > 0.5:
        return WIN
    if raw_reward < -0.5:
        return LOSS
    return DRAW


class EpisodeMetrics:
    """Rolling windows plus the per-env accumulators feeding them."""

    def __init__(self, num_envs, short=50, window=100, long_window=500):
        self.num_envs = num_envs
        self.outcomes = deque(maxlen=window)
        self.outcomes_long = deque(maxlen=long_window)
        self.rewards = deque(maxlen=short)
        self.shaping = deque(maxlen=short)
        self.lengths = deque(maxlen=short)
        self.ally_building_hp_end = deque(maxlen=short)
        self.enemy_building_hp_end = deque(maxlen=short)

        self.ep_reward = np.zeros(num_envs)
        self.ep_shaping = np.zeros(num_envs)
        self.ep_steps = np.zeros(num_envs, dtype=np.int64)

    def accumulate(self, shaped_rewards, shaping):
        self.ep_reward += shaped_rewards
        self.ep_shaping += shaping
        self.ep_steps += 1

    def finish_episode(self, i, raw_reward, ally_hp_end, enemy_hp_end):
        """Close out env `i`'s episode and return its outcome (+1/-1/0)."""
        self.rewards.append(self.ep_reward[i])
        self.shaping.append(self.ep_shaping[i])
        self.lengths.append(self.ep_steps[i])
        self.ally_building_hp_end.append(ally_hp_end)
        self.enemy_building_hp_end.append(enemy_hp_end)
        outcome = outcome_of(raw_reward)
        self.outcomes.append(outcome)
        self.outcomes_long.append(outcome)
        self.reset_env(i)
        return outcome

    def reset_env(self, i):
        """Clear env `i`'s accumulators without recording an outcome.

        Pipeline 2 uses it for scenario episodes, which start handicapped and
        are kept out of the headline win rate.
        """
        self.ep_reward[i] = 0
        self.ep_shaping[i] = 0
        self.ep_steps[i] = 0

    def counts(self):
        outcomes = np.array(self.outcomes)
        wins = int((outcomes == WIN).sum())
        losses = int((outcomes == LOSS).sum())
        draws = int((outcomes == DRAW).sum())
        return wins, losses, draws, len(outcomes)

    def summary(self):
        """The numbers both console lines print. Empty windows give NaN, not 0.
        """
        wins, losses, draws, n = self.counts()
        decided = wins + losses
        long_ = np.array(self.outcomes_long)
        wins_long = int((long_ == WIN).sum())
        decided_long = wins_long + int((long_ == LOSS).sum())
        return {
            "n": n,
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / n if n else float("nan"),
            "loss_rate": losses / n if n else float("nan"),
            "draw_rate": draws / n if n else float("nan"),
            "decided": decided,
            # NaN when nothing was decided: an agent that draws every game must
            # not read as one that loses every decided game.
            "decisive_win_rate": wins / decided if decided else float("nan"),
            "decisive_win_rate_long": (wins_long / decided_long
                                       if decided_long else float("nan")),
            "avg_reward": _mean(self.rewards),
            "avg_shaping": _mean(self.shaping),
            "avg_length": _mean(self.lengths),
            "ally_building_hp_end": _mean(self.ally_building_hp_end),
            "enemy_building_hp_end": _mean(self.enemy_building_hp_end),
        }


def _mean(window):
    return float(np.mean(window)) if window else float("nan")
