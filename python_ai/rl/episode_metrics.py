"""Per-episode bookkeeping: the rolling windows both pipelines report from.

Nine `deque`s with hand-picked maxlens, declared identically in two files and
appended to from inside a nested loop. Collected here so the windows and the
statistics computed from them cannot drift apart, and so "what does
Win_Rate_100 actually count" has one answer.

WINDOW SIZES ARE NOT ARBITRARY:

  100 outcomes  the curriculum gate's resolution. A 100-episode win rate has a
                sampling band of roughly +/-0.1, which can look like "learning
                then forgetting" when it is only noise.
  500 outcomes  the same rate over a window narrow enough to show a real trend
                change instead of that band.
   50 episodes  everything continuous (reward, shaping sum, length, building
                HP), where the quantity is smooth and 50 is enough.
"""
from collections import deque

import numpy as np

WIN, LOSS, DRAW = 1, -1, 0


def outcome_of(raw_reward):
    """+1 / -1 / 0 from the RAW (unshaped) engine reward.

    The 0.5 thresholds are how a timeout is told apart from a decisive result:
    the engine pays +/-1 for a king kill and ~0 for a timeout, and the wrappers
    never set `truncated` for a natural end.
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

    # -- per-step -----------------------------------------------------------
    def accumulate(self, shaped_rewards, shaping):
        self.ep_reward += shaped_rewards
        self.ep_shaping += shaping
        self.ep_steps += 1

    # -- per-episode --------------------------------------------------------
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

        Pipeline 2 needs this for SCENARIO episodes, which are deliberately kept
        out of the matchup histories: an injected threat is a handicap, and
        averaging it into the headline W/L/D would make the win rate a mix of
        two different games.
        """
        self.ep_reward[i] = 0
        self.ep_shaping[i] = 0
        self.ep_steps[i] = 0

    # -- read-out -----------------------------------------------------------
    def counts(self):
        outcomes = np.array(self.outcomes)
        wins = int((outcomes == WIN).sum())
        losses = int((outcomes == LOSS).sum())
        draws = int((outcomes == DRAW).sum())
        return wins, losses, draws, len(outcomes)

    def summary(self):
        """The numbers both console lines print. Empty windows give NaN rather
        than 0, so "no data yet" cannot be mistaken for "measured zero"."""
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
            # NaN, not 0.0 and not None: an undefined rate must read the same
            # way everywhere in this dict, or a caller has to branch on three
            # conventions to ask one question. 0.0 was the actively misleading
            # one -- an agent that DRAWS EVERY GAME has no decided games, and
            # reporting that as a 0.00 decisive win rate is indistinguishable
            # from losing every decided game, which is a different diagnosis
            # with a different fix. Draw-everything is precisely the timeout
            # pathology DRAW_PENALTY exists to fight, so this misreported at
            # the one moment it mattered most.
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
