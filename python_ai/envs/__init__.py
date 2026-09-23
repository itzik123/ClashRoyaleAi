"""Gym environments and the opponents that live inside them.

`gym_wrapper` is phase 1's env (team 1 is the UtilityTeacher or the C++
heuristic); `selfplay_env` is phase 2's (a frozen snapshot, a scripted bot, or
the heuristic). `scenarios` and `scenario_offense` reshape the start-state
distribution, not the reward.
"""
