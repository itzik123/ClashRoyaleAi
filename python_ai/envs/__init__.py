"""Gym environments and the opponents that live inside them.

`gym_wrapper` is phase 1's env (team 1 is the UtilityTeacher or the C++
heuristic); `selfplay_env` is phase 2's (team 1 is a frozen snapshot, a
scripted bot, or the built-in heuristic). `scenarios` and `scenario_offense`
reshape the START-STATE distribution rather than the reward."""
