# Checkpoints from before the 2026-08-07 movement-speed fix

`model_weights.pth` (phase 1, 2026-07-30) and `model_weights_selfplay.pth`
(phase 2, 2026-08-01), archived when troop movement was corrected from ~4-5x
the real game's speed to real-game speed (commit bd7b05b, and
perception/UPSTREAM_REQUESTS.md item 9).

These are NOT architecturally dead -- the observation layout is unchanged, so
they still load. They are STRATEGICALLY dead: every timing relationship they
learned was trained at roughly five times real speed, so a troop crossing a
defender's range now absorbs ~5x more shots than it did when these weights
were fit. Their win-rate history describes a different game.

Kept rather than deleted so the old physics can still be probed if a question
comes up that only a policy trained under them can answer -- which is exactly
the reason stage checkpointing was added in the first place.

## Also here: the only surviving record of the 130k-episode lineage

`training_selfplay_pfsp_pre_cannonfix.log` and `replays_sample/` (3 of the 126
phase-2 replays that used to be in `python_ai/replays/`, sampled from the start,
middle and end of the run).

Both are **local-only** — `.gitignore` covers `*.log` and `replays_sample/`,
matching the repo's existing rule that replays and checkpoints are not tracked.
A fresh clone will not have them, which is why every number they produced is
written out below rather than left implicit in the files.

Kept back from the 2026-08-08 cleanup because that run's TensorBoard event file
is already gone, so these are the last evidence of a measured behaviour worth
being able to re-check: **~30% of ALL placements landed on the two cells
(11,2)/(11,3), behind its own King, across every card in the deck** -- Giant
47.7%, Valkyrie 37.8%, Cannon 28.0%.

The Cannon figure reproduces the 27.9% recorded in `train.py`'s
`tower_potential` docstring, which is the measurement that justified splitting
deployed-building damage out of the tower potential on 2026-08-06. What that
docstring does not say, and what these files show, is that **the concentration
was never Cannon-specific**: the Cannon was not even the worst card, and the
building/tower price asymmetry cannot explain a Giant going behind its own King
half the time. Whatever drove it was a property of the placement head, not of
the Cannon's reward. Every one of these files predates the reward fix, so none
of them says whether that fix worked.
