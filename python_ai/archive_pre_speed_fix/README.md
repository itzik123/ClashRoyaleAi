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
