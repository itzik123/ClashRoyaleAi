# Archived 2026-08-25 — pre speed-tier checkpoints

Moved aside for a clean-slate phase 1 restart, not deleted.

Both were trained on an engine that no longer exists. `4b31a42` (2026-08-24,
"Speed tiers: the engine had none, and 104 of 131 troops were wrong") rewrote
151 registry speed literals — Hog Rider 1.6 -> 2.0 tiles/s, Giant 0.6 -> 0.75 —
and its own commit message states that measured strength is void until
retrained. `model_weights.pth` additionally predates DEPLOY_TIME_TICKS
(2026-08-19).

| file | episodes | state carried |
|---|---|---|
| `model_weights.pth` | 7,063 | phase `random_opponent`, curriculum stage 4, ent_coef place 0.0345 / card 0.0106 |
| `model_weights_selfplay.pth` | 31,312 | reference_roster naming 3 snapshots that no longer exist |

**Why both.** `Phase2Trainer.load_checkpoint` tests its OWN checkpoint before
falling back to bootstrapping from pipeline 1's output. Leaving the selfplay
file in place would have resumed episode 31,312 at handoff instead of seeding
from the fresh phase-1 net — the same silent-resume failure one phase later.

Restore by moving either file back up one directory.
