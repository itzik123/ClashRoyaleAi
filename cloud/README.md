# `cloud/` — running phase 1 and phase 2 on a rented Linux box

Nothing in here is imported by the training path. It is deployment scaffolding:
a scaled `PPOConfig`, a process supervisor, and three systemd units.

**Nothing in here edits `python_ai/`.** Both trainers already accept `cfg=`
(`train.py:247`, `train_selfplay.py:109`) and already route every metric through
one `SummaryWriter` (`rl/base_trainer.py:221`), so scaling and W&B mirroring are
both reachable from outside the read-only tree. The two places that genuinely
*cannot* be fixed from out here are recorded as **`TODO.md` item 9**.

---

## Why Linux, and why this exact box

**The engine is already portable.** `include/` and `src/` have **zero** hits for
`windows.h`, `__declspec`, `_MSC_VER`, `WIN32`, `#pragma warning`,
`__forceinline` or `intrin.h`. The Catch2 suite already built and ran under WSL
`g++ 13.3.0` on 2026-08-20. The port is a filename fix, not a rewrite.

**The bottleneck is not the one `num_envs` addresses.** 87% of wall clock is the
PPO update (CNN trunk 43%, placement head 41%, LSTM 16%) — dense math on
500-row minibatches, which is a GPU problem. `num_envs` does not shrink that; it
makes each update *bigger*. So the GPU is the reliable multiplier and env
scaling is the speculative one. See `CLAUDE.md`, "Scaling the training loop".

**The instance is bought for its cores, not its GPU.** The rollout is N worker
processes each running the engine plus a `UtilityTeacher` that costs
**0.62 ms/decision at stage 0 and 12.4 ms at stage 5** — 20×. Size for stage 5.
This is why the cheapest GPU pods are a trap: a RunPod RTX 4090 at $0.34/hr
allocates **6 vCPU**, fewer threads than the laptop this is migrating off.

| | GPU | vCPU | RAM | $/hr |
|---|---|---|---|---|
| RunPod RTX 4090 | 4090 | 6 | 41 GB | $0.34 |
| RunPod L40S | L40S | 16 | 94 GB | $0.79 |
| **Lambda `gpu_1x_a10`** | **A10 24 GB** | **30** | **226 GB** | **$1.29** |
| AWS `g6.8xlarge` | L4 | 32 | 128 GB | $2.01 |

Prices checked August 2026; reconfirm at launch.

---

## Deploy order

```bash
git clone <repo> /opt/clash && cd /opt/clash
./cloud/bootstrap.sh                      # apt, venv, build, all four gates

# Size num_envs from PHYSICAL cores, not the vCPU headline.
lscpu -p=Core,Socket | grep -cv '^#'

# Re-measure where the hour goes. Do NOT inherit the laptop's 87/13 split.
OMP_NUM_THREADS=1 .venv/bin/python -m python_ai.tools.profile_training \
    --mode async --episodes 60

# Inspect the derived hyperparameters before committing to them.
.venv/bin/python cloud/launch.py --phase 1 --dry-run

printf 'WANDB_API_KEY=...\n' > /opt/clash/.env && chmod 600 /opt/clash/.env
sudo mkdir -p /var/log/clash && sudo chown ubuntu /var/log/clash
sudo cp cloud/clash*.service cloud/clash*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now clash clash-sync.timer
journalctl -u clash -f
```

---

## What `launch.py` derives, and why

`scaled_config(num_envs)` changes exactly three things and deliberately leaves
everything else alone.

| | N=8 baseline | N=28 | why |
|---|---|---|---|
| `num_minibatches` | 8 | 14 | holds rows/minibatch near 1000 |
| rows/minibatch | 500 | 1000 | GPU-saturating, still trust-region safe |
| `ppo_epochs` | 4 | 3 | more data per rollout, less reuse |
| **optimizer steps/rollout** | **32** | **42** | **must grow with N, not stay flat** |
| `lr` | 3.0e-4 | 3.93e-4 | √ scaling on the minibatch, discounted 0.75 |
| `gamma`, `gae_lambda` | 0.99 / 0.9 | unchanged | per-trajectory, not per-batch |
| `update_timestep`, `bptt_chunk` | 500 / 25 | unchanged | BPTT structure, not a batch size |

Derived across the range (`cloud/launch.py --num-envs N --dry-run`):

| N | `num_minibatches` | rows/mb | `ppo_epochs` | opt steps | `lr` |
|---|---|---|---|---|---|
| 8 | 8 | 500 | 4 | 32 | 3.00e-4 |
| 16 | 8 | 1000 | 4 | 32 | 3.93e-4 |
| 28 | 14 | 1000 | 3 | 42 | 3.93e-4 |
| 32 | 16 | 1000 | 3 | 48 | 3.93e-4 |
| 64 | 32 | 1000 | 2 | 64 | 3.93e-4 |
| 128 | 64 | 1000 | 2 | 128 | 3.93e-4 |

The LR is constant above N=16 because it scales on the **minibatch**, and the
minibatch is deliberately held constant. Optimizer steps are what grow.
`cloud/test_launch.py` pins both properties, plus monotonicity — the first
version of this rule cut `ppo_epochs` on raw data growth and produced **24**
steps at N=16 against the baseline's 32, which is the exact regime it exists to
avoid. The test caught it; the rule now floors `num_minibatches` at the baseline
and only cuts epochs while the cut still leaves more steps than the baseline.

**`num_minibatches` is the lever that matters.** Optimizer steps per rollout are
`num_minibatches × ppo_epochs` — a constant 32 at the defaults, *independent of
`num_envs`*. Leave it at 8 and scaling N only makes each of the same 32 steps
bigger, buying samples while holding gradient steps flat. That is the opposite
of what `CLAUDE.md` says to budget in.

**√ scaling, not linear.** Linear LR scaling is an SGD/momentum result. Adam's
per-coordinate normalization already absorbs most of the magnitude change,
leaving the noise term ∝ 1/√B. The further 0.75 discount is because `eps_clip`
is a *hard* trust region: overshoot and every step clips, which reads as
ClipFrac saturating while the effective step size collapses.

`launch.py` **refuses to start** if `rl/curriculum.OUTCOME_WINDOW` gives the
stage gate fewer than 4 updates of evidence at the chosen `num_envs`. That is
the failure this whole directory is most likely to cause and the least likely
to be noticed — see `TODO.md` item 9.

---

## The three traps these files exist to defuse

**1. `OMP_NUM_THREADS=1` is mandatory and nothing in the training path sets it.**
`torch.set_num_threads` appears in 16 eval harnesses and **zero** trainers.
Phase 2 runs a frozen `MicroRoyaleNet` inside *every* worker
(`envs/selfplay_env.py:407`), so 28 workers × ~30 default intraop threads is 840
threads over 30 cores. Set in `clash.service`; workers inherit it.

**2. systemd's default `KillMode` reaps phase 2 the instant phase 1 hands off.**
`train.py:497` `Popen`s `train_selfplay.py` and returns, so `train.py` exits 0
with a live child. `KillMode=control-group` tears down the cgroup on that exit.
The run dies with a clean exit code and nothing in any log.
`clash.service` sets `KillMode=process`; `supervise.sh` re-adopts the survivor.

**3. The handed-off phase 2 gets the *wrong* config.** `train_selfplay.py:483`
constructs `Phase2Trainer()` with no cfg, so the auto-spawned child runs at
`CLASH_NUM_ENVS` with `num_minibatches`/`lr`/`ppo_epochs` at their N=8 defaults.
`supervise.sh` retires it and relaunches through `launch.py`. Safe because
`Phase1Trainer.on_finish` saves the checkpoint *before* `launch_pipeline2()`
(`train.py:472-477`).

---

## Before committing 60 hours: settle whether env scaling helps at all

The GPU is worth ~6× more gradient steps per hour at *identical*
hyperparameters — no algorithmic risk. Scaling `num_envs` only pays if the run
is gradient-noise-limited, and ClipFrac 0.275 with critic explained variance
0.718 at N=8 suggests it may not be.

```bash
# ~2 h per arm. Plot win rate against GRADIENT STEP, not wall clock.
#   curves overlay     -> not noise-limited; keep N=8-16 and bank the GPU win
#   N=28 clearly above -> scale, and keep launch.py's overrides
CLASH_NUM_ENVS=8  CLASH_WEIGHTS=/tmp/armA.pth CLASH_LOGDIR=/tmp/armA \
    .venv/bin/python cloud/launch.py --phase 1 --no-wandb
CLASH_NUM_ENVS=28 CLASH_WEIGHTS=/tmp/armB.pth CLASH_LOGDIR=/tmp/armB \
    .venv/bin/python cloud/launch.py --phase 1 --no-wandb
```

Fix *n* in advance and do not extend either arm after seeing a marginal result.
This project has already had one *p* = 0.044 collapse to *p* = 0.553 at 4× the
power.

Both `CLASH_WEIGHTS` and `CLASH_LOGDIR` must be **absolute** to redirect an
experiment arm off the live checkpoint — `rl/checkpointing.py` anchors relative
values on `PACKAGE_DIR`/`REPO_ROOT` and honours an absolute override verbatim.
