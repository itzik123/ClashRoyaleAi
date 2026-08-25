#!/usr/bin/env python3
"""Cloud entry point for both pipelines: scaled PPOConfig + TensorBoard -> W&B.

WHY THIS FILE EXISTS AT ALL, rather than an edit to `python_ai/`
----------------------------------------------------------------
`python_ai/` is read-only by default and a training run is usually live against
it. Everything this file needs is already reachable without touching it:

  * both trainers accept `cfg=` (`train.py:247`, `train_selfplay.py:109`), so
    every hyperparameter below is an argument, not a patch;
  * `Phase2Trainer.__init__` supplies `PHASE2_ENTROPY` itself, so passing a cfg
    does NOT lose the deliberately-different phase-2 entropy settings;
  * the trainers already route ~30 `add_scalar` calls through one
    `SummaryWriter` (`rl/base_trainer.py:221`), so `sync_tensorboard=True`
    mirrors all of them with no instrumentation.

THE ONE THING THIS FILE CANNOT FIX FROM OUT HERE
------------------------------------------------
`train.py`'s `launch_pipeline2()` does `subprocess.Popen([sys.executable, "-u",
train_selfplay.py])`, and `train_selfplay.py:483` constructs `Phase2Trainer()`
with NO cfg. So an automatic handoff starts phase 2 with `num_envs` from the
environment but `num_minibatches`/`lr`/`ppo_epochs` at their N=8 defaults --
which is precisely the large-batch regime `scaled_config` exists to avoid.
`supervise.sh` retires that child and re-launches phase 2 through here. See
TODO.md item 9 for the one-line `os.execv` change that would remove the need.

USAGE
    python cloud/launch.py --phase 1
    python cloud/launch.py --phase 2
    python cloud/launch.py --phase auto     # picks by which checkpoint exists
    python cloud/launch.py --dry-run        # print the derived config, run nothing
"""
import argparse
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from python_ai.rl.config import PPOConfig  # noqa: E402


# --------------------------------------------------------------------------
# the scaling rule
# --------------------------------------------------------------------------
#: The configuration every measured baseline in CLAUDE.md was earned at.
BASELINE_ENVS = 8

#: Rows (timesteps) per minibatch at the baseline: (500/25)*8/8 segments x 25.
#: Recomputed from PPOConfig rather than restated, so it cannot go stale if a
#: default moves -- CLAUDE.md's no-second-copies rule applied to a number that
#: is derivable.
def _rows_per_minibatch(cfg, num_minibatches):
    segments = (cfg.update_timestep // cfg.bptt_chunk) * cfg.num_envs
    return (segments // max(1, num_minibatches)) * cfg.bptt_chunk


#: Ceiling on minibatch rows. Two constraints meet here and both are real:
#:
#:   * BELOW it the GPU is launch-bound -- a 1.88M-param net over 34x18 convs
#:     does not saturate an A10 at 500 rows, so small minibatches waste the
#:     hardware the migration was for.
#:   * ABOVE it you are in the large-batch regime: `num_minibatches` is what
#:     sets optimizer steps per rollout (num_minibatches * ppo_epochs), so
#:     letting the minibatch grow with num_envs buys samples while HOLDING
#:     gradient steps constant -- the opposite of what CLAUDE.md says to budget.
#:
#: 2x the baseline keeps the required sqrt(2) LR bump modest. Raise it only
#: with a measurement showing the GPU is still launch-bound at the current N.
TARGET_MINIBATCH_ROWS = 1000


def scaled_config(num_envs, base=None):
    """The PPOConfig for `num_envs`, derived from the N=8 baseline.

    Three coupled adjustments, each with its own reason:

    `num_minibatches` -- chosen so rows/minibatch lands as close to
        TARGET_MINIBATCH_ROWS as possible WHILE DIVIDING THE SEGMENT COUNT
        EVENLY. `PPOUpdater` computes `seg_mb_size = n_segments //
        num_minibatches` and walks the permutation in strides of it, so an
        uneven split is tolerated but leaves one ragged trailing minibatch per
        epoch. An exact divisor avoids that.

    `lr` -- scaled by sqrt(rows/baseline_rows). SQRT, NOT LINEAR: linear scaling
        is an SGD/momentum result, and Adam's per-coordinate normalization
        already absorbs most of the gradient-magnitude change, leaving the noise
        term proportional to 1/sqrt(B). Discounted further below because
        eps_clip=0.2 is a HARD trust region -- overshoot the LR and every step
        simply clips, which shows up as ClipFrac saturating while the effective
        step size collapses.

    `ppo_epochs` -- reduced as the data per rollout grows. More optimizer steps
        over more samples drifts further from the behaviour policy within one
        rollout; 4 epochs was tuned when a rollout was 4,000 transitions.
    """
    base = base or PPOConfig()
    if num_envs == base.num_envs and num_envs == BASELINE_ENVS:
        return base

    segments = (base.update_timestep // base.bptt_chunk) * num_envs
    baseline_rows = ((base.update_timestep // base.bptt_chunk) * BASELINE_ENVS
                     // 8) * base.bptt_chunk          # 500 at the defaults

    # Exact divisors of the segment count, scored by distance to the target.
    #
    # FLOORED AT THE BASELINE COUNT. Without the floor, N=14 scores 7
    # minibatches best (280/7 = 40 segments = exactly 1000 rows) and 7 < 8 is a
    # step-count REGRESSION against the baseline while carrying 1.75x the data
    # -- the large-batch regime, entered while nominally optimising for it.
    divisors = [d for d in range(base.num_minibatches, segments + 1)
                if segments % d == 0]
    num_minibatches = min(
        divisors,
        key=lambda d: abs((segments // d) * base.bptt_chunk
                          - TARGET_MINIBATCH_ROWS))

    rows = (segments // num_minibatches) * base.bptt_chunk
    growth = rows / baseline_rows

    # sqrt scaling, then a 0.75 discount against the clip trust region.
    lr = base.lr * (1.0 + 0.75 * (growth ** 0.5 - 1.0))

    # Less reuse as the data per rollout grows -- but ONLY while the cut still
    # leaves more gradient steps than the baseline had.
    #
    # Tying the cut to raw data growth instead was the defect this comment
    # replaces: at N=16 the minibatch count is unchanged (rows/minibatch is
    # held at ~2x baseline, so num_minibatches = N/2 only reaches 8 at N=16),
    # and cutting 4 -> 3 epochs there gave 8x3 = 24 optimizer steps against the
    # baseline's 32. Twice the data and three quarters of the gradient steps is
    # exactly the failure `scaled_config` exists to prevent.
    baseline_steps = base.num_minibatches * base.ppo_epochs
    ppo_epochs = base.ppo_epochs
    while (ppo_epochs > 2
           and num_minibatches * (ppo_epochs - 1) >= baseline_steps * 1.25):
        ppo_epochs -= 1

    return PPOConfig(
        num_envs=num_envs,
        num_minibatches=num_minibatches,
        ppo_epochs=ppo_epochs,
        lr=round(lr, 8),
        # Everything below is DELIBERATELY unchanged. gamma and gae_lambda are
        # per-trajectory temporal parameters and have nothing to do with how
        # many trajectories run in parallel; update_timestep/bptt_chunk are the
        # truncated-BPTT structure (500/25 = 20 chunks), not a batch size.
        gamma=base.gamma,
        gae_lambda=base.gae_lambda,
        eps_clip=base.eps_clip,
        update_timestep=base.update_timestep,
        bptt_chunk=base.bptt_chunk,
        max_grad_norm=base.max_grad_norm,
        vf_clip_std_frac=base.vf_clip_std_frac,
        aux_elixir_coef=base.aux_elixir_coef,
        aux_elixir_scale=base.aux_elixir_scale,
        save_every_episodes=base.save_every_episodes,
        replay_every_episodes=base.replay_every_episodes,
    )


def describe(cfg):
    """One block a human can diff against CLAUDE.md's documented baseline."""
    segments = (cfg.update_timestep // cfg.bptt_chunk) * cfg.num_envs
    rows = _rows_per_minibatch(cfg, cfg.num_minibatches)
    base = PPOConfig(num_envs=BASELINE_ENVS)
    return "\n".join([
        "  num_envs            %d   (baseline %d)" % (cfg.num_envs, BASELINE_ENVS),
        "  segments/rollout    %d" % segments,
        "  num_minibatches     %d   (baseline %d)" % (cfg.num_minibatches,
                                                      base.num_minibatches),
        "  rows/minibatch      %d   (baseline 500, target %d)" % (
            rows, TARGET_MINIBATCH_ROWS),
        "  optimizer steps/roll %d  (baseline 32)" % (
            cfg.num_minibatches * cfg.ppo_epochs),
        "  transitions/rollout %d" % (cfg.update_timestep * cfg.num_envs),
        "  ppo_epochs          %d   (baseline %d)" % (cfg.ppo_epochs,
                                                      base.ppo_epochs),
        "  lr                  %.3e (baseline %.3e)" % (cfg.lr, base.lr),
        "  gamma/gae_lambda    %.2f / %.2f  (UNCHANGED by design)" % (
            cfg.gamma, cfg.gae_lambda),
        "  buffer obs memory   %.2f GB" % (
            cfg.update_timestep * cfg.num_envs * 13606 * 4 / 1e9),
    ])


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def resolve_phase(requested):
    """`auto` picks the phase from which checkpoint exists on disk.

    Resolved through `rl.checkpointing.weights_path`, never cwd-relative -- the
    whole point of commit 15a3509 is that the launch directory no longer decides
    whether a run resumes.

    The fallback exists so `--dry-run` works on a machine with no training env:
    `checkpointing` imports torch at module scope (for save/load), while
    `weights_path` itself is pure path arithmetic. The fallback anchors on the
    SAME `python_ai.PACKAGE_DIR` and reproduces the same
    absolute-name-passes-through contract, so it is one derivation point with a
    lighter import, not a second copy of the location.
    """
    if requested != "auto":
        return int(requested)
    name = os.environ.get("CLASH_WEIGHTS_SELFPLAY", "model_weights_selfplay.pth")
    try:
        from python_ai.rl.checkpointing import weights_path
        selfplay = weights_path(name)
    except ImportError:
        import python_ai
        selfplay = (name if os.path.isabs(name)
                    else os.path.join(python_ai.PACKAGE_DIR, name))
    return 2 if os.path.exists(selfplay) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="auto", choices=["1", "2", "auto"])
    ap.add_argument("--num-envs", type=int,
                    default=int(os.environ.get("CLASH_NUM_ENVS", 28)))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the derived config and exit")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = scaled_config(args.num_envs)
    phase = resolve_phase(args.phase)

    print("=" * 66)
    print("ClashRoyaleEnv cloud launch -- pipeline %d" % phase)
    print("=" * 66)
    print(describe(cfg))

    # THE CURRICULUM WINDOW CHECK, deliberately BEFORE the --dry-run return:
    # this is the single thing most worth knowing before deploying, and a dry
    # run that hid it would be worse than useless.
    #
    # `rl/curriculum.OUTCOME_WINDOW` is denominated in EPISODES while the gate's
    # real evidence is measured in UPDATES, and at ~12.24 episodes per update
    # (943 ep/h / 77 upd/h, CLAUDE.md "Measured baselines") the two diverge as
    # num_envs grows: N=8 gives the gate 8.2 updates of evidence, N=28 gives
    # 2.3, N=128 gives half of one. Below ~4 the ladder advances on sampling
    # noise and then calls outcome_history.clear().
    from python_ai.rl.curriculum import OUTCOME_WINDOW
    episodes_per_update = 12.24 * (cfg.num_envs / BASELINE_ENVS)
    updates_of_evidence = OUTCOME_WINDOW / episodes_per_update
    print("  curriculum window   %d episodes = %.1f updates of evidence "
          "(8.2 at baseline)" % (OUTCOME_WINDOW, updates_of_evidence))
    print("=" * 66, flush=True)

    blocked = phase == 1 and updates_of_evidence < 4.0
    if blocked:
        safe_n = max(BASELINE_ENVS,
                     int(OUTCOME_WINDOW * BASELINE_ENVS / (12.24 * 4.0)))
        print("")
        print("  *** REFUSING TO START -- narrow curriculum window ***")
        print("  rl/curriculum.OUTCOME_WINDOW is %d episodes, which at "
              "num_envs=%d is only %.1f" % (OUTCOME_WINDOW, cfg.num_envs,
                                            updates_of_evidence))
        print("  updates of evidence. The stage gate would advance on sampling")
        print("  noise, clear the window, and the run would read as fast")
        print("  convergence into a flatline. See TODO.md item 9.")
        print("")
        print("  Fix (python_ai/rl/curriculum.py):")
        print("      OUTCOME_WINDOW = 100 * max(1, num_envs // 8)")
        print("  or lower --num-envs to %d, or set "
              "CLASH_ALLOW_NARROW_WINDOW=1 to override." % safe_n)
        print("", flush=True)
        if os.environ.get("CLASH_ALLOW_NARROW_WINDOW") == "1":
            print("  CLASH_ALLOW_NARROW_WINDOW=1 -- proceeding anyway.",
                  flush=True)
            blocked = False

    if args.dry_run:
        return 2 if blocked else 0
    if blocked:
        return 2

    import torch

    # Workers inherit OMP_NUM_THREADS=1 from the unit file (see cloud/README.md
    # -- nothing in the training path calls torch.set_num_threads, and phase 2
    # runs a frozen MicroRoyaleNet inside EVERY worker). The main process feeds
    # the GPU and wants whatever cores the workers are not using.
    spare = (os.cpu_count() or 8) - cfg.num_envs
    torch.set_num_threads(max(1, spare))
    print("  torch threads       %d (main process); OMP_NUM_THREADS=%s (workers)"
          % (max(1, spare), os.environ.get("OMP_NUM_THREADS", "UNSET -- FIX THIS")))
    print("  device              %s"
          % ("cuda:" + torch.cuda.get_device_name(0)
             if torch.cuda.is_available() else "cpu (NO GPU VISIBLE)"), flush=True)

    if not args.no_wandb and os.environ.get("CLASH_WANDB", "1") == "1":
        import wandb
        # A STABLE id plus resume="allow": base_trainer.py:220 wipes the
        # TensorBoard directory whenever load_checkpoint() returns False, so
        # without this every preemption would start a fresh W&B run and the
        # charts would fragment across restarts.
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "clash-royale-rl"),
            id=os.environ.get("WANDB_RUN_ID", "clash-phase%d" % phase),
            resume="allow",
            sync_tensorboard=True,
            config={"phase": phase, "num_envs": cfg.num_envs, "lr": cfg.lr,
                    "num_minibatches": cfg.num_minibatches,
                    "ppo_epochs": cfg.ppo_epochs,
                    "update_timestep": cfg.update_timestep,
                    "bptt_chunk": cfg.bptt_chunk})

    if phase == 1:
        from python_ai.trainers.train import Phase1Trainer
        Phase1Trainer(cfg=cfg).run()
    else:
        from python_ai.trainers.train_selfplay import Phase2Trainer
        Phase2Trainer(cfg=cfg).run()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
