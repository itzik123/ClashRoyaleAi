"""Where a run's weights go, and which of them another run may play against.

Three destinations, deliberately separate directories:

  the live checkpoint     resumed from; carries optimizer + training state
  historical_checkpoints/ pipeline 2's PFSP opponent pool, ordered by mtime
  stage_checkpoints/      diagnostic-only, one per curriculum-stage transition

Mixing the last two would silently change which opponents self-play samples --
the pool is read as a weakest-to-strongest ladder by save order.
"""
import os
import time

import torch

# Historical self-play (pipeline #2, train_selfplay.py) needs a library of past
# versions of this same policy to play against, weakest to strongest -- these
# are saved here as bare weights-only snapshots (never resumed-from for further
# gradient training, so no optimizer/training-state needed) periodically
# throughout THIS training run, not just at the end, so the library already
# spans a useful weak->strong range by the time pipeline #2 starts. Filenames
# are timestamp-prefixed rather than keyed by this run's own episode count --
# train_selfplay.py sorts the shared folder by save order (mtime) to build a
# single weakest-to-strongest queue, since it saves its own new (and by then
# much stronger) snapshots into the same folder as pipeline #2 progresses, and
# those two runs' episode counters aren't on the same scale.
HISTORICAL_CHECKPOINT_DIR = "historical_checkpoints"

# Lowered 5000 -> 2000 on 2026-08-09, as a RE-DENOMINATION rather than a change
# of intent. The 2026-08-07 movement-speed fix left gradient steps per hour
# unchanged but cut episodes per hour 2,873 -> 1,301, so one episode now carries
# ~2.2x more transitions and ~2.2x more policy change. 5,000 episodes had come
# to mean what ~11,000 used to; 5000 / 2.2 ~= 2,270, rounded to 2,000.
#
# Deliberately NOT 1,000, which was the other candidate: that would make the
# pool ~5x denser than the original design, and 5,000 was itself a judgement
# call rather than a measured optimum, so there is nothing to justify
# overshooting it. Two costs bound this from above -- every extra pool member
# dilutes the PFSP share of every other (which is what already forced
# DEFENSIVE_SCRIPTED_MIN_WEIGHT up to 0.8), and each worker keeps its OWN local
# per-opponent win-rate estimate, so more members means fewer games each and a
# noisier (1 - winrate)^2 weighting.
#
# MIN_OPPONENT_AGE_EPISODES in train_selfplay.py is kept at 3x this value; see
# its comment. The two are coupled and changing one alone silently changes
# which snapshots are eligible.
HISTORICAL_CHECKPOINT_INTERVAL_EPISODES = 2000

# Diagnostic-only snapshots, one per curriculum-stage transition: the exact
# policy that just cleared a stage's 80%-over-100-episodes gate, captured
# BEFORE the opponent gets harder.
#
# Deliberately a SEPARATE directory from HISTORICAL_CHECKPOINT_DIR, not extra
# files in it: that one is pipeline #2's PFSP opponent pool, ordered by mtime
# as a weakest->strongest ladder, so dropping additional checkpoints in would
# silently change which opponents self-play samples and how often. These are
# for offline analysis only and nothing ever trains against them.
#
# The motivating question, which could not be answered on the previous run
# because the checkpoints had already been deleted: the narrow policy measured
# at stage 4 (8 of 288 placement cells, 5 of 8 deck cards, one lane) was
# measured ONLY against that stage's 1.4x-elixir opponent. Whether that
# narrowness is a pathological collapse or a rational response to being
# out-elixired is decidable -- but only by replaying ONE fixed policy against
# SEVERAL stages' opponents, which requires having kept the per-stage weights.
STAGE_CHECKPOINT_DIR = "stage_checkpoints"

def save_stage_snapshot(net, directory, stage, episodes_completed, teacher_stage, reason):
    """Weights-only snapshot tagged with the curriculum state it was taken at.

    Weights only (no optimizer/training state) on purpose -- these are never
    resumed from, only loaded for offline behavioral probes, so carrying Adam's
    buffers would triple the file size for nothing. The metadata is what makes
    a probe reproducible: it records which opponent strength this policy was
    actually trained against, so a later comparison can replay it against a
    DIFFERENT multiplier and attribute the difference correctly.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"stage{stage}_ep{episodes_completed:08d}.pth")
    torch.save({
        "model": net.state_dict(),
        "curriculum_stage": stage,
        "episodes_completed": episodes_completed,
        "teacher_stage": teacher_stage,
        "reason": reason,
    }, path)
    print(f">>> Stage snapshot saved to {path} ({reason}, teacher_stage={teacher_stage})")
    return path


def save_historical_snapshot(net, episodes_completed, pipeline,
                             directory=HISTORICAL_CHECKPOINT_DIR):
    """Bare weights into the shared PFSP pool, timestamp-prefixed.

    Timestamp-prefixed rather than keyed by episode count because BOTH pipelines
    drop snapshots into this one folder on two unrelated episode scales -- see
    HISTORICAL_CHECKPOINT_DIR. `pipeline` ("pipeline1"/"pipeline2") is still
    embedded because train_selfplay's age gate matches on it: only pipeline 2's
    own recent snapshots are too young to be eligible.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory,
        f"{int(time.time() * 1000)}_{pipeline}_ep{episodes_completed:08d}.pth")
    torch.save({"model": net.state_dict()}, path)
    print(f">>> Historical snapshot saved to {path}")
    return path
