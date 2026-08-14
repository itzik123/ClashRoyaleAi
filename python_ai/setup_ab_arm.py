"""Build a FULL training checkpoint for one arm of a controlled A/B.

THE TRAP THIS EXISTS TO AVOID, and it has already cost this project one
inconclusive experiment. Seeding a bare `{"model": state_dict}` takes
`train.py`'s legacy-checkpoint path, which resets `episodes_completed` to 0.
`placement_entropy_target(0)` then returns ENTROPY_TARGET_PLACEMENT_START = 0.65
against a converged policy measuring ~0.11, so the controller spends the whole
run inflating the policy toward a target meant for a fresh net -- and BOTH arms
end near-uniform, which is indistinguishable from "the treatment did nothing".

So every arm resumes through the FULL path: model + optimizer + curriculum +
a real episode count. Usage:

    python setup_ab_arm.py --src model_weights_hires.pth --dst _runs/x/model_weights.pth
"""
import argparse
import os
import sys
from collections import deque

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="weights to seed from; bare state_dict or full checkpoint")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--episodes", type=int, default=64309,
                    help="episode count the arm resumes AT. Drives the placement "
                         "entropy target -- the whole reason this script exists.")
    ap.add_argument("--stage", type=int, default=5,
                    help="curriculum stage; 5 is the final 1.5x-opponent stage, "
                         "which is the regime prove_placement measures in")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = args.src if os.path.isabs(args.src) else os.path.join(here, args.src)
    blob = torch.load(src, map_location="cpu", weights_only=False)
    model = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)

    # No optimizer state is carried over. Adam moments belong to the parameters
    # that produced them, and `model_weights_hires.pth` was written by a
    # DIFFERENT optimizer (distillation, placement pathway only) than the one
    # PPO is about to run. A fresh optimizer is honest; a borrowed one would
    # apply stale second moments to weights they never saw. train.py accepts an
    # empty state dict and starts clean, and both arms get the same treatment,
    # which is what the comparison actually requires.
    ckpt = {
        "model": model,
        "optimizer": {},
        "curriculum_stage": args.stage,
        "stage_start_episode": args.episodes,
        "episodes_completed": args.episodes,
        "outcome_history": deque(maxlen=100),
        "phase": "mirror",
        "phase_deck_episode_start": args.episodes,
        "random_phase_episode_start": args.episodes,
        "deck_curriculum_stage": args.stage,
    }
    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst}: {len(model)} tensors, resuming at episode "
          f"{args.episodes}, stage {args.stage}")


if __name__ == "__main__":
    main()
