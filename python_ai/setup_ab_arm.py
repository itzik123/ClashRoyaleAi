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

    # No optimizer MOMENTS are carried over, but a real, empty-state Adam
    # state_dict is: torch's load_state_dict reads `param_groups` and raises
    # KeyError on a bare {}, and train.py's full-resume path is gated on the
    # "optimizer" key being present -- omitting it drops the arm onto the legacy
    # path and straight into the entropy-target trap this file exists to avoid.
    #
    # Moments are dropped deliberately. Adam's second moments belong to the
    # parameters that produced them, and model_weights_hires.pth was written by
    # a different optimizer (distillation, placement pathway only) than the one
    # PPO is about to run. Both arms get the same fresh start, which is what the
    # comparison requires. lr must match train.py's own 3e-4, since
    # load_state_dict overwrites the hyperparameters the trainer just set.
    from model import MicroRoyaleNet
    probe = MicroRoyaleNet(num_ability_slots=0)
    fresh_opt = torch.optim.Adam(probe.parameters(), lr=3e-4).state_dict()

    ckpt = {
        "model": model,
        "optimizer": fresh_opt,
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
