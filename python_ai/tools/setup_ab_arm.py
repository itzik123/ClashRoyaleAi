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

from python_ai.rl.checkpointing import atomic_save

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401


def _widen_optimizer(opt_state, old_model, new_model):
    """Carry Adam moments across a checkpoint that gained parameters.

    THE FAILURE THIS FIXES, caught by smoke-testing pipeline 2 before the long
    run rather than at hour 0 of it:

        ValueError: loaded state dict contains a parameter group that doesn't
        match the size of optimizer's group

    `place_hires` added 6 parameters, so the shipping checkpoint's optimizer
    describes 26 and the current net has 32. torch matches optimizer state to
    parameters by POSITIONAL INDEX, and train_selfplay.py -- unlike train.py --
    loads it unconditionally, so this is fatal on resume.

    Dropping the state would work and is wrong: it discards Adam's second
    moments for the trunk, LSTM and critic, none of which changed, and those
    are what keep a warm resume stable. So the indices are remapped by NAME.

    The new parameters land at positions 22-27, in the MIDDLE of the ordering,
    not appended -- so a naive "keep 0..25, append 26..31" remap would silently
    hand the trunk's moments to the placement head. That is the kind of error
    that produces a run which trains, looks healthy, and is subtly wrong.

    Old ordering is reconstructed as "current order minus the new names", valid
    because nothing else moved in the module registration order; the count
    assertion below is what makes that assumption load-bearing rather than
    hoped-for.
    """
    if not opt_state or "param_groups" not in opt_state:
        return opt_state
    from python_ai.models.net import MicroRoyaleNet

    net = MicroRoyaleNet(num_ability_slots=0)
    new_names = [n for n, _ in net.named_parameters()]
    added = set(new_model) - set(old_model)
    old_names = [n for n in new_names if n not in added]

    group = opt_state["param_groups"][0]
    if len(group["params"]) != len(old_names):
        raise SystemExit(
            f"cannot remap optimizer state: checkpoint describes "
            f"{len(group['params'])} parameters but reconstructing the old "
            f"ordering gives {len(old_names)}. The module registration order "
            f"changed by more than an append -- remap by hand.")

    name_to_new = {n: i for i, n in enumerate(new_names)}
    remap = {old_i: name_to_new[old_names[pos]]
             for pos, old_i in enumerate(group["params"])}
    state = {remap[int(k)]: v for k, v in opt_state["state"].items()
             if int(k) in remap}

    out = dict(opt_state)
    out["state"] = state
    # Every parameter listed, new ones with no state -- Adam initializes
    # exp_avg/exp_avg_sq lazily on first step, so an absent entry is correct
    # and is exactly what a fresh parameter should get.
    out["param_groups"] = [dict(group, params=list(range(len(new_names))))]
    print(f"  optimizer: remapped {len(state)} moment entries by name, "
          f"{len(new_names) - len(state)} fresh")
    return out


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
    ap.add_argument("--base", default=None,
                    help="a FULL checkpoint to inherit training state from "
                         "(optimizer moments, league roster, entropy "
                         "coefficients, episode count). --src then supplies only "
                         "the weights. Use when the weights come from an offline "
                         "distillation but the run should continue a real life.")
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
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
    if args.base:
        # Inherit a real training life and swap only the weights. Safe here
        # BECAUSE the weights differ from the base only in the placement
        # pathway -- verified tensor by tensor: of 27 shared tensors, 18 are
        # bit-identical and the 9 that changed are card_id_embed, place_ctx and
        # place_up. The trunk, LSTM, critic and aux head are untouched, so the
        # inherited Adam moments still belong to the parameters that produced
        # them. Do NOT use --base across a change that moves the trunk.
        base = args.base if os.path.isabs(args.base) else os.path.join(here, args.base)
        ckpt = torch.load(base, map_location="cpu", weights_only=False)
        prev = ckpt["model"]
        shared = set(prev) & set(model)
        moved = [k for k in sorted(shared) if not torch.equal(prev[k].float(),
                                                              model[k].float())]
        ckpt["model"] = model
        ckpt["optimizer"] = _widen_optimizer(ckpt.get("optimizer"), prev, model)
        atomic_save(ckpt, args.dst)
        print(f"wrote {args.dst}: {len(model)} tensors "
              f"({len(set(model) - set(prev))} new, {len(moved)} changed, "
              f"{len(shared) - len(moved)} identical), inheriting training "
              f"state from {args.base} at episode {ckpt.get('episodes_completed')}")
        print(f"  changed: {moved}")
        return

    from python_ai.models.net import MicroRoyaleNet
    from python_ai.rl.curriculum import CURRICULUM_STAGES
    probe = MicroRoyaleNet(num_ability_slots=0)
    fresh_opt = torch.optim.Adam(probe.parameters(), lr=3e-4).state_dict()

    ckpt = {
        "model": model,
        "optimizer": fresh_opt,
        "curriculum_stage": args.stage,
        # STAMP THE TABLE, or load_state_dict reads this index as a legacy
        # six-rung one and remaps it by horizon: --stage 5 silently became
        # rung 10, the top of the ladder (audit 04 C4).
        "teacher_table_size": len(CURRICULUM_STAGES),
        "stage_start_episode": args.episodes,
        "episodes_completed": args.episodes,
        "outcome_history": deque(maxlen=100),
        "phase": "mirror",
        "phase_deck_episode_start": args.episodes,
        "random_phase_episode_start": args.episodes,
        "deck_curriculum_stage": args.stage,
    }
    atomic_save(ckpt, args.dst)
    print(f"wrote {args.dst}: {len(model)} tensors, resuming at episode "
          f"{args.episodes}, stage {args.stage}")


if __name__ == "__main__":
    main()
