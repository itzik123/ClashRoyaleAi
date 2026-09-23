"""Build a full training checkpoint for one arm of a controlled A/B.

A bare `{"model": state_dict}` takes train.py's legacy path, which resets
`episodes_completed` to 0; the placement entropy target for episode 0 is meant
for a fresh net, so the controller inflates a converged policy toward uniform
in both arms, indistinguishable from "the treatment did nothing". Every arm
therefore resumes through the full path: model + optimizer + curriculum + a
real episode count.

    python setup_ab_arm.py --src model_weights_hires.pth --dst _runs/x/model_weights.pth
"""
import argparse
import os
import sys
from collections import deque

import torch

from python_ai.rl.checkpointing import atomic_save

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401


def _widen_optimizer(opt_state, old_model, new_model):
    """Carry Adam moments across a checkpoint that gained parameters.

    torch matches optimizer state to parameters by positional index, and
    train_selfplay.py loads it unconditionally, so a size mismatch is fatal on
    resume. Dropping the state would discard the moments of every unchanged
    parameter, so indices are remapped by name. New parameters can land in the
    middle of the ordering, where a naive "keep, then append" remap would hand
    one module's moments to another.

    The old ordering is "current order minus the new names", valid only if
    nothing else moved; the count check below enforces that.
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
    # New parameters get no state: Adam initialises it lazily on the first
    # step.
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
                    help="curriculum rung, an index into the current table")
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

    # A real, empty-state Adam state_dict, without moments: torch's
    # load_state_dict raises on a bare {}, and train.py's full-resume path
    # requires the "optimizer" key. Moments are not carried because they belong
    # to whichever optimizer produced these weights; both arms get the same
    # fresh start. lr must match train.py's 3e-4, since load_state_dict
    # overwrites it.
    if args.base:
        # Inherit a real training life and swap only the weights. Safe only
        # when the weights differ from the base in the placement pathway alone,
        # so the inherited moments still belong to their parameters; never
        # across a change that moves the trunk.
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
        # Stamp the table, or load_state_dict reads this index as a legacy
        # six-rung one and remaps it by horizon.
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
