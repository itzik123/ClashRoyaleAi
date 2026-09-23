"""Re-initialise a checkpoint's auxiliary heads, for a change of opponent
distribution.

`aux_card_head` predicts the opponent's next card over all 185 ids. A
checkpoint trained against one deck is confidently wrong about any other
distribution, worse than uniform, and the head reads `hx` without detach, so
that error backpropagates through the LSTM and the trunk. A fresh head starts
near uniform and relearns on what it will actually see. (`ppo.py` also caps how
hard a stale head can pull.)

Only the two auxiliary heads change; the acting path is untouched and plays
identically on its first step. Their Adam moments are zeroed too, or the old
momentum would undo the reset. `cycle_id_head` is reset as well: its gradient
is detached, but its readout is the console's `CycleId`.

    python_ai/venv/Scripts/python.exe -m python_ai.tools.reset_aux_heads \\
        --weights model_weights_phase6.pth [--out model_weights_phase6b.pth]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from python_ai.rl.checkpointing import atomic_save  # noqa: E402

#: Prefixes re-initialised. Both are auxiliary readouts, off the acting path.
AUX_PREFIXES = ("aux_card_head.", "cycle_id_head.")


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(
        python_ai.PACKAGE_DIR, path)


def reset(state_dict, optim_state=None, prefixes=AUX_PREFIXES, seed=0):
    """Re-init matching tensors in place. Returns (names, n_params)."""
    gen = torch.Generator().manual_seed(seed)
    touched, n = [], 0
    # Name -> position: the optimizer's state is keyed by parameter order.
    order = {name: i for i, name in enumerate(state_dict)}
    for name, t in state_dict.items():
        if not name.startswith(prefixes):
            continue
        if name.endswith(".bias"):
            t.zero_()
        else:
            # nn.Linear's own init, so the head starts where a fresh one would:
            # near-uniform logits.
            tmp = torch.empty_like(t)
            nn.init.kaiming_uniform_(tmp, a=5 ** 0.5, generator=gen)
            t.copy_(tmp)
        touched.append(name)
        n += t.numel()
        if optim_state is not None:
            st = optim_state.get("state", {})
            key = order[name]
            if key in st:
                # Zero Adam's moments, or they walk the new weights back toward
                # the discarded head.
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st[key]:
                        st[key][k].zero_()
                if "step" in st[key]:
                    st[key]["step"] = torch.zeros_like(st[key]["step"]) \
                        if torch.is_tensor(st[key]["step"]) else 0
    return touched, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = _resolve(args.weights)
    dst = _resolve(args.out) if args.out else src
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "model" in ck):
        raise SystemExit("expected a full training checkpoint with a 'model' key")

    names, n = reset(ck["model"], ck.get("optimizer"), seed=args.seed)
    if not names:
        raise SystemExit("no auxiliary head tensors found -- nothing reset")

    # atomic_save, never torch.save: --out defaults to overwriting the source,
    # so a torn write would destroy it.
    atomic_save(ck, dst)
    total = sum(t.numel() for t in ck["model"].values())
    print(f"reset {len(names)} tensors, {n:,} parameters "
          f"({n / total:.2%} of {total:,}) in {os.path.basename(dst)}")
    for name in names:
        print(f"  {name}")
    print("\nthe acting path (trunk, LSTM, card head, placement, critic) is "
          "untouched;\nAdam moments for the reset tensors are zeroed.")


if __name__ == "__main__":
    main()
