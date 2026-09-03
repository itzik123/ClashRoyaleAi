"""Re-initialise the auxiliary heads in a checkpoint. One-shot, for a deck change.

WHY A CHECKPOINT NEEDS THIS AT ALL
----------------------------------
`aux_card_head` predicts WHICH CARD the opponent plays next, over all 185 card
ids. A checkpoint trained against the 2.6 mirror has only ever seen 8 of them,
so its head is not merely untrained on the rest -- it is CONFIDENT about a
distribution that no longer exists.

Measured on 2026-09-03, resuming `model_weights_phase5.pth` into the 16-deck
pool: the head's cross-entropy read **13.76**, against **ln(185) = 5.22** for
predicting uniformly. Worse than uniform is the signature of a stale classifier,
and it is not a harmless readout: the head reads `hx` with no detach, so that
error backpropagates through the LSTM and the entire trunk. The weighted term
reached ~7x the actor loss and the policy's win rate fell 0.58 -> 0.00 in 126
episodes.

`ppo.py`'s stale-head cap bounds how hard that can pull. This tool removes the
cause rather than the symptom: a freshly initialised head starts at ~uniform and
relearns on the distribution it will actually see, instead of spending thousands
of episodes unlearning a confident wrong answer.

WHAT IT DOES NOT TOUCH
----------------------
Only the two auxiliary heads. The trunk, the LSTM, the card head, the placement
heads and the critic are left exactly as they are -- this is not a partial
restart, and the policy that comes out plays identically on its first step. The
optimizer moments for the reset tensors are zeroed too, or Adam would carry the
old head's momentum into the new one and undo the reset over the first updates.

`cycle_id_head` is reset for the same reason even though its gradient is
detached from everything upstream: its readout is what the console reports as
`CycleId`, and a stale one makes that diagnostic lie.

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

#: Prefixes whose parameters are re-initialised. Both are auxiliary readouts,
#: neither is on the acting path.
AUX_PREFIXES = ("aux_card_head.", "cycle_id_head.")


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(
        python_ai.PACKAGE_DIR, path)


def reset(state_dict, optim_state=None, prefixes=AUX_PREFIXES, seed=0):
    """Re-init matching tensors in place. Returns (names, n_params)."""
    gen = torch.Generator().manual_seed(seed)
    touched, n = [], 0
    # Index of parameter name -> position, so the optimizer's integer-keyed
    # state can be cleared for exactly the tensors that moved. Optimizer state
    # is keyed by ORDER, which is why this has to be derived from the same
    # ordered dict rather than guessed.
    order = {name: i for i, name in enumerate(state_dict)}
    for name, t in state_dict.items():
        if not name.startswith(prefixes):
            continue
        if name.endswith(".bias"):
            t.zero_()
        else:
            # Same scheme nn.Linear uses for a fresh layer, so the head starts
            # where a newly constructed one would -- near-uniform logits.
            tmp = torch.empty_like(t)
            nn.init.kaiming_uniform_(tmp, a=5 ** 0.5, generator=gen)
            t.copy_(tmp)
        touched.append(name)
        n += t.numel()
        if optim_state is not None:
            st = optim_state.get("state", {})
            key = order[name]
            if key in st:
                # Adam would otherwise carry the OLD head's first and second
                # moments into the new weights and walk them straight back
                # toward the distribution being discarded.
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

    torch.save(ck, dst)
    total = sum(t.numel() for t in ck["model"].values())
    print(f"reset {len(names)} tensors, {n:,} parameters "
          f"({n / total:.2%} of {total:,}) in {os.path.basename(dst)}")
    for name in names:
        print(f"  {name}")
    print("\nthe acting path (trunk, LSTM, card head, placement, critic) is "
          "untouched;\nAdam moments for the reset tensors are zeroed.")


if __name__ == "__main__":
    main()
