"""Loading a trained policy, and the shapes a caller needs before it has one.

Kept free of trainers, environments and eval harnesses
(`tests/test_package_layout.py`), so a probe that loads one checkpoint does not
import the training stack.
"""
import os

import torch

from python_ai.models.net import MicroRoyaleNet

# Re-exported from its single definition on the net.
LSTM_HIDDEN = MicroRoyaleNet.LSTM_HIDDEN


# Warn once per context label (which encodes the checkpoint path): PFSP reloads
# opponents on every reset, and one mismatched checkpoint would otherwise
# repeat the warning thousands of times.
_warned_mismatches = set()


def _is_width_extension(key, saved, own_state):
    """True when `saved` is this net's matrix with input columns appended.

    A layer that starts reading extra features (e.g. the heads after the cycle
    skip) grows only along dim 1, with the old features first, so zero-padding
    the new columns reproduces the old layer exactly. Not extended to dim 0: a
    new output row is a competing 0.0 logit, not a no-op.
    """
    own = own_state.get(key)
    return (own is not None
            and saved.dim() == 2 and own.dim() == 2
            and saved.shape[0] == own.shape[0]
            and saved.shape[1] < own.shape[1])


def load_state_dict_flexible(net, state_dict, context_label):
    """Load state_dict into net. Returns True on a clean, fully matching load.

    On an architecture mismatch, loads every tensor whose shape still matches
    (and zero-pads widened inputs), leaving the rest freshly initialised
    instead of discarding the checkpoint. Returns False then, and the caller
    should not load the paired optimizer state.
    """
    try:
        net.load_state_dict(state_dict)
        return True
    except RuntimeError:
        own_state = net.state_dict()
        compatible = {k: v for k, v in state_dict.items()
                      if k in own_state and v.shape == own_state[k].shape}
        grown = {k: v for k, v in state_dict.items()
                 if k not in compatible and _is_width_extension(k, v, own_state)}
        for k, v in grown.items():
            widened = torch.zeros_like(own_state[k])
            widened[:, :v.shape[1]] = v
            compatible[k] = widened
        skipped = sorted(set(state_dict.keys()) - set(compatible.keys()))
        own_state.update(compatible)
        net.load_state_dict(own_state)
        if context_label not in _warned_mismatches:
            _warned_mismatches.add(context_label)
            # Only a non-empty `skipped` discards trained weights; otherwise
            # the net merely has parameters the checkpoint predates.
            missing = sorted(set(own_state.keys()) - set(state_dict.keys()))
            if grown:
                # Report widened layers separately, so they are not read as
                # discards.
                print(f"[{context_label}] Widened layer(s) warm-started by "
                      f"zero-padding appended columns (lossless): "
                      f"{sorted(grown)}")
            if skipped:
                print(f"[{context_label}] Architecture mismatch -- warm-started "
                      f"{len(compatible)}/{len(state_dict)} tensor(s), "
                      f"DISCARDED (shape changed): {skipped}")
            else:
                print(f"[{context_label}] Checkpoint predates this architecture "
                      f"-- all {len(compatible)} of its tensor(s) loaded; "
                      f"fresh (not in checkpoint): {missing}")
        return False


def load_net(path, device, verbose=True):
    """A checkpoint path -> an eval-mode MicroRoyaleNet.

    Accepts a full training checkpoint (with a "model" key) or a bare
    state_dict.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = MicroRoyaleNet().to(device)
    clean = load_state_dict_flexible(net, state, path)
    net.eval()
    if verbose:
        eps = ckpt.get("episodes_completed", "?") if isinstance(ckpt, dict) else "?"
        print(f"  loaded {os.path.basename(path)} (episodes_completed={eps}, clean_load={clean})")
        if not clean:
            # load_state_dict_flexible has just said whether anything trained
            # was lost.
            print("    ^ see the line above for whether anything trained was lost.")
    return net
