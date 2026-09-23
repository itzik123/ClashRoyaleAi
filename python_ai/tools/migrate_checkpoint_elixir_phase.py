"""Widen a checkpoint's scalar `extra` branch for a newly appended extra scalar.

Only one layer depends on NUM_EXTRA_SCALARS:

    scalar_mlp.extra : Linear(NUM_EXTRA_SCALARS, 12)

The four scalar branches concatenate to a fixed 64, so the LSTM and everything
downstream are untouched.

The new column is zero-padded rather than left for load_state_dict_flexible to
discard: discarding re-randomises the trained columns too, while a zero column
keeps the migrated net's output identical to the original's on every
observation.

Adam's `exp_avg` and `exp_avg_sq` mirror each parameter's shape, so both are
padded with zeros as well (no history, which is true); migrating the weight
alone produces a file that fails on `optimizer.load_state_dict`.

The target width is read from the engine, and the script refuses a stale .pyd
rather than reporting "nothing to do".

    python_ai/venv/Scripts/python.exe -m python_ai.tools.migrate_checkpoint_elixir_phase \\
        --in  python_ai/model_weights_phase4.pth \\
        --out python_ai/model_weights_phase5.pth
"""
import argparse
import os
import sys

# Put the repo root on sys.path, for script and -m invocation alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import python_ai  # noqa: F401  -- the sys.path append that finds the .pyd
from python_ai import engine_constants as EC
from python_ai.models.net import MicroRoyaleNet
from python_ai.envs import gym_wrapper
# atomic_save, never torch.save: a torn write would truncate the file a run
# resumes from.
from python_ai.rl.checkpointing import atomic_save

# The one parameter whose width tracks NUM_EXTRA_SCALARS.
EXTRA_W = "scalar_mlp.extra.weight"
EXTRA_B = "scalar_mlp.extra.bias"


def _param_index(net, name):
    """Position of `name` in net.parameters() order.

    Adam's state dict is keyed by that position, not by name; registration
    order is stable for a fixed architecture.
    """
    for i, (n, _p) in enumerate(net.named_parameters()):
        if n == name:
            return i
    raise KeyError(f"{name} is not a parameter of this net")


def _pad_columns(t, new_width):
    """Right-pad a [rows, old] tensor to [rows, new_width] with zeros."""
    rows, old = t.shape
    if old > new_width:
        raise ValueError(
            f"cannot shrink {tuple(t.shape)} to width {new_width}: this script "
            "only widens. A narrower target means the observation lost a "
            "scalar, which is not an append and needs a deliberate decision "
            "about which column to drop.")
    out = torch.zeros(rows, new_width, dtype=t.dtype, device=t.device)
    out[:, :old] = t
    return out


def migrate(ckpt, target_width, verbose=True):
    """Widen the extra branch in-place in `ckpt`. Returns (old_width, changed)."""
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if EXTRA_W not in sd:
        raise KeyError(
            f"{EXTRA_W} absent -- this checkpoint predates the four-branch "
            "ScalarEncoder (2026-08-27) and needs a different migration than "
            "this one.")

    old_width = sd[EXTRA_W].shape[1]
    if old_width == target_width:
        if verbose:
            print(f"  {EXTRA_W} is already {tuple(sd[EXTRA_W].shape)} -- nothing to do.")
        return old_width, False

    original = sd[EXTRA_W].clone()
    sd[EXTRA_W] = _pad_columns(sd[EXTRA_W], target_width)
    if verbose:
        print(f"  {EXTRA_W}  {tuple(original.shape)} -> {tuple(sd[EXTRA_W].shape)}")
        print(f"  {EXTRA_B}  {tuple(sd[EXTRA_B].shape)} (unchanged -- bias is per-OUTPUT)")

    # The trained columns must survive bit-exactly and the new ones must be
    # zero.
    assert torch.equal(sd[EXTRA_W][:, :old_width], original), \
        "trained columns were altered by the pad"
    assert torch.count_nonzero(sd[EXTRA_W][:, old_width:]) == 0, \
        "new columns are not zero"

    # --- optimizer moments ---
    opt = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
    if opt is None:
        if verbose:
            print("  no optimizer state in this file (inference-only checkpoint)")
        return old_width, True

    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    idx = _param_index(net, EXTRA_W)
    state = opt.get("state", {})
    # Keys may be int or str depending on how the file was written.
    entry = state.get(idx, state.get(str(idx)))
    if entry is None:
        if verbose:
            print(f"  optimizer has no state for param {idx} ({EXTRA_W}) -- "
                  "not yet stepped, nothing to pad")
        return old_width, True

    for moment in ("exp_avg", "exp_avg_sq"):
        if moment in entry and torch.is_tensor(entry[moment]):
            before = tuple(entry[moment].shape)
            entry[moment] = _pad_columns(entry[moment], target_width)
            if verbose:
                print(f"  optimizer[{idx}].{moment}  {before} -> "
                      f"{tuple(entry[moment].shape)}  (zeros = no history, which is true)")

    return old_width, True


def verify(path, target_width):
    """Load the migrated file as training will, and prove the pad is inert.

    Two claims that can fail independently: the file loads with no silent
    discard, and the net's output does not depend on the new scalar.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck

    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    hard = [k for k in list(missing) + list(unexpected) if "extra" in k]
    print(f"  strict-ish load: {len(missing)} missing, {len(unexpected)} unexpected, "
          f"{len(hard)} of them on the extra branch")
    if hard:
        raise SystemExit(f"!! extra-branch keys did not load: {hard}")
    net.eval()

    # Drive the real module: a check that re-reads the tensor it just wrote
    # cannot fail. Shapes come from the net: the encoder's `extra_start` is
    # relative to the scalar tail, not to the whole observation like the
    # engine's EXTRA_SCALARS_START.
    enc = net.scalar_mlp
    lo = torch.randn(4, net.scalar_size)
    hi = lo.clone()
    new_col = enc.extra_start + target_width - 1
    lo[:, new_col] = 0.0
    hi[:, new_col] = 1.0          # the whole range the phase scalar can take
    with torch.no_grad():
        a, b = enc(lo), enc(hi)
    delta = (a - b).abs().max().item()
    print(f"  max |encoder(new=0) - encoder(new=1)| = {delta:.3e}  (must be exactly 0)")
    if delta != 0.0:
        raise SystemExit("!! the new column is not inert -- the pad did not take")
    print("  the migrated net is bit-identical to the original on every "
          "observation it could already see.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="model_weights_phase4.pth",
                    help="checkpoint to read (relative paths resolve against python_ai/)")
    ap.add_argument("--out", dest="dst", default=None,
                    help="where to write (default: <in stem>.elixirphase.pth)")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing --out")
    a = ap.parse_args()

    src = a.src if os.path.isabs(a.src) else os.path.join(python_ai.PACKAGE_DIR, a.src)
    if not os.path.exists(src):
        raise SystemExit(f"!! no such checkpoint: {src}")

    if a.dst is None:
        stem, ext = os.path.splitext(src)
        dst = f"{stem}.elixirphase{ext}"
    else:
        dst = a.dst if os.path.isabs(a.dst) else os.path.join(python_ai.PACKAGE_DIR, a.dst)

    # Never write onto the input: a run may be resuming from it, and a failed
    # in-place rewrite is unrecoverable.
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("!! --out must differ from --in; migrate to a new file "
                         "so the original stays resumable if this goes wrong.")
    if os.path.exists(dst) and not a.force:
        raise SystemExit(f"!! {dst} exists; pass --force to overwrite.")

    target = EC.NUM_EXTRA_SCALARS
    # observation_size() is an instance method; derive the total as
    # ClashEnv::observationSize() does.
    print(f"engine reports NUM_EXTRA_SCALARS = {target}, "
          f"observation size = {EC.CYCLE_START + EC.CYCLE_BLOCK_SIZE}")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    print(f"\nread {os.path.basename(src)} "
          f"(episode {ck.get('episodes_completed', '?')})")

    sd = ck["model"] if "model" in ck else ck
    if sd[EXTRA_W].shape[1] == target:
        raise SystemExit(
            f"\n!! {EXTRA_W} is already width {target}, so there is nothing to\n"
            "   migrate. If you expected a change, the .pyd in python_ai/ is\n"
            "   STALE -- it still reports the pre-change NUM_EXTRA_SCALARS.\n"
            "   Rebuild the engine and run tools/audit/verify_pyd.py first;\n"
            "   migrating against a stale binding produces a file that is wrong\n"
            "   in a way nothing downstream will report.")

    print("\nmigrating:")
    old, changed = migrate(ck, target)

    atomic_save(ck, dst)
    print(f"\nwrote {dst}")

    print("\nverifying:")
    verify(dst, target)
    print(f"\nOK -- extra branch {old} -> {target}. "
          "Point CLASH_WEIGHTS at the new file, or rename it into place.")


if __name__ == "__main__":
    main()
