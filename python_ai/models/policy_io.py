"""Loading a trained policy, and the shapes a caller needs before it has one.

WHY THIS EXISTS
---------------
Before this module, any script that merely wanted to load a checkpoint wrote

    from python_ai.trainers.expert_iteration import load_net

and eighteen of them did. `expert_iteration.py` was then a 1,143-line
experiment script whose own imports pulled in `bc_pretrain`, the search harness,
`train` and `gym_wrapper` -- so reading one `.pth` file dragged in both PPO
trainers, the behaviour-cloning module and the search harness. A one-line probe
paid for the entire training stack, and every one of those modules became
impossible to change without considering eighteen callers that never wanted it.

(The 2026-08-20 restructuring split that file four ways and gave the search its
own package, so the magnet is smaller now -- but the rule that produced this
module is unchanged, and `tests/test_package_layout.py` now enforces it: nothing
under `models/` may import a trainer, an environment or an eval harness.)

That is a dependency magnet: a module acquires a useful helper, and the helper's
consumers inherit everything else the module happens to import. The fix is not
to make `expert_iteration` smaller -- it is that "load a policy" was never an
expert-iteration concern in the first place.

`load_state_dict_flexible` moved here from `train.py` for the same reason. It
was placed there because `train_selfplay.py` already imported from `train.py`
and the reverse would have been circular -- a real constraint, but it made four
non-training scripts import a 2,400-line trainer for one 30-line function.
Nothing here imports either trainer, so the cycle cannot recur.

WHAT BELONGS HERE
-----------------
Only what is needed to turn a checkpoint path into a ready-to-run net, plus the
recurrent shape that callers need in order to seed it. Search configuration
stays in `search/config.py`'s SearchCfg; the deployable configuration is in
`shipping.py`. This module deliberately knows nothing about either.
"""
import os

import torch

from python_ai.models.net import MicroRoyaleNet

# The recurrent width, re-exported from its single definition on the net itself.
# Five separate files used to type `LSTM_HIDDEN = 256` and four more imported it
# from `search_ab_test`, so the same literal existed nine ways -- exactly the
# duplicated-constant drift CLAUDE.md forbids. Deriving it means a change to the
# LSTM width propagates instead of silently disagreeing.
LSTM_HIDDEN = MicroRoyaleNet.LSTM_HIDDEN


# Deduped per unique context_label rather than per call: pipeline 2's
# PFSP calls set_historical_opponent -- and therefore load_state_dict_flexible --
# on EVERY episode reset in EVERY worker, so without this a single genuinely
# mismatched checkpoint floods the log with an identical line every episode for
# as long as PFSP keeps sampling it (observed: ~4000 repeats of one line).
# context_label already encodes the checkpoint path, so this dedupes naturally.
_warned_mismatches = set()


def _is_width_extension(key, saved, own_state):
    """True when `saved` is this net's matrix with input columns APPENDED.

    A layer that starts reading extra features -- `card_head`, `value_head`,
    `place_ctx` and `place_ctx_hi` when the 2026-08-28 cycle skip connection
    made them read `cat((hx, cycle_feat))` -- grows only along dim 1, and
    `torch.cat` puts the OLD features first. So the saved columns keep their
    meaning at `[:, :old_width]` and zeroing the remainder reproduces the old
    layer EXACTLY (`cat(hx, c) @ W.T == hx @ W_old.T + c @ 0`). Warm-starting
    it is therefore lossless, not an approximation.

    Deliberately NOT extended to dim 0. A matrix that gained output rows has
    new units with no trained counterpart, and a zero row there is not a no-op
    -- it is a 0.0 logit competing with trained ones. That case keeps falling
    through to the discard path. The asymmetry is the whole point: the check
    must be as narrow as the mathematical identity that justifies it.
    """
    own = own_state.get(key)
    return (own is not None
            and saved.dim() == 2 and own.dim() == 2
            and saved.shape[0] == own.shape[0]
            and saved.shape[1] < own.shape[1])


def load_state_dict_flexible(net, state_dict, context_label):
    """Loads state_dict into net. Returns True on a clean, fully-matching load.

    On an architecture mismatch (e.g. a card-roster change resizing the hand
    one-hot encoding, which is the only part of MicroRoyaleNet that depends on
    NUM_CARD_IDS -- see models/net.py's scalar_size), falls back to loading only
    the tensors whose shape still matches, leaving the rest at their fresh
    initialization instead of crashing outright. The CNN/LSTM/action heads are
    independent of NUM_CARD_IDS, so this warm-starts on everything except the
    one incompatible layer rather than discarding a whole checkpoint (and,
    upstream of this function, an entire opponent-history library, for
    pipeline 2's callers) over it.

    Returns False when this fallback path was taken -- the caller should NOT
    then load a paired optimizer state dict, since Adam's per-parameter
    buffers would be stale/mismatched for whatever just got reinitialized.
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
            # Two very different situations reach this branch and only one of
            # them loses trained weights:
            #   * `skipped` non-empty -- the checkpoint carried a tensor this
            #     net cannot use. Something trained was DISCARDED.
            #   * `skipped` empty -- every tensor the checkpoint had was loaded;
            #     the net simply has parameters that postdate it (e.g. the
            #     zero-initialized `place_hires` branch added 2026-08-14, which
            #     is an exact no-op at init). Nothing trained was lost.
            # Reporting both as "re-initialized" is how a harmless load gets
            # read as a discarded placement head.
            missing = sorted(set(own_state.keys()) - set(state_dict.keys()))
            if grown:
                # Report growth separately and BEFORE the discard case. These
                # were previously counted as discards, which is how losing the
                # card head, both placement contexts and the critic at once
                # read as an ordinary warm start (see
                # tests/test_flexible_load_grows_widened_heads.py).
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

    Accepts both shapes this project writes: a full training checkpoint (a dict
    with a "model" key alongside optimizer state and counters) and a bare
    state_dict. Callers should not have to know which one they were handed.
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
            # load_state_dict_flexible has just printed WHICH case this is:
            # tensors discarded (trained weights lost) or merely tensors the
            # checkpoint predates (nothing lost). Do not restate it as the
            # alarming case -- every checkpoint written before the 2026-08-14
            # `place_hires` branch takes this path harmlessly.
            print("    ^ see the line above for whether anything trained was lost.")
    return net
