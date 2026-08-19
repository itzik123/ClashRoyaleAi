"""Loading a trained policy, and the shapes a caller needs before it has one.

WHY THIS EXISTS
---------------
Before this module, any script that merely wanted to load a checkpoint wrote

    from python_ai.trainers.expert_iteration import load_net

and eighteen of them did. `expert_iteration.py` is a 1,143-line experiment
script whose own imports pull in `bc_pretrain`, `search_ab_test`, `train` and
`gym_wrapper` -- so reading one `.pth` file dragged in both PPO trainers, the
behaviour-cloning module and the search harness. A one-line probe paid for the
entire training stack, and every one of those modules became impossible to
change without considering eighteen callers that never wanted it.

That is a dependency magnet: a module acquires a useful helper, and the helper's
consumers inherit everything else the module happens to import. The fix is not
to make `expert_iteration` smaller -- it is that "load a policy" was never an
expert-iteration concern in the first place.

`load_state_dict_flexible` moved here from `train.py` for the same reason. It
was placed there because `train_selfplay.py` already imported from `train.py`
and the reverse would have been circular -- a real constraint, but it made four
non-training scripts import a 2,469-line trainer for one 30-line function.
Nothing here imports either trainer, so the cycle cannot recur.

WHAT BELONGS HERE
-----------------
Only what is needed to turn a checkpoint path into a ready-to-run net, plus the
recurrent shape that callers need in order to seed it. Search configuration
stays in `expert_iteration.SearchCfg`; the deployable configuration stays in
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


# Deduped per unique context_label rather than per call: train_selfplay.py's
# PFSP calls set_historical_opponent -- and therefore load_state_dict_flexible --
# on EVERY episode reset in EVERY worker, so without this a single genuinely
# mismatched checkpoint floods the log with an identical line every episode for
# as long as PFSP keeps sampling it (observed: ~4000 repeats of one line).
# context_label already encodes the checkpoint path, so this dedupes naturally.
_warned_mismatches = set()


def load_state_dict_flexible(net, state_dict, context_label):
    """Loads state_dict into net. Returns True on a clean, fully-matching load.

    On an architecture mismatch (e.g. a card-roster change resizing the hand
    one-hot encoding, which is the only part of MicroRoyaleNet that depends on
    NUM_CARD_IDS -- see model.py's scalar_size), falls back to loading only
    the tensors whose shape still matches, leaving the rest at their fresh
    initialization instead of crashing outright. The CNN/LSTM/action heads are
    independent of NUM_CARD_IDS, so this warm-starts on everything except the
    one incompatible layer rather than discarding a whole checkpoint (and,
    upstream of this function, an entire opponent-history library, for
    train_selfplay.py's callers) over it.

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
