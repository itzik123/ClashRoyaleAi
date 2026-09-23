"""One guarded optimizer step, used by every training loop in the package.

A single non-finite gradient turns every parameter NaN in one step,
permanently: `clip_grad_norm_` multiplies the NaN through rather than bounding
it, and Adam's moments carry it forward. The step is dropped instead, since a
non-finite gradient has no descent direction. `tests/test_rl_optim_step.py`
fails on an optimizer loop written without it. It matters most for the
exploiter, whose NaN snapshot would crash the main run when sampled from the
shared pool.
"""
import torch
import torch.nn as nn


def clip_and_step(optimizer, parameters, max_norm):
    """Clip the gradients, then step only if they are finite.

    Returns True if the optimizer stepped. `parameters` may be a generator; it
    is consumed exactly once.
    """
    total_norm = nn.utils.clip_grad_norm_(parameters, max_norm)
    if not bool(torch.isfinite(total_norm)):
        # Clear the poisoned gradients so the next backward does not accumulate
        # onto them.
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True
