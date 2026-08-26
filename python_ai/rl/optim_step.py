"""One guarded optimizer step, for every loop in the package that takes one.

A single non-finite gradient turns EVERY parameter to NaN in one step, and
PERMANENTLY. `clip_grad_norm_` does not stop it -- it scales by
`max_norm / (total_norm + eps)`, and when `total_norm` is nan that factor is
nan too, so the clip MULTIPLIES the poison into every tensor rather than
bounding it. Adam's moment estimates then carry the NaN forward, so no later
clean batch recovers. Measured 2026-08-26 on a 32-tensor net: 32 of 32 NaN
after one bad step, still 32 of 32 after a clean one.

WHY THIS IS A MODULE AND NOT FOUR MORE COPIES OF AN `if`. There are five
optimizer loops here -- `rl/ppo.py`, `trainers/exploiter.py`,
`trainers/bc_pretrain.py`, `trainers/distill_tactics.py`,
`trainers/expert_distill.py` -- plus two in the measurement harnesses. Every
one of them had written the same unguarded pair. This package's own rule is
that a second copy of a decision is a scheduled defect; the guard is a decision.
`tests/test_rl_optim_step.py` carries a static check that fails on a sixth loop
written without it.

THE EXPLOITER IS THE ONE THAT ESCAPES ITS OWN PROCESS. It snapshots into the
SHARED PFSP pool, so a poisoned burst deposits an opponent whose logits are all
NaN -- and `Categorical` raises on those, crashing the MAIN run on whichever
reset happens to sample it. A burst that quietly learned nothing would have
been the mild version.

DROPPING THE STEP IS THE CORRECT RESPONSE, not scaling it down: there is no
descent direction in a non-finite gradient, so the right step size is zero. The
caller is told, so it can report it -- a guard that silently eats every update
is the no-learning failure it exists to prevent.
"""
import torch
import torch.nn as nn


def clip_and_step(optimizer, parameters, max_norm):
    """Clip `parameters`' gradients, then step ONLY if they are finite.

    Returns True if the optimizer stepped, False if the step was dropped.
    Callers should surface a False -- see `UpdateStats.nonfinite_skips`.

    `parameters` may be a generator (`net.parameters()` is one). It is consumed
    exactly once here; an implementation that iterated it a second time would
    silently see an empty sequence and report success on an unclipped step.
    """
    total_norm = nn.utils.clip_grad_norm_(parameters, max_norm)
    if not bool(torch.isfinite(total_norm)):
        # Clear the poisoned gradients rather than leaving them in `.grad` for
        # the next backward to accumulate ON TOP of -- otherwise one bad batch
        # takes the following GOOD one down with it.
        optimizer.zero_grad(set_to_none=True)
        return False
    optimizer.step()
    return True
