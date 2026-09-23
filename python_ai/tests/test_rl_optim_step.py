"""One guarded optimizer step, used by every loop that takes one.

A single non-finite gradient turns every parameter to NaN permanently:
`clip_grad_norm_` multiplies the NaN through the clip, and Adam's moments carry
it forward. The exploiter's loop also snapshots into the shared PFSP pool, so a
poisoned burst would crash the main run. The guard lives in one function, and
the static test at the bottom stops a new loop from skipping it.
"""
import copy

import pytest
import torch
import torch.nn as nn


def _net_and_opt():
    torch.manual_seed(0)
    net = nn.Linear(4, 2)
    return net, torch.optim.Adam(net.parameters(), lr=1e-2)


def test_a_finite_gradient_steps_and_reports_true():
    from python_ai.rl.optim_step import clip_and_step
    net, opt = _net_and_opt()
    before = net.weight.detach().clone()
    opt.zero_grad()
    net(torch.randn(3, 4)).sum().backward()
    assert clip_and_step(opt, net.parameters(), 0.5) is True
    assert not torch.equal(net.weight.detach(), before)


def test_a_nan_gradient_does_not_step_and_reports_false():
    from python_ai.rl.optim_step import clip_and_step
    net, opt = _net_and_opt()
    before = net.weight.detach().clone()
    opt.zero_grad()
    (net(torch.randn(3, 4)).sum() * float("nan")).backward()
    assert clip_and_step(opt, net.parameters(), 0.5) is False
    assert torch.equal(net.weight.detach(), before)
    assert torch.isfinite(net.weight).all()


def test_an_inf_gradient_does_not_step():
    from python_ai.rl.optim_step import clip_and_step
    net, opt = _net_and_opt()
    before = net.weight.detach().clone()
    opt.zero_grad()
    (net(torch.randn(3, 4)).sum() * float("inf")).backward()
    assert clip_and_step(opt, net.parameters(), 0.5) is False
    assert torch.equal(net.weight.detach(), before)


def test_the_dropped_gradient_is_cleared_not_left_to_accumulate():
    """A skipped step must clear its poisoned `.grad`, or the next backward adds
    to it.
    """
    from python_ai.rl.optim_step import clip_and_step
    net, opt = _net_and_opt()
    opt.zero_grad()
    (net(torch.randn(3, 4)).sum() * float("nan")).backward()
    clip_and_step(opt, net.parameters(), 0.5)
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in net.parameters())

    before = net.weight.detach().clone()
    opt.zero_grad()
    net(torch.randn(3, 4)).sum().backward()
    assert clip_and_step(opt, net.parameters(), 0.5) is True
    assert torch.isfinite(net.weight).all()
    assert not torch.equal(net.weight.detach(), before)


def test_a_generator_of_parameters_is_accepted_once():
    """`net.parameters()` is a generator; clipping consumes it, so iterating it a
    second time would silently see nothing.
    """
    from python_ai.rl.optim_step import clip_and_step
    net, opt = _net_and_opt()
    before = net.weight.detach().clone()
    opt.zero_grad()
    (net(torch.randn(3, 4)).sum() * float("nan")).backward()
    assert clip_and_step(opt, net.parameters(), 0.5) is False
    assert torch.equal(net.weight.detach(), before)


def test_no_optimizer_loop_in_the_package_steps_unguarded():
    """The drift guard: a new loop with a bare `clip_grad_norm_` /
    `optimizer.step()` pair reintroduces the whole failure.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("tests/", "venv/", "archive")):
            continue
        # `optim_step.py` is the guard itself. `profile_training.py` times a
        # raw `opt.step()` on a synthetic loss; guarding it would change what
        # it measures.
        if rel in ("rl/optim_step.py", "tools/profile_training.py"):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            code = line.split("#", 1)[0]
            if not re.search(r"\.step\(\)\s*$", code):
                continue
            if "optimizer.step()" not in code and "opt.step()" not in code:
                continue
            window = "\n".join(lines[max(0, i - 6):i])
            if "clip_and_step" in window:
                continue
            offenders.append(f"{rel}:{i + 1}: {line.strip()}")
    assert not offenders, (
        "unguarded optimizer steps:\n" + "\n".join(offenders))
