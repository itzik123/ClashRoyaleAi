"""Forward+backward cost of ONE PPO update chunk, at the shape the update
really runs at -- the instrument for the 15% throughput budget.

WHY THIS FILE EXISTS. Phase 4 changes three things inside the network
(receptive field, scalar encoder, BPTT horizon) and every one of them is an
accuracy bet paid for in wall clock. This repo has an explicit rule against
answering "is it better" from a forward pass, but "what does it COST" is
exactly a forward-pass question, and it is the one that decides whether a
change is admissible at all: 943 ep/hour on a CPU-only box, with CLAUDE.md
putting the CNN trunk at 43% of update wall-clock and the placement head at
41%.

THE SHAPE IS THE MEASUREMENT. `rl/ppo.py` runs the trunk on a FLAT (L*B) batch
and loops only the LSTMCell -- that asymmetry is why `forward_sequence` was
worth 1.82x, and a benchmark at rollout shape (B=8) would measure a different
network. L and B are derived from PPOConfig here rather than restated, so this
cannot drift away from the update it claims to model.

TWO VARIANTS IN ONE PROCESS, INTERLEAVED. `tools/audit/collision_bench.cpp`
holds both the old and new collision paths in a single binary for a reason this
file inherits: a CPU A/B split across two processes measures thermal state,
BLAS thread placement and scheduler luck at least as much as it measures the
code. `compare()` therefore alternates A,B,A,B... and reports MEDIANS, so drift
during the run lands on both arms equally. Medians, not means, because the
tail here is OS preemption -- one descheduled repeat moves a mean and not a
median.

Run it:

    python_ai/venv/Scripts/python.exe -m python_ai.eval.profile_architecture
"""
import argparse
import os
import statistics
import sys
import time

import torch

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.rl.config import PPOConfig  # noqa: E402
from python_ai.rl.optim_step import clip_and_step  # noqa: E402


def update_chunk_shape(cfg=None):
    """(L, B) of one PPO minibatch, derived from the config, never restated.

    Mirrors rl/ppo.py: `segments_per_rollout` chunks of `bptt_chunk` steps are
    split across `num_minibatches`, so one minibatch is B segments of L steps
    and the trunk sees L*B flat rows.
    """
    cfg = cfg or PPOConfig()
    segments = cfg.segments_per_rollout
    return cfg.bptt_chunk, max(1, segments // cfg.num_minibatches)


def chunks_per_update(cfg=None):
    cfg = cfg or PPOConfig()
    return cfg.ppo_epochs * cfg.num_minibatches


def make_chunk_runner(net, L, B, obs_dim=None, seed=0):
    """Build the batch and optimizer ONCE; return a closure that runs one chunk.

    Allocation is hoisted out deliberately. The observation batch alone is
    L*B*obs_dim floats -- 500 x 13,976 = 28 MB -- and re-allocating it per timed
    round churns enough memory to move the arm that happens to run after it.
    That is not a hypothetical: the first version of this file rebuilt the batch
    and an Adam state per round and reported a baseline 35% slower than the same
    net measured alone.
    """
    obs_dim = obs_dim or (net.spatial_size + net.scalar_size)
    g = torch.Generator().manual_seed(seed)
    flat = torch.randn(L * B, obs_dim, generator=g)
    hx = torch.zeros(B, net.LSTM_HIDDEN)
    cx = torch.zeros_like(hx)
    card_actions = torch.randint(0, net.hand_size + 1, (L, B), generator=g)
    cover_slot = torch.randint(0, net.hand_size + 1, (L, B), generator=g)
    reset_seq = torch.ones(L, B)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    def run_one():
        """Same call sequence as rl/ppo.py: the budget is on the whole update."""
        opt.zero_grad(set_to_none=True)
        feats, cembeds, spatial, hires = net.extract_features_hires(flat)
        feats = feats.view(L, B, -1)
        hires = hires.view(L, B, *hires.shape[1:])
        spatial = spatial.view(L, B, *spatial.shape[1:])
        obs_seq = flat.view(L, B, -1)
        cembeds = cembeds.view(L, B, net.hand_size + 1, -1)
        cmask = net.affordability_mask(flat).view(L, B, net.hand_size + 1)
        cl, pl, vals, aux, _, cf_pl = net.forward_sequence(
            feats, cembeds, spatial, obs_seq, cmask, card_actions,
            reset_seq, (hx, cx), extra_card_idx_seq=cover_slot,
            hires_seq=hires)
        loss = (cl.float().nan_to_num().pow(2).mean()
                + pl.float().nan_to_num().pow(2).mean()
                + vals.float().pow(2).mean() + aux.float().pow(2).mean()
                + cf_pl.float().nan_to_num().pow(2).mean())
        loss.backward()
        # The real update clips before stepping (rl/ppo.py -> clip_and_step),
        # and clipping 1.9M parameters is not free, so timing a bare
        # `opt.step()` here would under-report the update it claims to model.
        clip_and_step(opt, net.parameters(), 0.5)

    return run_one


def time_once(run_one):
    t0 = time.perf_counter()
    run_one()
    return 1000.0 * (time.perf_counter() - t0)


def time_update_chunk(net, L, B, repeats=12, warmup=3, obs_dim=None):
    """Median ms for one fwd+bwd+step at BPTT-minibatch shape."""
    run_one = make_chunk_runner(net, L, B, obs_dim=obs_dim)
    for _ in range(warmup):
        run_one()
    return statistics.median(time_once(run_one) for _ in range(repeats)), None


def param_table(net):
    """Trainable parameters per top-level module, plus the total."""
    rows = {}
    for name, mod in net.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if n:
            rows[name] = n
    direct = sum(p.numel() for p in net.parameters(recurse=False)
                 if p.requires_grad)
    if direct:
        rows["(direct params)"] = direct
    rows["TOTAL"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return rows


def compare(factories, L=None, B=None, repeats=12, cfg=None, shapes=None):
    """Interleaved A/B/A/B... so drift lands on every arm equally.

    `factories` is {label: callable-returning-net}. Nets are built ONCE up
    front (construction cost is not what we are measuring) and then timed in
    round-robin order.
    """
    cfg = cfg or PPOConfig()
    if L is None or B is None:
        L, B = update_chunk_shape(cfg)
    nets = {label: make() for label, make in factories.items()}
    for net in nets.values():
        net.train()
    # `shapes` lets one arm run at a different (L, B) -- needed to compare BPTT
    # horizons, where the whole point is that the shape changes. It stays a
    # fair comparison only because L*B (the flat rows through the trunk, 84% of
    # the update) is held constant by construction; the caller is responsible
    # for that and _fmt prints the rows so it can be checked.
    shapes = shapes or {}
    runners = {label: make_chunk_runner(net, *shapes.get(label, (L, B)))
               for label, net in nets.items()}

    per_arm = {label: [] for label in nets}
    # A full warmup round for EVERY arm before any timed round, so no arm pays
    # another's lazy-allocation cost.
    for run_one in runners.values():
        run_one()
        run_one()
    # ORDER IS ALTERNATED, not merely round-robin. Measured on this box with
    # two IDENTICAL arms, a fixed order reported the second arm 6.8% slower --
    # it inherits the cache the first arm just evicted, every round, so the bias
    # never averages out and lands entirely on whichever arm is listed last.
    # That is larger than the effects being measured here, so a fixed order
    # would have "found" a cost in a change that made none. Reversing every
    # other round puts each arm first half the time.
    order = list(runners)
    for r in range(repeats):
        for label in (order if r % 2 == 0 else order[::-1]):
            per_arm[label].append(time_once(runners[label]))

    return {
        "L": L, "B": B, "flat_rows": L * B,
        "chunks_per_update": chunks_per_update(cfg),
        "arms": {
            label: {
                "median_ms": statistics.median(v),
                "min_ms": min(v),
                "params": param_table(nets[label]),
            }
            for label, v in per_arm.items()
        },
    }


def _fmt(result, baseline_label):
    L, B = result["L"], result["B"]
    n = result["chunks_per_update"]
    base = result["arms"][baseline_label]
    lines = [
        f"BPTT chunk L={L}  minibatch segments B={B}  flat rows={L * B}",
        f"one PPO update = {n} chunks",
        "",
        f"{'arm':<28}{'ms/chunk':>10}{'s/update':>11}{'params':>12}{'vs base':>10}",
    ]
    for label, arm in result["arms"].items():
        ratio = arm["median_ms"] / base["median_ms"]
        lines.append(
            f"{label:<28}{arm['median_ms']:>10.1f}"
            f"{n * arm['median_ms'] / 1000:>11.2f}"
            f"{arm['params']['TOTAL']:>12,}{ratio:>9.3f}x")
    lines.append("")
    for label, arm in result["arms"].items():
        lines.append(f"-- {label} parameters --")
        for k, v in arm["params"].items():
            lines.append(f"     {k:<24}{v:>12,}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()

    factories = {
        "baseline (no context)": lambda: MicroRoyaleNet(
            num_ability_slots=0, context_dilations=()),
        "B1 dilated context": lambda: MicroRoyaleNet(num_ability_slots=0),
    }
    result = compare(factories, repeats=args.repeats)
    print(_fmt(result, "baseline (no context)"))


if __name__ == "__main__":
    main()
