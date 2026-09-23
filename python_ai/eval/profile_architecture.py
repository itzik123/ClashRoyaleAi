"""Forward+backward cost of one PPO update chunk, at the shape the update really
runs at.

The shape is the measurement: rl/ppo.py runs the trunk on a flat (L*B) batch
and loops only the LSTMCell, so a benchmark at rollout shape would measure a
different network. L and B are derived from PPOConfig.

Variants run in one process, interleaved A,B,A,B... with the order reversed
every other round, and medians are reported, so thermal drift and OS preemption
land on every arm equally.

    python_ai/venv/Scripts/python.exe -m python_ai.eval.profile_architecture
"""
import argparse
import os
import statistics
import sys
import time

import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.rl.config import PPOConfig  # noqa: E402
from python_ai.rl.optim_step import clip_and_step  # noqa: E402


def update_chunk_shape(cfg=None):
    """(L, B) of one PPO minibatch, derived from the config.

    Mirrors rl/ppo.py: `segments_per_rollout` chunks of `bptt_chunk` steps
    split across `num_minibatches`, so the trunk sees L*B flat rows.
    """
    cfg = cfg or PPOConfig()
    segments = cfg.segments_per_rollout
    return cfg.bptt_chunk, max(1, segments // cfg.num_minibatches)


def chunks_per_update(cfg=None):
    cfg = cfg or PPOConfig()
    return cfg.ppo_epochs * cfg.num_minibatches


def make_chunk_runner(net, L, B, obs_dim=None, seed=0):
    """Build the batch and optimizer once; return a closure that runs one chunk.

    Allocation is hoisted out because re-allocating a ~28 MB batch per round
    churns enough memory to slow whichever arm runs next.
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
        """Same call sequence as rl/ppo.py."""
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
        # The real update clips before stepping, and clipping is not free.
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
    """Interleaved timing of every arm, so drift lands on each equally.

    `factories` is {label: callable-returning-net}; nets are built once up
    front.
    """
    cfg = cfg or PPOConfig()
    if L is None or B is None:
        L, B = update_chunk_shape(cfg)
    nets = {label: make() for label, make in factories.items()}
    for net in nets.values():
        net.train()
    # `shapes` lets an arm run at a different (L, B), for comparing BPTT
    # horizons. That is fair only while L*B, the flat rows through the trunk,
    # is held constant; the caller must ensure it.
    shapes = shapes or {}
    runners = {label: make_chunk_runner(net, *shapes.get(label, (L, B)))
               for label, net in nets.items()}

    per_arm = {label: [] for label in nets}
    # Warm every arm before any timed round, so no arm pays another's lazy
    # allocation.
    for run_one in runners.values():
        run_one()
        run_one()
    # Alternate the order: with a fixed order the second arm inherits the cache
    # the first evicted, and reads slower every round.
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
