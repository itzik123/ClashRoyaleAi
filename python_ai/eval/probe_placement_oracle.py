"""Can search actually aim this card, or only rank the head's noise?

A gate on distilling search into the placement head: distilling an expert that
cannot aim degrades the student.

On states where the card is in hand, affordable, and the board offers it the
most, five arms are played through the engine and scored by it:

    policy today        what the greedy policy does here (may be a no-op)
    card @ argmax       the card forced at its placement head's argmax cell
    card @ search top-k search over the head's own top-k cells (no widening)
    card @ search WIDE  search over the widened spatial proposal set
    card @ oracle       the best cell in that same widened set, by engine rollout

The score is net tower HP conceded over a fixed horizon (ally lost minus enemy
lost; lower is better). The oracle ranges over the widened set, not the whole
board, isolating the critic's ranking from proposal coverage. During a rollout
our side no-ops and the opponent plays, the convention `search.search_action`
uses.

`card @ argmax` is evaluated twice and the two must agree exactly; a non-zero
delta means the arms are not paired and every number is opponent randomness.

    ... -m python_ai.eval.probe_placement_oracle --weights model_weights_phase6.pth
    ... -m python_ai.eval.probe_placement_oracle --cards "The Log" --states 20
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.eval.measure_deck_matchups import log_catch_map  # noqa: E402
from python_ai.eval.stats import paired  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402
from python_ai.search import search as S  # noqa: E402
from python_ai.search.config import SearchCfg  # noqa: E402

#: The value map each card is aimed by, shared with probe_card_discrimination.
VALUE_MAP = {
    "Fireball": lambda o: tactics.spell_catch_map(o),
    "The Log": log_catch_map,
    # Cannon is the positive control: a harness that reproduces its known
    # result makes a null on the other cards trustworthy. Its opportunity is a
    # scalar, broadcast to share one code path.
    "Cannon": lambda o: np.full((1, 1), tactics.threat_level(o), np.float32),
}


def build_net(path, device):
    """Always via `policy_io.load_net`; a hand-rolled constructor once loaded a
    net with a random first conv layer.
    """
    return load_net(path, device)


def make_env(stage, seed, deck):
    """Phase 1's env as the run configures it: teacher opponent, a pool deck.

    The key is `opponent`; an unread key such as `opponent_kind` silently falls
    back to the C++ heuristic on the mirror.
    """
    env = gym_wrapper.MicroRoyaleEnv({
        "ai_deck": list(gym_wrapper.DEFAULT_DECK),
        "opp_deck": list(deck.card_ids),
        "opponent": "teacher",
        "teacher_stage": int(stage),
    })
    env.game.seed(seed)
    env.reset()
    return env


def tower_hp(sim):
    """(ally, enemy) total tower HP; board slots 0=King, 1=LEFT, 2=RIGHT."""
    ally = sum(max(0.0, sim.get_tower_hp(0, s)) for s in range(3))
    enemy = sum(max(0.0, sim.get_tower_hp(1, s)) for s in range(3))
    return ally, enemy


def rollout_score(snap, action, horizon):
    """Net tower HP conceded by taking `action` here, then no-opping. Lower is
    better.
    """
    sim = snap.snapshot()
    a0, e0 = tower_hp(sim)
    card, x, y = action
    res = sim.step(card, x, y)
    done = res.done
    for _ in range(horizon - 1):
        if done:
            break
        res = sim.step(S.NOOP, 0.0, 0.0)
        done = res.done
    a1, e1 = tower_hp(sim)
    return (a0 - a1) - (e0 - e1)


@torch.no_grad()
def collect_states(net, card_name, *, episodes, stage, seed0, device, want,
                   max_steps=400):
    """The `want` highest-opportunity states where the card is live.

    Each carries the snapshot the arms replay from and the net's hidden state
    on arrival; scoring a stored board with a zeroed hidden state would
    evaluate a different position.
    """
    from python_ai.opponents import deck_pool

    vmap = VALUE_MAP[card_name]
    deck = list(gym_wrapper.DEFAULT_DECK)
    card_id = next(c for c in deck if E.get_card_info(c)["name"] == card_name)
    # Round-robin over the pool, so every deck is represented.
    pool = deck_pool.load_pool()
    found = []
    for ep in range(episodes):
        env = make_env(stage, seed0 + ep, pool[ep % len(pool)])
        hidden = (torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN, device=device),
                  torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN, device=device))
        obs = env.game.get_observation_for_team(0)
        done, steps = False, 0
        while not done and steps < max_steps:
            o32 = np.asarray(obs, dtype=np.float32)
            obs_t = torch.tensor(o32, device=device).unsqueeze(0)
            cl, ce, sm, _, hn = S.policy_head(net, obs_t, hidden)
            mask = net.affordability_mask(obs_t)
            hand = net.hand_card_ids(obs_t)[0].tolist()
            slot = next((k for k, c in enumerate(hand) if c == card_id), None)
            if slot is not None and bool(mask[0, slot]):
                opp = float(vmap(o32).max())
                if opp > 1e-6:
                    found.append({
                        "opp": opp,
                        "snap": env.game.snapshot(),
                        "obs_t": obs_t,
                        "hidden": (hidden[0].clone(), hidden[1].clone()),
                        "slot": slot,
                        "ep": ep,
                        "tick": env.game.get_current_tick(),
                    })
            g_card, gx, gy, _ = S.greedy_from_logits(net, obs_t, cl, ce, sm, hn)
            obs, _, term, trunc, _ = env.step(
                {"card_index": g_card, "target_x": gx, "target_y": gy})
            done = bool(term or trunc)
            hidden = hn
            steps += 1
    # De-duplicate before ranking: consecutive decisions see almost the same
    # board, so a naive top-N returns copies of one moment. Keep states
    # MIN_TICK_GAP apart, and at most a third from any episode.
    MIN_TICK_GAP = 100
    per_ep_cap = max(1, want // 3)
    found.sort(key=lambda r: -r["opp"])
    kept, used = [], {}
    for r in found:
        ticks = used.setdefault(r["ep"], [])
        if len(ticks) >= per_ep_cap:
            continue
        if any(abs(r["tick"] - t) < MIN_TICK_GAP for t in ticks):
            continue
        ticks.append(r["tick"])
        kept.append(r)
        if len(kept) >= want:
            break
    print(f"  {len(found)} candidate states -> {len(kept)} kept "
          f"(>= {MIN_TICK_GAP} ticks apart, <= {per_ep_cap}/episode, "
          f"from {len(used)} episodes)")
    return kept


@torch.no_grad()
def arms_for_state(net, st, card_name, cfg, device, horizon):
    """Every arm's action for one state. Returns {arm: (card, x, y)}."""
    obs_t, hidden = st["obs_t"], st["hidden"]
    cl, ce, sm, _, hn = S.policy_head(net, obs_t, hidden)
    slot = st["slot"]

    # Policy today.
    g_card, gx, gy, _ = S.greedy_from_logits(net, obs_t, cl, ce, sm, hn)
    out = {"policy": (g_card, gx, gy)}

    # The card forced at its own placement argmax.
    idx = torch.tensor([slot], device=device)
    pl = net.placement_given_card(hn[0], ce, idx, obs_t, sm)
    pmask = net.placement_mask(obs_t, idx)
    pl = pl.masked_fill(~pmask, float("-inf"))
    cell = int(pl.argmax(dim=-1).item())
    x_t, y_t = net.cell_to_xy(torch.tensor([cell]))
    out["argmax"] = (slot, float(x_t.item()), float(y_t.item()))
    out["argmax_ctrl"] = out["argmax"]          # the identical-arms control

    # Make this card search's only expanded arm, so the comparison is about
    # cells.
    forced = torch.full_like(cl, float("-inf"))
    forced[0, slot] = 0.0
    greedy_here = out["argmax"]

    saved = S.WIDE_PROPOSAL_TOP1
    try:
        S.WIDE_PROPOSAL_TOP1 = 0.0                # never widen
        narrow = S.build_candidates(net, obs_t, forced, ce, sm, hn,
                                    greedy_here, 1, cfg.k_cells)
        S.WIDE_PROPOSAL_TOP1 = saved              # widen where flat
        wide = S.build_candidates(net, obs_t, forced, ce, sm, hn,
                                  greedy_here, 1, cfg.k_cells)
    finally:
        S.WIDE_PROPOSAL_TOP1 = saved

    out["_narrow_cands"] = narrow
    out["_wide_cands"] = wide
    return out


@torch.no_grad()
def critic_rank(net, snap, cands, cfg, hidden, device):
    """Pick from `cands` as search does: roll out, score by the critic."""
    if len(cands) == 1:
        return cands[0]
    final_obs, terminal = [], []
    for card, x, y in cands:
        sim = snap.snapshot()
        res = sim.step(card, x, y)
        done = res.done
        for _ in range(cfg.horizon - 1):
            if done:
                break
            res = sim.step(S.NOOP, 0.0, 0.0)
            done = res.done
        final_obs.append(sim.get_observation_for_team(0))
        terminal.append((done, float(res.reward)))
    batch = torch.tensor(np.asarray(final_obs, dtype=np.float32), device=device)
    feats, _, _ = net.extract_features(batch)
    n = len(cands)
    hx = hidden[0].expand(n, MicroRoyaleNet.LSTM_HIDDEN).contiguous()
    cx = hidden[1].expand(n, MicroRoyaleNet.LSTM_HIDDEN).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(feats, (hx, cx))
    scores = values.squeeze(-1).clone()
    for i, (done, reward) in enumerate(terminal):
        if done:
            scores[i] = S.terminal_score(reward, cfg.terminal_weight)
    return cands[int(scores.argmax().item())]


def run_card(net, card_name, args, device):
    cfg = SearchCfg(horizon=args.horizon, k_cards=1, k_cells=args.k_cells)
    t0 = time.time()
    states = collect_states(net, card_name, episodes=args.episodes,
                            stage=args.stage, seed0=args.seed, device=device,
                            want=args.states)
    print(f"  collected {len(states)} states in {time.time()-t0:.0f}s "
          f"(opportunity {states[-1]['opp']:.0f} .. {states[0]['opp']:.0f})")
    if len(states) < 6:
        print("  TOO FEW STATES -- not reporting a number off this.")
        return None

    order = ["policy", "argmax", "argmax_ctrl", "narrow", "wide", "oracle"]
    per_arm = {k: [] for k in order}
    n_wide, n_narrow = [], []

    for st in states:
        arms = arms_for_state(net, st, card_name, cfg, device, args.horizon)
        snap, hidden = st["snap"], st["hidden"]
        narrow_c, wide_c = arms["_narrow_cands"], arms["_wide_cands"]
        n_narrow.append(len(narrow_c))
        n_wide.append(len(wide_c))

        arms["narrow"] = critic_rank(net, snap, narrow_c, cfg, hidden, device)
        arms["wide"] = critic_rank(net, snap, wide_c, cfg, hidden, device)
        # Oracle: the best cell in the widened set, ranked by the engine.
        oracle_scores = [(rollout_score(snap, a, args.horizon), a)
                         for a in wide_c]
        arms["oracle"] = min(oracle_scores)[1]

        for k in order:
            per_arm[k].append(rollout_score(snap, arms[k], args.horizon))

    ctrl = float(np.abs(np.array(per_arm["argmax"])
                        - np.array(per_arm["argmax_ctrl"])).max())
    base = np.array(per_arm["policy"], dtype=np.float64)

    print(f"\n  {card_name}: net tower HP conceded over {args.horizon}s, "
          f"n={len(states)} states (LOWER IS BETTER)")
    print(f"  {'arm':<22}{'mean':>10}{'vs policy':>12}{'better in':>12}")
    labels = {"policy": "policy today", "argmax": "card @ argmax",
              "narrow": "card @ search top-k", "wide": "card @ search WIDE",
              "oracle": "card @ engine oracle"}
    rows = {}
    for k in ["policy", "argmax", "narrow", "wide", "oracle"]:
        v = np.array(per_arm[k], dtype=np.float64)
        delta = base - v                      # positive = this arm concedes less
        better = int((delta > 0).sum())
        rows[k] = {"mean": float(v.mean()), "delta": float(delta.mean()),
                   "better": better, "vec": v}
        tag = "" if k == "policy" else f"{delta.mean():>+12.0f}"
        bt = "" if k == "policy" else f"{better:>7}/{len(v)}"
        print(f"  {labels[k]:<22}{v.mean():>10.0f}{tag}{bt}")

    print(f"\n  identical-arms control (argmax twice), max |delta| = {ctrl:.1f}"
          f"   {'PAIRED' if ctrl < 1e-6 else '*** NOT PAIRED ***'}")
    print(f"  candidates per decision: narrow {np.mean(n_narrow):.1f}, "
          f"wide {np.mean(n_wide):.1f}")

    # The gate: how much of the oracle's headroom over the head's argmax does
    # the critic's ranking of the widened set collect?
    head = rows["argmax"]["vec"]
    wide = rows["wide"]["vec"]
    orac = rows["oracle"]["vec"]
    ceiling = float((head - orac).mean())
    got = float((head - wide).mean())
    capture = (got / ceiling) if abs(ceiling) > 1e-9 else float("nan")
    # `paired` reports b - a and both arms are HP conceded, so pass wide as a
    # and argmax as b.
    pr = paired(wide, head)
    print(f"\n  WIDE vs argmax: {got:>+8.0f} HP   "
          f"95% CI [{pr.lo:+.0f}, {pr.hi:+.0f}]   "
          f"{pr.better} better / {pr.worse} worse / {pr.tied} tied   "
          f"sign p = {pr.p:.4g}")
    print(f"  oracle ceiling: {ceiling:>+8.0f} HP   "
          f"widened search captures {capture*100:.0f}% of it")
    return {"card": card_name, "n": len(states), "rows": rows,
            "capture": capture, "ceiling": ceiling, "got": got,
            "p": pr.p, "ctrl": ctrl,
            "n_wide": float(np.mean(n_wide)),
            "n_narrow": float(np.mean(n_narrow))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="model_weights_phase6.pth")
    ap.add_argument("--cards", default="The Log,Fireball")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=20,
                    help="decision steps rolled forward when SCORING an arm")
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    device = torch.device("cpu")
    path = args.weights
    if not os.path.isabs(path):
        path = os.path.join(python_ai.PACKAGE_DIR, path)
    net = build_net(path, device)
    print(f"weights: {path}")
    print(f"horizon: {args.horizon} decisions   stage: {args.stage}   "
          f"WIDE_PROPOSAL_TOP1: {S.WIDE_PROPOSAL_TOP1}")

    results = []
    for card in [c.strip() for c in args.cards.split(",") if c.strip()]:
        print(f"\n=== {card} ===")
        r = run_card(net, card, args, device)
        if r:
            results.append(r)

    print("\n" + "=" * 66)
    print("GATE: does widened search have something real to teach this card?")
    for r in results:
        verdict = ("YES" if (r["got"] > 0 and r["p"] < 0.05)
                   else "NO -- distilling it would inject noise")
        print(f"  {r['card']:<12} {r['got']:>+7.0f} HP over the head's argmax, "
              f"p={r['p']:.3g}, {r['capture']*100:>3.0f}% of oracle   {verdict}")


if __name__ == "__main__":
    main()
