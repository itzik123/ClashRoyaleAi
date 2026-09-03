"""Can search actually AIM this card, or can it only rank the head's noise?

WHY THIS EXISTS
---------------
Commit 1b77f27 measured widened search against an engine oracle for ONE card,
the Cannon, and found it captured 81% of the ceiling (+994 tower HP of a
possible +1223) where un-widened search got +236. That result is the entire
justification for distilling search into the placement head.

It was never measured for The Log or Fireball -- the two cards the placement
constraint was actually diagnosed on (`quality_hi` 0.217 and 0.685). The Log in
particular has the flattest head in the deck (top-1 0.0288), so it is the card
most dependent on widening and the least likely to have a critic that can rank
the widened set.

Distilling an expert that cannot aim is not neutral. CLAUDE.md records that
more data from a WEAK expert actively degrades selectivity (p0 rises faster than
p1, ratio 3.07 -> 2.31 -> 2.17). So this is a GATE, not a curiosity: it decides
whether expert iteration has anything real to teach these two cards.

WHAT IS MEASURED
----------------
On states where the target card is in hand, affordable, and the board offers it
the MOST (top of the collected opportunity distribution -- aiming only matters
where there is something to aim at), five arms are each PAID through the engine
and scored by the engine:

    policy today        whatever the greedy policy does here (may be a no-op)
    card @ argmax       the card forced at its placement head's argmax cell
    card @ search top-k search over the head's own top-k cells (widening OFF)
    card @ search WIDE  search over the widened spatial proposal set
    card @ oracle       the best cell in that same widened set, by ENGINE

Score is net tower HP conceded over a fixed horizon -- ally tower HP lost minus
enemy tower HP lost, so LOWER IS BETTER and one number covers a card that
defends and a card that trades.

THE ORACLE IS DELIBERATELY OVER THE WIDENED SET, NOT THE WHOLE BOARD. The
question distillation needs answered is "of the cells widening proposes, how
much of the available value does the CRITIC's ranking capture" -- that isolates
scoring quality from proposal coverage, and proposal coverage is the half
already known to work. A whole-board oracle would blend the two.

DURING A ROLLOUT OUR SIDE NO-OPS AND THE OPPONENT PLAYS. That isolates the
value of this one placement instead of blending it with whatever the policy
does next, and it is the same convention `search.search_action` uses for its own
candidate rollouts, so the arms are not being scored under a different physics
from the one search optimises against.

THE IDENTICAL-ARMS CONTROL IS LOAD-BEARING. `card @ argmax` is evaluated TWICE
and the two must agree exactly. They only do if `snapshot()` copies the
opponent's RNG state, i.e. if the arms are genuinely paired. If that control
prints a non-zero delta, every other number in the table is measuring opponent
randomness and none of them mean anything.

Usage:

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

#: The value map each card is aimed by -- the same two `probe_card_discrimination`
#: uses, so "opportunity" means one thing across both harnesses.
VALUE_MAP = {
    "Fireball": lambda o: tactics.spell_catch_map(o),
    "The Log": log_catch_map,
    # Cannon is the POSITIVE CONTROL, not a target. Commit 1b77f27 measured it
    # at +1223 HP of oracle ceiling with widening capturing 81%, so this
    # harness reproducing that is what makes a NULL on the other two
    # trustworthy rather than just evidence the harness is broken. Its
    # opportunity is threat on our half -- a scalar, not a cell map, so it is
    # broadcast to the board shape purely to share one code path.
    "Cannon": lambda o: np.full((1, 1), tactics.threat_level(o), np.float32),
}


def build_net(path, device):
    """Always via `policy_io.load_net`.

    Hand-rolling this passed `len(DEFAULT_DECK)` as MicroRoyaleNet's first
    positional argument, which is NOT the deck size, and the loader then
    discarded `cnn_trunk.0.weight` as a shape mismatch -- a probe running on a
    randomly-initialised first conv layer, reporting numbers the whole time.
    """
    return load_net(path, device)


def make_env(stage, seed, deck):
    """Phase 1's env as the RUN configures it: teacher opponent, a pool deck.

    THE KEY IS `opponent`, NOT `opponent_kind`. The first version of this probe
    passed `opponent_kind` -- which `gym_wrapper` does not read -- so it
    silently fell back to "builtin" and measured against the C++
    HeuristicOpponent on the 2.6 MIRROR: exactly the distribution this file's
    own docstring criticises `expert_collect` for using, and the one that ranks
    16th of 16 on opportunity for these cards. A config key that is ignored
    rather than rejected is the same silent-default hazard as the four stale
    arena copies.
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
    """(ally, enemy) total tower HP, board slots 0=King, 1=LEFT, 2=RIGHT."""
    ally = sum(max(0.0, sim.get_tower_hp(0, s)) for s in range(3))
    enemy = sum(max(0.0, sim.get_tower_hp(1, s)) for s in range(3))
    return ally, enemy


def rollout_score(snap, action, horizon):
    """Net tower HP conceded by taking `action` here, then no-opping.

    LOWER IS BETTER. One number covers defence (ally HP preserved) and trade
    (enemy HP taken), which a card like Fireball does both of.
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
    """States where the card is live and the board offers it the most.

    Returns the `want` highest-opportunity states, each carrying the snapshot
    the arms are replayed from and the hidden state the net had on arrival --
    the LSTM's memory is part of the state, and scoring a stored board with a
    zeroed hidden state would evaluate a different position.
    """
    from python_ai.opponents import deck_pool

    vmap = VALUE_MAP[card_name]
    deck = list(gym_wrapper.DEFAULT_DECK)
    card_id = next(c for c in deck if E.get_card_info(c)["name"] == card_name)
    # Round-robin over the pool, so every deck is represented rather than
    # whichever ones PFSP would have favoured.
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
    # DE-DUPLICATE BEFORE RANKING. `skip_frames = 10` means consecutive
    # decisions see almost the same board, so a naive "top N by opportunity"
    # happily returns N copies of a single moment -- n=16 that is really n=2,
    # with paired statistics computed over the duplicates. States are kept at
    # least MIN_TICK_GAP apart within an episode, and no episode may supply
    # more than a third of the sample.
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

    # policy today -- this project's own definition of "the policy playing".
    g_card, gx, gy, _ = S.greedy_from_logits(net, obs_t, cl, ce, sm, hn)
    out = {"policy": (g_card, gx, gy)}

    # the card forced at its own placement argmax
    idx = torch.tensor([slot], device=device)
    pl = net.placement_given_card(hn[0], ce, idx, obs_t, sm)
    pmask = net.placement_mask(obs_t, idx)
    pl = pl.masked_fill(~pmask, float("-inf"))
    cell = int(pl.argmax(dim=-1).item())
    x_t, y_t = net.cell_to_xy(torch.tensor([cell]))
    out["argmax"] = (slot, float(x_t.item()), float(y_t.item()))
    out["argmax_ctrl"] = out["argmax"]          # the identical-arms control

    # Force this card to be search's ONLY expanded arm, so the comparison is
    # about CELLS. Left to itself search would often expand a different card
    # and the arms would stop being about placement at all.
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
    """Pick from `cands` the way search does: roll out, score by the critic."""
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
        # ORACLE: the best cell in the widened set, ranked by the ENGINE.
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

    # The number the gate turns on: how much of the oracle's headroom over the
    # head's own argmax does the critic's ranking of the widened set collect?
    head = rows["argmax"]["vec"]
    wide = rows["wide"]["vec"]
    orac = rows["oracle"]["vec"]
    ceiling = float((head - orac).mean())
    got = float((head - wide).mean())
    capture = (got / ceiling) if abs(ceiling) > 1e-9 else float("nan")
    # `paired` reports b - a. Both arms are "HP conceded", so the improvement
    # widening buys is argmax - wide: pass wide as a and argmax as b.
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
