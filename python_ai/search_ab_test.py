"""1-ply decision-time search vs the raw policy: a PAIRED win-rate A/B.

Consumes `ClashRoyaleEnv.snapshot()` (perception/UPSTREAM_REQUESTS.md item 13).
At each decision the search arm proposes K candidate actions, rolls each one
forward on a throwaway copy of the live environment, scores the resulting
position with the critic, and plays the best. The policy arm plays the same
network greedily, exactly as `train_selfplay.evaluate_against_roster` does.

WHY PAIRED, AND WHY THAT IS THE POINT
-------------------------------------
The engine's RNG still cannot be seeded (UPSTREAM item 7 is open), so the usual
way to compare two agents is unpaired, and UPSTREAM item 7 works out the cost:
~1,568 episodes per arm to resolve a 5-point win-rate difference at 80% power.

snapshot() sidesteps that. Each trial resets ONE environment, snapshots it, and
hands both arms a bit-identical copy -- same shuffled opening hand, same
heuristic-opponent lane, same everything at t=0. The shared opening is removed
from the variance rather than averaged over, which is what pairing buys, and it
needs no engine RNG change at all.

WHAT THIS CANNOT TELL YOU
-------------------------
UPSTREAM item 13 records a previous attempt at this question whose confidence
interval came out 15x wider than the effect, and the honest reading of that was
"underpowered null", not "search does not work". Two things guard against
repeating it:

  * `--trials` is reported alongside a paired CI, and the CI is printed whether
    or not it excludes zero.
  * DEVIATION RATE is reported first. If search almost never disagrees with the
    greedy policy, the two arms are near-identical by construction and the
    win-rate comparison carries no information regardless of how many episodes
    are run. That diagnostic is cheap and it is the one that says whether the
    experiment measured anything at all.

Run:
    python_ai/venv/Scripts/python.exe python_ai/search_ab_test.py --trials 100
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clash_royale_env  # noqa: E402
from gym_wrapper import DEFAULT_DECK  # noqa: E402
from model import MicroRoyaleNet  # noqa: E402
from policy_io import LSTM_HIDDEN, load_state_dict_flexible  # noqa: E402

HAND_SIZE = clash_royale_env.ClashRoyaleEnv.HAND_SIZE
NOOP = HAND_SIZE  # the no-op arm of the card head is the column past the hand


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

@torch.no_grad()
def _policy_head(net, obs_t, hidden):
    """One network forward. Returns the pieces both arms need.

    Deliberately still returns FIVE values and calls the plain
    extract_features. Six callers across five files unpack exactly five, and
    model.py's own extract_features docstring records the same reasoning for
    keeping that wrapper rather than widening its signature. The hires map is
    recovered locally in _build_candidates instead -- see there.
    """
    card_mask = net.affordability_mask(obs_t)
    features, card_embeds, spatial_map = net.extract_features(obs_t)
    card_logits, _, _, value, hidden_next = net.step_lstm_and_card(features, hidden, card_mask)
    return card_logits, card_embeds, spatial_map, value, hidden_next


@torch.no_grad()
def _greedy_from_logits(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next):
    """argmax over MASKED card logits then argmax cell -- identical to
    train_selfplay.evaluate_against_roster, so the control arm IS this
    project's own definition of 'the policy playing'."""
    card_idx_t = card_logits.argmax(dim=-1)
    place_logits = net.placement_given_card(hidden_next[0], card_embeds, card_idx_t, obs_t, spatial_map)
    cell = place_logits.argmax(dim=-1)
    x_t, y_t = net.cell_to_xy(cell)
    return int(card_idx_t.item()), float(x_t.item()), float(y_t.item()), place_logits


@torch.no_grad()
def _build_candidates(net, obs_t, card_logits, card_embeds, spatial_map, hidden_next,
                      greedy, k_cards, k_cells):
    """Greedy action first, then alternatives from the top of each head.

    Greedy is candidate 0 deliberately: search then deviates only when the
    critic prefers something else, so it can never be worse than the policy
    BY THE CRITIC'S OWN MEASURE. Any loss it takes is the critic being wrong,
    which is a different and more interesting failure than search being wrong.
    """
    # Canonicalise a no-op greedy to (NOOP, 0, 0) before seeding `seen`.
    # ClashEnv::step ignores placement whenever cardIndex >= HAND_SIZE, so
    # (NOOP, gx, gy) and (NOOP, 0, 0) are THE SAME ACTION -- but as tuples they
    # differ, so without this the no-op is emitted twice and the two copies roll
    # out identically and score identically.
    #
    # That cost nothing for the win-rate A/B (the duplicate always ties, and
    # argmax returns the lower index, which is greedy). It is fatal for
    # distribution distillation: with the policy no-oping ~80-90% of the time,
    # the majority of rows ended up with exactly two candidates that were the
    # same action, giving a target with a measured median value spread of
    # EXACTLY 0.0 -- a uniform distribution over one action, carrying no signal
    # while still contributing gradient.
    if greedy[0] >= HAND_SIZE:
        greedy = (NOOP, 0.0, 0.0)
    cands = [greedy]
    seen = {greedy}

    # Build the pre-pool trunk activation ONCE for this decision. Without it
    # every placement_given_card below rebuilds it from obs internally
    # (model.py: `hires_map` defaults to None -> recomputed), so a k_cards=3
    # decision ran cnn_trunk[:2] up to four times on identical input -- on the
    # deadline-bounded path search_action_deadline exists to budget. Passing it
    # is the seam model.py documents for hot callers, and the two routes are
    # pinned bit-identical by
    # test_python_ai.test_recomputed_hires_equals_the_passed_one.
    hires_map = net.hires_features(obs_t)

    n_cards = card_logits.shape[-1]
    top_cards = torch.topk(card_logits[0], min(k_cards, n_cards)).indices
    for c in top_cards:
        if not torch.isfinite(card_logits[0, c]):
            continue  # masked: unaffordable, and playCard would drop it silently
        card_idx = int(c.item())
        if card_idx >= HAND_SIZE:
            key = (NOOP, 0.0, 0.0)  # placement is meaningless for the no-op
            if key not in seen:
                seen.add(key)
                cands.append(key)
            continue
        c_t = c.view(1)
        place_logits = net.placement_given_card(hidden_next[0], card_embeds, c_t, obs_t,
                                                spatial_map, hires_map=hires_map)
        top_cells = torch.topk(place_logits[0], k_cells).indices
        for cell in top_cells:
            if not torch.isfinite(place_logits[0, cell]):
                continue  # masked: illegal placement for this card class
            x_t, y_t = net.cell_to_xy(cell.view(1))
            key = (card_idx, float(x_t.item()), float(y_t.item()))
            if key not in seen:
                seen.add(key)
                cands.append(key)
    return cands


@torch.no_grad()
def _search_action(net, env, obs_t, card_logits, card_embeds, spatial_map, hidden_next,
                   greedy, cfg, device, return_details=False):
    """Roll every candidate forward on its own snapshot, score, pick the best.

    `return_details` appends the full (candidates, scores) pair as a 4th return
    value. Off by default, so every existing caller keeps the same 3-tuple and
    the arm that measured +0.319 is untouched. It exists for expert-iteration
    distillation: the argmax alone discards how MUCH better the winner was, and
    that margin is the only thing distinguishing "waiting is marginally better"
    from "playing here is a blunder".
    """
    cands = _build_candidates(net, obs_t, card_logits, card_embeds, spatial_map,
                              hidden_next, greedy, cfg.k_cards, cfg.k_cells)
    if len(cands) == 1:
        # One candidate carries no preference information at all -- there is
        # nothing to rank, so distillation must skip these rows rather than
        # train on a degenerate one-hot.
        if return_details:
            return greedy, False, 1, None
        return greedy, False, 1

    final_obs = []
    terminal = []
    for card_idx, x, y in cands:
        sim = env.snapshot()
        result = sim.step(card_idx, x, y)
        done = result.done
        for _ in range(cfg.horizon - 1):
            if done:
                break
            result = sim.step(NOOP, 0.0, 0.0)
            done = result.done
        final_obs.append(sim.get_observation_for_team(0))
        terminal.append((done, float(result.reward)))

    batch = torch.tensor(np.asarray(final_obs, dtype=np.float32), device=device)
    feats, _, _ = net.extract_features(batch)
    # Candidates are scored from the SAME pre-rollout hidden state. That is an
    # approximation -- the LSTM has not consumed the rollout's intervening
    # observations -- but it is common to every candidate and costs one batched
    # forward instead of horizon x K sequential ones. Advancing the recurrence
    # per candidate would be more faithful and roughly `horizon` times dearer.
    n = len(cands)
    hx = hidden_next[0].expand(n, LSTM_HIDDEN).contiguous()
    cx = hidden_next[1].expand(n, LSTM_HIDDEN).contiguous()
    _, _, _, values, _ = net.step_lstm_and_card(feats, (hx, cx))
    scores = values.squeeze(-1).clone()

    # A rollout that ENDED is not a position to be valued -- the critic's
    # estimate of a finished game is meaningless. Use the real outcome instead,
    # weighted to dominate any bootstrapped value.
    for i, (done, reward) in enumerate(terminal):
        if done:
            scores[i] = reward * cfg.terminal_weight

    best = int(scores.argmax().item())
    if return_details:
        return cands[best], best != 0, len(cands), (cands, scores.detach().cpu().numpy())
    return cands[best], best != 0, len(cands)


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------

@torch.no_grad()
def play_episode(net, env, device, use_search, cfg):
    hidden = (torch.zeros(1, LSTM_HIDDEN, device=device),
              torch.zeros(1, LSTM_HIDDEN, device=device))
    obs = env.get_observation_for_team(0)
    reward, steps, deviations, cand_total = 0.0, 0, 0, 0
    done = False

    while not done and steps < cfg.max_steps:
        obs_t = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
        card_logits, card_embeds, spatial_map, _, hidden_next = _policy_head(net, obs_t, hidden)
        greedy_card, gx, gy, _ = _greedy_from_logits(
            net, obs_t, card_logits, card_embeds, spatial_map, hidden_next)
        greedy = (greedy_card, gx, gy)

        if use_search:
            action, deviated, n_cands = _search_action(
                net, env, obs_t, card_logits, card_embeds, spatial_map,
                hidden_next, greedy, cfg, device)
            deviations += int(deviated)
            cand_total += n_cands
        else:
            action = greedy

        hidden = hidden_next
        result = env.step(action[0], action[1], action[2])
        obs, reward, done = result.observation, float(result.reward), result.done
        steps += 1

    return reward, steps, deviations, cand_total


def outcome_score(reward):
    """win 1.0 / draw 0.5 / loss 0.0 -- the same convention as the Elo roster."""
    if reward > 0.5:
        return 1.0
    if reward < -0.5:
        return 0.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100, help="paired episodes (2 games each)")
    ap.add_argument("--weights", default="model_weights_selfplay.pth")
    ap.add_argument("--horizon", type=int, default=4, help="decision steps rolled forward (1 = 1s)")
    ap.add_argument("--k-cards", type=int, default=3)
    ap.add_argument("--k-cells", type=int, default=2)
    ap.add_argument("--terminal-weight", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-ticks", type=int, default=3600)
    # The curriculum's hardest rung. At 1.0 the ep-64k policy wins ~100% of
    # games and BOTH arms saturate, so the paired delta is pinned at zero by a
    # ceiling rather than by search being useless -- measured, not assumed:
    # 4/4 trials came back 1.000 vs 1.000. A comparison needs an opponent with
    # headroom on both sides. CLAUDE.md records this policy family plateauing
    # around 0.86 at 1.5x, which is where the resolution is.
    ap.add_argument("--opp-elixir", type=float, default=1.5,
                    help="opponent elixir multiplier; 1.0 saturates at this skill level")
    ap.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0 = no limit")
    cfg = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = cfg.weights if os.path.isabs(cfg.weights) else os.path.join(here, cfg.weights)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = MicroRoyaleNet().to(device)
    clean = load_state_dict_flexible(net, state, weights_path)
    net.eval()

    episodes = ckpt.get("episodes_completed", "?") if isinstance(ckpt, dict) else "?"
    print(f"weights          : {os.path.basename(weights_path)} "
          f"(episodes_completed={episodes}, clean_load={clean})")
    print(f"search           : K<={1 + cfg.k_cards * cfg.k_cells} candidates, "
          f"horizon={cfg.horizon} steps ({cfg.horizon}s), critic-scored")
    print(f"opponent         : C++ HeuristicOpponent at {cfg.opp_elixir}x elixir")
    print(f"pairing          : both arms start from one snapshot of the same reset")
    print(f"trials           : {cfg.trials} paired ({2 * cfg.trials} episodes)")
    print()

    diffs, a_scores, b_scores = [], [], []
    dev_total, dev_steps, cand_total = 0, 0, 0
    a_time = b_time = 0.0
    started = time.perf_counter()

    for trial in range(cfg.trials):
        if cfg.time_budget and (time.perf_counter() - started) > cfg.time_budget:
            print(f"\n[time budget reached after {trial} trials]")
            break

        root = clash_royale_env.ClashRoyaleEnv(list(DEFAULT_DECK), list(DEFAULT_DECK), cfg.max_ticks)
        root.set_opponent_elixir_multiplier(cfg.opp_elixir)
        root.reset()
        base = root.snapshot()  # the shared opening both arms will play

        t0 = time.perf_counter()
        r_a, steps_a, _, _ = play_episode(net, base.snapshot(), device, False, cfg)
        a_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        r_b, steps_b, dev, cands = play_episode(net, base.snapshot(), device, True, cfg)
        b_time += time.perf_counter() - t0

        dev_total += dev
        dev_steps += steps_b
        cand_total += cands

        sa, sb = outcome_score(r_a), outcome_score(r_b)
        a_scores.append(sa)
        b_scores.append(sb)
        diffs.append(sb - sa)

        if (trial + 1) % 10 == 0:
            n = len(diffs)
            print(f"  trial {trial + 1:4d} | policy {np.mean(a_scores):.3f} "
                  f"search {np.mean(b_scores):.3f} | delta {np.mean(diffs):+.3f} "
                  f"| deviation {dev_total / max(1, dev_steps):.1%} "
                  f"| {(time.perf_counter() - started) / n:.1f}s/trial")

    n = len(diffs)
    if n == 0:
        print("no trials completed")
        return

    diffs = np.asarray(diffs)
    mean_d = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo, hi = mean_d - 1.96 * se, mean_d + 1.96 * se

    print("\n" + "=" * 68)
    print(f"trials (paired)        : {n}")
    print(f"policy   win rate      : {np.mean(a_scores):.4f}")
    print(f"search   win rate      : {np.mean(b_scores):.4f}")
    print(f"paired delta           : {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  significant?         : {'YES' if (lo > 0 or hi < 0) else 'no -- CI includes 0'}")

    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    print(f"  search better/worse/same : {wins} / {losses} / {ties}")

    print(f"\ndeviation rate         : {dev_total / max(1, dev_steps):.2%} "
          f"({dev_total} of {dev_steps} decisions)")
    print(f"mean candidates/decision: {cand_total / max(1, dev_steps):.2f}")
    if dev_total == 0:
        print("  !! search NEVER disagreed with the greedy policy. The two arms are")
        print("     the same agent, so the win-rate comparison above measures nothing")
        print("     whatever its CI says.")

    print(f"\nwall clock             : policy {a_time / n:.2f}s/ep, search {b_time / n:.2f}s/ep "
          f"({b_time / max(1e-9, a_time):.1f}x)")

    # Power check, stated up front rather than left for the reader to work out.
    if n > 1 and se > 0:
        detectable = 1.96 * se
        print(f"\nsmallest effect this n could resolve: +/-{detectable:.4f} "
              f"({detectable * 100:.1f} win-rate points)")
        if abs(mean_d) < detectable:
            needed = int(math.ceil((1.96 * diffs.std(ddof=1) / max(1e-6, abs(mean_d))) ** 2))
            print(f"observed effect is INSIDE the noise floor. Resolving an effect this "
                  f"size would need ~{needed} paired trials.")


if __name__ == "__main__":
    main()
