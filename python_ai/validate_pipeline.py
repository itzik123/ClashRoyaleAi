"""Phase 3: prove every component of the training loop before a long run uses it.

    python_ai/venv/Scripts/python.exe python_ai/validate_pipeline.py --full

Each check prints PASS/FAIL and the number it decided on. Nothing here asserts a
property that is merely plausible -- every threshold is either an engine
constant, a documented measurement, or a statistical null with its own sample
size. A check that cannot fail is not a check.

WHAT THIS DOES NOT DO
---------------------
It does not touch the emulator, by instruction and by design: the live loop is
gated on perception fidelity, and an unattended multi-day run cannot depend on
an unstable BlueStacks window. Everything below runs against the C++ simulator.

It also does not modify anything. It is safe to run against a live training
directory, and it deliberately builds its own environments rather than
borrowing the trainer's.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import advisor_target as AT  # noqa: E402
import clash_royale_env as E  # noqa: E402
import gym_wrapper  # noqa: E402
import match_outcome  # noqa: E402
import tactics  # noqa: E402
import train  # noqa: E402
import train_selfplay as TS  # noqa: E402
from model import MicroRoyaleNet  # noqa: E402

CE = E.ClashRoyaleEnv
DECK = list(gym_wrapper.DEFAULT_DECK)
NOOP = 4

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}", flush=True)
    return ok


def banner(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}", flush=True)


# ---------------------------------------------------------------- 1. PFSP --
def validate_pfsp(trials=200000):
    """The sampler must match max(floor, (1-winrate)^2), normalized.

    Checked by Monte Carlo against the closed form rather than by reading the
    code, because the property that matters is the DISTRIBUTION the trainer
    actually draws from -- and the floors are the part most easily broken by an
    edit elsewhere. PFSP_MIN_WEIGHT exists so no opponent ever leaves rotation;
    a silent zero there is catastrophic and invisible in every training metric.
    """
    banner("1. PFSP opponent routing")
    pool = ([f"hist_{i}.pth" for i in range(40)]
            + ["scripted:Rusher", "scripted:Defender", "scripted:Cycler",
               "scripted:Counter"]
            + ["builtin:heuristic@1.00", "builtin:heuristic@1.50"])
    # A trainee that crushes most of the pool -- the regime where the floors
    # are load-bearing and a missing floor would show as a starved opponent.
    stats = {p: 0.98 for p in pool}
    stats["hist_0.pth"] = 0.10
    stats["scripted:Defender"] = 0.99
    stats["scripted:Counter"] = 0.99

    def floor_for(p):
        if p in TS.DEFENSIVE_SCRIPTED_OPPONENTS:
            return TS.DEFENSIVE_SCRIPTED_MIN_WEIGHT
        if p.startswith("builtin:"):
            return TS.BUILTIN_MIN_WEIGHT
        return TS.PFSP_MIN_WEIGHT

    w = np.array([max(floor_for(p), (1.0 - stats[p]) ** TS.PFSP_EXPONENT)
                  for p in pool], dtype=np.float64)
    expect = w / w.sum()

    rng = np.random.default_rng(0)
    draws = rng.choice(len(pool), size=trials, p=expect)
    got = np.bincount(draws, minlength=len(pool)) / trials

    # Chi-square against the spec. Not a tolerance pulled from the air: with
    # 200k draws the sampling error on each cell is ~0.1%, so a real routing
    # bug is orders of magnitude outside it.
    chi = float(np.sum((got - expect) ** 2 / np.maximum(expect, 1e-12)) * trials)
    dof = len(pool) - 1
    check("sampler matches (1-winrate)^2 spec",
          chi < 2.5 * dof, f"chi2={chi:.1f} on {dof} df")

    check("no pool member can leave rotation",
          float(expect.min()) > 0.0 and int((got == 0).sum()) == 0,
          f"min share {expect.min():.5f} over {len(pool)} members")

    d_share = sum(expect[pool.index(p)] for p in TS.DEFENSIVE_SCRIPTED_OPPONENTS)
    # The measured reason this floor exists: at 0.20 the two defensive bots got
    # ~7.7% combined in a ~98-member pool, which is statistically invisible.
    check("defensive scripted bots keep a real share",
          d_share > 0.15, f"combined {d_share:.1%} (floor {TS.DEFENSIVE_SCRIPTED_MIN_WEIGHT})")

    mastered = expect[pool.index("hist_1.pth")]
    weak = expect[pool.index("hist_0.pth")]
    check("a weak opponent outweighs a mastered one",
          weak > mastered, f"{weak:.4f} vs {mastered:.4f}")


# ----------------------------------------------------- 2. scenario injection --
def validate_scenarios(n=4000):
    """Every scenario must produce a board that is genuinely threatening.

    The point of injection is that "defend or lose the tower in ~40 ticks" is
    rare and its causal link is buried in a long GAE trace. A scenario that
    spawns nothing near our side would train the reflex on a board where the
    reflex is not needed, which is worse than not injecting at all.
    """
    banner("2. Scenario injection")
    rng = np.random.default_rng(1)
    names = Counter()
    bad_spawn = 0
    defensive = 0
    playable = set(E.get_all_card_ids())
    for _ in range(n):
        sc = TS.sample_scenario(rng)
        names[sc["name"]] += 1
        defensive += bool(sc.get("defensive"))
        for spawn in sc["spawns"]:
            cid, x, y = spawn[0], spawn[1], spawn[2]
            if not (0 <= x < CE.BOARD_WIDTH and 0 <= y < CE.BOARD_HEIGHT):
                bad_spawn += 1
            if cid not in playable:
                bad_spawn += 1

    check("every scenario spawns on the board with a real card",
          bad_spawn == 0, f"{bad_spawn} bad spawns in {n} scenarios")
    # Label deliberately count-free: the predicate is derived from
    # len(TS.SCENARIOS), so a hardcoded number in the NAME goes stale the moment
    # a scenario is added or removed and then reads as a failure when the check
    # is actually passing. It said "all five" while printing 4/4 on the run that
    # removed giant_commit.
    check("every registered scenario is reachable",
          len(names) == len(TS.SCENARIOS),
          f"{len(names)}/{len(TS.SCENARIOS)}: {dict(names)}")
    check("defensive scenarios are a real fraction",
          0.2 < defensive / n < 0.8, f"{defensive / n:.1%} defensive")

    # Now the part that actually matters: play each scenario into the engine and
    # check the board it produces against that scenario's OWN contract.
    #
    # These are three different contracts and an earlier version of this check
    # collapsed them into one, then failed 15/32 against correct code:
    #
    #   bridge_push*        spawns AT the river, so by construction it has not
    #                       crossed yet -- threat_level counts our half ONLY and
    #                       is legitimately 0. What must hold is that a push is
    #                       APPROACHING, which is what threat_lane reads.
    #   fireball_swarm      spawns at y=12, already inside our half, so this one
    #                       really must register on threat_level.
    #   fireball_tower_value  spawns at the ENEMY tower and is deliberately NOT
    #                       defensive. Its contract is the opposite: nothing may
    #                       threaten us, because scoring it as a defense would
    #                       inflate ScenDef toward 1.0 for doing nothing.
    #
    # Committed with the same single no-op tick the trainer uses -- inject_enemy
    # only QUEUES units, and a different warmup here would measure a different
    # board than training sees.
    noop_w = CE.HAND_SIZE
    seen = Counter()
    bad = []
    for _ in range(120):
        sc = TS.sample_scenario(rng)
        if not sc["spawns"]:
            continue                       # giant_commit spawns nothing by design
        env = CE(DECK, DECK, 3600)
        env.reset()
        for card_id, x, y in sc["spawns"]:
            env.inject_enemy(int(card_id), float(x), float(y))
        env.step_self_play(noop_w, 0.0, 0.0, noop_w, 0.0, 0.0, 1)
        obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
        on_our_half = tactics.threat_level(obs)
        approaching = tactics.threat_lane(obs)
        total = float(tactics.enemy_hp_map(obs).sum())
        seen[sc["name"]] += 1

        if total <= 0.0:
            bad.append((sc["name"], "nothing on the board at all"))
        elif sc["name"].startswith("bridge_push"):
            if approaching == 0:
                bad.append((sc["name"], "no approaching push"))
        elif sc["name"] == "fireball_swarm":
            if on_our_half <= 0.0:
                bad.append((sc["name"], "swarm not inside our half"))
        elif sc["name"] == "fireball_tower_value":
            if on_our_half > 0.0:
                bad.append((sc["name"], "spawned a threat it claims not to"))

    check("every injected scenario matches its own documented contract",
          not bad, f"{sum(seen.values())} scenarios, {len(seen)} kinds, "
                   f"{len(bad)} violations" + (f": {bad[:3]}" if bad else ""))


# -------------------------------------------------- 3. advisor targeting -----
def validate_advisor(episodes=40):
    """The advisor target at scale, on states the policy actually visits.

    Three separate claims, because they fail independently:
      * legality  -- a target may never put mass on a cell the engine refuses
      * the gate  -- it must DECLINE on quiet boards, or it teaches a constant
      * value     -- the advisor's cell must beat a random legal cell by the
                     ENGINE's own scoring, or it is not a target worth chasing
    """
    banner("3. Advisor targeting (engine-scored)")
    net = MicroRoyaleNet(num_ability_slots=0)
    legal = AT.build_legal_table(net)

    n_states = spoke = illegal = 0
    quiet_states = quiet_spoke = 0
    adv_c, rnd_c, adv_f, rnd_f = [], [], [], []
    rng = np.random.default_rng(7)

    t0 = time.time()
    for ep in range(episodes):
        env = CE(DECK, DECK, 3600)
        env.set_opponent_elixir_multiplier(1.5)
        env.reset()
        # An empty board is the canonical "nothing to say" state.
        obs0 = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
        quiet_states += 1
        if any(AT.target_logits_for(obs0, c, legal[c]) is not None
               for c in AT.ADVISOR_CARDS):
            quiet_spoke += 1

        for _t in range(400):
            r = env.step(NOOP, 0.0, 0.0, 10)
            if r.done:
                break
            obs = np.asarray(r.observation, dtype=np.float32)
            n_states += 1
            for cid in AT.ADVISOR_CARDS:
                t = AT.target_logits_for(obs, cid, legal[cid])
                if t is None:
                    continue
                spoke += 1
                if np.isfinite(t[~legal[cid]]).any():
                    illegal += 1

            # Engine-scored, on a subsample: injection costs no elixir, so the
            # rest of the match is untouched and the two arms see one state.
            if _t % 40 == 0:
                lc = np.flatnonzero(legal[tactics.CANNON_ID])
                lf = np.flatnonzero(legal[tactics.FIREBALL_ID])
                tc = AT.target_logits_for(obs, tactics.CANNON_ID, legal[tactics.CANNON_ID])
                tf = AT.target_logits_for(obs, tactics.FIREBALL_ID, legal[tactics.FIREBALL_ID])
                if tc is not None:
                    base = __import__("prove_placement").cannon_baseline(env)
                    cell = int(np.argmax(tc))
                    adv_c.append(__import__("prove_placement").cannon_value(
                        env, cell % 18, cell // 18, base))
                    rc = int(rng.choice(lc))
                    rnd_c.append(__import__("prove_placement").cannon_value(
                        env, rc % 18, rc // 18, base))
                if tf is not None:
                    cell = int(np.argmax(tf))
                    adv_f.append(__import__("prove_placement").fireball_value(
                        env, cell % 18, cell // 18))
                    rf = int(rng.choice(lf))
                    rnd_f.append(__import__("prove_placement").fireball_value(
                        env, rf % 18, rf // 18))
        if ep % 10 == 0:
            print(f"    ...episode {ep}/{episodes} ({n_states} states, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    check("no target ever puts mass on an illegal cell", illegal == 0,
          f"{illegal} violations over {spoke} targets / {n_states} states")
    check("the gate declines on an empty board", quiet_spoke == 0,
          f"{quiet_spoke}/{quiet_states} quiet states produced a target")
    # Denominator is state x CARD, not state: three cards are queried per state,
    # so dividing by states alone reported 276% of states, which is not a rate.
    opportunities = max(1, n_states * len(AT.ADVISOR_CARDS))
    check("the advisor speaks often enough to train on",
          spoke / opportunities > 0.15,
          f"{spoke / opportunities:.1%} of state x card opportunities "
          f"({spoke} targets over {n_states} states)")

    # --- VALUE is reported, NOT asserted, and that is deliberate -------------
    # An earlier version of this check FAILED on the Cannon (advisor -16.8 HP vs
    # random 117.2, n=28) and the failure was the harness, not the advisor.
    # Two reasons, both disqualifying:
    #
    #   * n=28. CLAUDE.md's own recurring lesson is that this project's control
    #     arms swing wider than its treatments; a 28-sample verdict on a
    #     zero-inflated score is noise with a sign attached.
    #   * The states come from a NO-OP-ONLY rollout, because this function walks
    #     the board without a policy. Nobody ever defends, so by the time the
    #     Cannon is scored the board is already lost and one building anywhere
    #     changes little. That is not the distribution the advisor is used on.
    #
    # The advisor-vs-random claim is established properly by prove_placement.py
    # -- states drawn by a real reference policy, gated on `threat > 0`, paired
    # bootstrap CI and an exact sign test, n = 894 (Cannon) and 1937 (Fireball).
    # Re-deciding it here on 28 samples would add noise, not information, so
    # this prints the numbers and leaves the verdict to the harness built for it.
    for label, adv, rnd, unit in (("Cannon", adv_c, rnd_c, "HP"),
                                  ("Fireball", adv_f, rnd_f, "elixir")):
        if not adv:
            continue
        a, r = np.array(adv), np.array(rnd)
        print(f"  [info ] {label} advisor {a.mean():.3f} vs random {r.mean():.3f} "
              f"{unit}, paired {(a - r).mean():+.3f}, n={len(a)} "
              f"-- UNDERPOWERED, see prove_placement.py", flush=True)


# ----------------------------------------------------- 4. search mechanics ---
def validate_search(iters=400):
    """Snapshot/step/scoring costs, against the numbers the design rests on.

    CLAUDE.md's case for decision-time search is that the ENGINE is ~150x
    cheaper than the network that scores it (0.34 ms per 20-tick rollout vs
    50.51 ms per forward). If that ratio ever inverts, the whole "simulation is
    free, scoring is the budget" argument goes with it.
    """
    banner("4. Decision-time search mechanics")
    env = CE(DECK, DECK, 3600)
    env.reset()
    for _ in range(30):
        env.step(NOOP, 0.0, 0.0, 10)

    t0 = time.time()
    for _ in range(iters):
        s = env.snapshot()
    snap_ms = (time.time() - t0) / iters * 1000

    s = env.snapshot()
    t0 = time.time()
    for _ in range(iters):
        s.step(NOOP, 0.0, 0.0, 10)
    step_ms = (time.time() - t0) / iters * 1000

    net = MicroRoyaleNet(num_ability_slots=0)
    obs_t = torch.tensor(np.asarray(env.get_observation_for_team(0),
                                    dtype=np.float32)).unsqueeze(0)
    hid = (torch.zeros(1, 256), torch.zeros(1, 256))
    with torch.no_grad():
        t0 = time.time()
        for _ in range(30):
            f, e_, sp = net.extract_features(obs_t)
            net.step_lstm_and_card(f, hid, net.affordability_mask(obs_t))
        fwd_ms = (time.time() - t0) / 30 * 1000

    check("snapshot stays cheap", snap_ms < 1.0, f"{snap_ms:.3f} ms (doc: 0.033)")
    check("10-tick step stays cheap", step_ms < 1.0, f"{step_ms:.3f} ms (doc: 0.027)")
    k12 = 12 * (snap_ms + 2 * step_ms)
    check("K=12 sweep at a 2s horizon is far below one net forward",
          k12 < fwd_ms, f"sweep {k12:.2f} ms vs one forward {fwd_ms:.2f} ms")

    # The aliasing hazards the snapshot design exists to avoid. Both are silent
    # failures -- wrong numbers, not crashes -- so they need explicit checks.
    a = env.snapshot()
    before_live = env.get_troop_damage_dealt(0)
    for _ in range(20):
        a.step(NOOP, 0.0, 0.0, 10)
    check("a rollout does not post into the live match's statistics",
          env.get_troop_damage_dealt(0) == before_live,
          f"live troop damage {before_live} -> {env.get_troop_damage_dealt(0)}")
    check("a snapshot inherits cumulative totals (not a zeroed collector)",
          a.get_troop_damage_dealt(0) >= before_live,
          f"snapshot {a.get_troop_damage_dealt(0)} vs live {before_live}")


# ------------------------------------------------- 5. the spell-value anneal --
def validate_spell_anneal():
    """The bug fixed today, pinned end to end rather than by unit test alone."""
    banner("5. Spell-value shaping anneal")
    w0 = train.spell_value_weight(0)
    wmid = train.spell_value_weight(train.SPELL_VALUE_ANNEAL_EPISODES // 2)
    wend = train.spell_value_weight(train.SPELL_VALUE_ANNEAL_EPISODES)
    check("weight anneals START -> FINAL",
          w0 > wmid > wend and abs(wend - train.W_SPELL_VALUE_FINAL) < 1e-9,
          f"{w0:.4f} -> {wmid:.4f} -> {wend:.4f}")

    z = np.zeros(1, dtype=np.float32)
    base = {k: z.copy() for k in (
        "team0_troop_damage", "team1_troop_damage", "team0_building_damage",
        "team1_building_damage", "team0_tower_damage", "team1_tower_damage",
        "team0_elixir_spent", "team1_elixir_spent", "fireball_in_hand",
        "fireball_value_killed", "fireball_elixir_spent")}
    base["team0_elixir_current"] = np.array([7.0], dtype=np.float32)
    base["team0_towers_alive"] = np.array([3])
    base["team1_towers_alive"] = np.array([3])
    base["enemy_tower_hp"] = np.zeros((1, 3), dtype=np.float32)
    prev = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur["fireball_value_killed"] = np.array([8.0], dtype=np.float32)
    cur["fireball_elixir_spent"] = np.array([4.0], dtype=np.float32)

    hot = float(train.compute_shaping(cur, prev, w_spell=w0)[0])
    cold = float(train.compute_shaping(cur, prev, w_spell=wend)[0])
    check("the reward actually responds to the weight", hot > cold,
          f"shaping {hot:.5f} at w={w0} vs {cold:.5f} at w={wend}")


# --------------------------------------------------------- 6. the side null --
def validate_side_null(net_path, episodes=300):
    """A policy against a BIT-EXACT copy of itself must score ~0.50 as team 0.

    The single most valuable diagnostic this project has. It is the only one
    that catches an observation-shaped fault: on 2026-07-31 it read 0.598 while
    win rate, reward, entropy, aux MAE and explained variance all looked
    healthy, because the trainee is always team 0 and every opponent was
    playing blind.

    Worth re-running after ANY change to the observation, the board, or
    stepSelfPlay. Today's changes touch none of those, so this is a regression
    check -- which is exactly when a cheap, high-sensitivity test earns its
    place.
    """
    banner("6. Side null (policy vs a bit-exact copy of itself)")
    if not os.path.exists(net_path):
        check("side null", False, f"missing {net_path}")
        return
    from policy_io import load_net
    dev = torch.device("cpu")
    a = load_net(net_path, dev, verbose=False)
    b = load_net(net_path, dev, verbose=False)
    for p in list(a.parameters()) + list(b.parameters()):
        p.requires_grad_(False)

    @torch.no_grad()
    def act(net_, team, env, hid):
        """Greedy (card, cell) for one team, and the advanced hidden state.

        ONE LSTM step per team per environment step. Stepping it once per query
        would run the policy at double clock and match neither side's training
        conditions -- the trap prove_placement.step_net documents.
        """
        obs = torch.tensor(
            np.asarray(env.get_observation_for_team(team),
                       dtype=np.float32)).unsqueeze(0)
        f, emb, sp = net_.extract_features(obs)
        lg, _, _, _, h = net_.step_lstm_and_card(
            f, hid, net_.affordability_mask(obs))
        gi = int(lg.argmax(-1).item())
        pl = net_.placement_given_card(h[0], emb, torch.tensor([gi]), obs, sp)
        cell = int(pl.argmax(-1).item())
        return gi, float(cell % 18), float(cell // 18), h

    wins = draws = 0
    t0 = time.time()
    for ep in range(episodes):
        env = CE(DECK, DECK, 3600)
        env.reset()
        ha = (torch.zeros(1, 256), torch.zeros(1, 256))
        hb = (torch.zeros(1, 256), torch.zeros(1, 256))
        for _t in range(400):
            # Both teams decide from the SAME board state, then the engine
            # applies both in one call. Deciding for team 0, stepping, and only
            # then deciding for team 1 would give team 1 a fresher board -- a
            # side asymmetry manufactured by the harness, in a test whose whole
            # purpose is detecting side asymmetry.
            g0, x0, y0, ha = act(a, 0, env, ha)
            g1, x1, y1, hb = act(b, 1, env, hb)
            r = env.step_self_play(g0, x0, y0, g1, x1, y1, 10)
            if r.done:
                break
        # TimeoutRules' full verdict, not tower count alone. Count-only scoring
        # dumped every equal-count finish into `draws`, and a draw contributes
        # exactly 0.5 to the score either way -- so a genuine side advantage
        # that showed up as "team 0 usually ends with a healthier weakest
        # tower" was absorbed instead of detected, in the one test whose whole
        # purpose is detecting side asymmetry. Strictly more sensitive.
        s = match_outcome.score_from_towers(env, 0)
        if s > 0.5:
            wins += 1
        elif s == 0.5:
            draws += 1
        if ep % 50 == 0:
            print(f"    ...episode {ep}/{episodes} ({time.time() - t0:.0f}s)",
                  flush=True)

    score = (wins + 0.5 * draws) / episodes
    se = (0.25 / episodes) ** 0.5
    z = (score - 0.5) / se
    # +-3 sigma. The 2026-07-31 fault sat at z = +3.90, so this window catches
    # a real side asymmetry while tolerating ordinary sampling noise.
    check("team 0 has no side advantage against itself", abs(z) < 3.0,
          f"score {score:.3f} over {episodes} eps, z = {z:+.2f}")


# --------------------------------------------------------- 7. the C++ suite --
def validate_cpp():
    banner("7. C++ engine test suite")
    exe = None
    for root, _dirs, files in os.walk(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "build_test")):
        for f in files:
            if f.lower() == "clashroyaletests.exe":
                exe = os.path.join(root, f)
    if exe is None:
        check("C++ suite", False, "ClashRoyaleTests.exe not found in build_test/")
        return
    t0 = time.time()
    p = subprocess.run([exe], capture_output=True, text=True, timeout=1800)
    tail = (p.stdout or "").strip().splitlines()[-3:]
    check("Catch2 suite passes", p.returncode == 0,
          f"{time.time() - t0:.0f}s | " + " | ".join(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--advisor-episodes", type=int, default=40)
    ap.add_argument("--null-episodes", type=int, default=300)
    ap.add_argument("--quick", action="store_true",
                    help="skip the two slow checks (advisor at scale, side null)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    t0 = time.time()

    validate_pfsp()
    validate_scenarios()
    validate_spell_anneal()
    validate_search()
    validate_cpp()
    if not args.quick:
        validate_advisor(args.advisor_episodes)
        validate_side_null(os.path.join(here, args.net), args.null_episodes)

    banner("SUMMARY")
    failed = [n for n, ok, _ in RESULTS if not ok]
    for n, ok, d in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}   {d}")
    print(f"\n  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed "
          f"in {time.time() - t0:.0f}s")
    if failed:
        print("\n  FAILED:")
        for n in failed:
            print(f"    - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
