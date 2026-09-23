"""Pre-flight: prove every component of the training loop before a long run uses
it.

    python_ai/venv/Scripts/python.exe python_ai/tools/validate_pipeline.py

Each check prints PASS/FAIL and the number it decided on. Every threshold is an
engine constant, a documented measurement, or a statistical null with its own
sample size; a check that cannot fail is not a check.

Runs against the C++ simulator only, never the emulator, and modifies nothing:
safe beside a live training directory.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

from python_ai.advisors import advisor_target as AT  # noqa: E402
import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.eval import match_outcome  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import scenarios  # noqa: E402
from python_ai.envs import scripted_opponents  # noqa: E402
from python_ai.envs import selfplay_env  # noqa: E402
from python_ai.rewards import shaping, weights  # noqa: E402
from python_ai.rl.config import PPOConfig  # noqa: E402
from python_ai.eval import prove_placement  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.engine_constants import BOARD_W  # noqa: E402


def cells_the_engine_refuses(env, card_id, target, team=0):
    """Cells carrying finite target mass that the engine will not accept.

    The engine is the independent oracle. `legal` comes from the net's own
    `_placement_legal` table; checking the target against that same table could
    never fail, because the target is built only on its cells.

    Returns a list of (cell, x, y); empty is clean.
    """
    flat = np.asarray(target, dtype=np.float64).reshape(-1)
    bad = []
    for cell in np.flatnonzero(np.isfinite(flat)):
        cell = int(cell)
        x, y = cell % BOARD_W, cell // BOARD_W
        if not env.is_valid_placement(int(card_id), float(x), float(y), team):
            bad.append((cell, x, y))
    return bad

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


# --- 1. PFSP ---
def validate_pfsp(trials=200000):
    """The sampler must match max(floor, (1-winrate)^2), normalised.

    Checked by Monte Carlo against the closed form, since what matters is the
    distribution actually drawn from, and a silent zero floor would drop an
    opponent from rotation invisibly.
    """
    banner("1. PFSP opponent routing")
    pool = ([f"hist_{i}.pth" for i in range(40)]
            + ["scripted:Rusher", "scripted:Defender", "scripted:Cycler",
               "scripted:Counter"]
            + ["builtin:heuristic@1.00", "builtin:heuristic@1.50"])
    # A trainee that crushes most of the pool: the regime where the floors
    # matter.
    stats = {p: 0.98 for p in pool}
    stats["hist_0.pth"] = 0.10
    stats["scripted:Defender"] = 0.99
    stats["scripted:Counter"] = 0.99

    def floor_for(p):
        if p in scripted_opponents.DEFENSIVE_SCRIPTED_OPPONENTS:
            return scripted_opponents.DEFENSIVE_SCRIPTED_MIN_WEIGHT
        if p.startswith("builtin:"):
            return selfplay_env.BUILTIN_MIN_WEIGHT
        return selfplay_env.PFSP_MIN_WEIGHT

    w = np.array([max(floor_for(p), (1.0 - stats[p]) ** selfplay_env.PFSP_EXPONENT)
                  for p in pool], dtype=np.float64)
    expect = w / w.sum()

    rng = np.random.default_rng(0)
    draws = rng.choice(len(pool), size=trials, p=expect)
    got = np.bincount(draws, minlength=len(pool)) / trials

    # Chi-square against the spec; with 200k draws a routing bug is orders of
    # magnitude outside sampling error.
    chi = float(np.sum((got - expect) ** 2 / np.maximum(expect, 1e-12)) * trials)
    dof = len(pool) - 1
    check("sampler matches (1-winrate)^2 spec",
          chi < 2.5 * dof, f"chi2={chi:.1f} on {dof} df")

    check("no pool member can leave rotation",
          float(expect.min()) > 0.0 and int((got == 0).sum()) == 0,
          f"min share {expect.min():.5f} over {len(pool)} members")

    d_share = sum(expect[pool.index(p)] for p in scripted_opponents.DEFENSIVE_SCRIPTED_OPPONENTS)
    # Without the floor, the two defensive bots are a statistically invisible
    # share of a large pool.
    check("defensive scripted bots keep a real share",
          d_share > 0.15, f"combined {d_share:.1%} (floor {scripted_opponents.DEFENSIVE_SCRIPTED_MIN_WEIGHT})")

    mastered = expect[pool.index("hist_1.pth")]
    weak = expect[pool.index("hist_0.pth")]
    check("a weak opponent outweighs a mastered one",
          weak > mastered, f"{weak:.4f} vs {mastered:.4f}")


# --- 2. scenario injection ---
def validate_scenarios(n=4000):
    """Every scenario must produce a genuinely threatening board; one that spawns
    nothing near our side trains the defence reflex where it is not needed.
    """
    banner("2. Scenario injection")
    rng = np.random.default_rng(1)
    names = Counter()
    bad_spawn = 0
    defensive = 0
    playable = set(E.get_all_card_ids())
    for _ in range(n):
        sc = scenarios.sample_scenario(rng)
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
    # Count-free label: the predicate derives from len(scenarios.SCENARIOS).
    check("every registered scenario is reachable",
          len(names) == len(scenarios.SCENARIOS),
          f"{len(names)}/{len(scenarios.SCENARIOS)}: {dict(names)}")
    check("defensive scenarios are a real fraction",
          0.2 < defensive / n < 0.8, f"{defensive / n:.1%} defensive")

    # Play each scenario into the engine and check the board against its own
    # contract:
    #
    #   bridge_push*          spawns at the river, not yet across, so threat_level (our half only) is legitimately 0; a push must be approaching (threat_lane)
    #   fireball_swarm        spawns inside our half, so it must register on threat_level
    #   fireball_tower_value  spawns at the enemy tower and is not defensive: nothing may threaten us, or ScenDef inflates for doing nothing
    #
    # Committed with the same single no-op tick the trainer uses, since
    # inject_enemy only queues units.
    noop_w = CE.HAND_SIZE
    seen = Counter()
    bad = []
    for _ in range(120):
        sc = scenarios.sample_scenario(rng)
        if not sc["spawns"]:
            continue                       # a scenario may spawn nothing by design
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


# --- 3. advisor targeting ---
def validate_advisor(episodes=40):
    """The advisor target at scale, on states the policy actually visits.

    Three claims that fail independently:
      * legality: no mass on a cell the engine refuses
      * the gate: it must decline on quiet boards, or it teaches a constant
      * value: reported, not asserted (see below)
    """
    banner("3. Advisor targeting (engine-scored)")
    net = MicroRoyaleNet(num_ability_slots=0)
    legal = AT.build_legal_table(net)

    n_states = spoke = illegal = checked = 0
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
                # Legality against the engine (see cells_the_engine_refuses),
                # subsampled: one pybind call per finite cell per card is too
                # slow for every state.
                if _t % 40 == 0:
                    checked += 1
                    if cells_the_engine_refuses(env, cid, t):
                        illegal += 1

            # Engine-scored on a subsample: injection is free, so both arms see
            # one state.
            if _t % 40 == 0:
                lc = np.flatnonzero(legal[tactics.CANNON_ID])
                lf = np.flatnonzero(legal[tactics.FIREBALL_ID])
                tc = AT.target_logits_for(obs, tactics.CANNON_ID, legal[tactics.CANNON_ID])
                tf = AT.target_logits_for(obs, tactics.FIREBALL_ID, legal[tactics.FIREBALL_ID])
                if tc is not None:
                    base = prove_placement.cannon_baseline(env)
                    cell = int(np.argmax(tc))
                    adv_c.append(prove_placement.cannon_value(
                        env, cell % BOARD_W, cell // BOARD_W, base))
                    rc = int(rng.choice(lc))
                    rnd_c.append(prove_placement.cannon_value(
                        env, rc % BOARD_W, rc // BOARD_W, base))
                if tf is not None:
                    cell = int(np.argmax(tf))
                    adv_f.append(prove_placement.fireball_value(
                        env, cell % BOARD_W, cell // BOARD_W))
                    rf = int(rng.choice(lf))
                    rnd_f.append(prove_placement.fireball_value(
                        env, rf % BOARD_W, rf // BOARD_W))
        if ep % 10 == 0:
            print(f"    ...episode {ep}/{episodes} ({n_states} states, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    check("no target puts mass on a cell the ENGINE refuses", illegal == 0,
          f"{illegal} violations over {checked} engine-checked targets "
          f"({spoke} targets / {n_states} states)")
    # A zero denominator would pass the line above vacuously.
    check("the engine legality probe actually ran", checked > 0,
          f"{checked} targets checked against the engine")
    check("the gate declines on an empty board", quiet_spoke == 0,
          f"{quiet_spoke}/{quiet_states} quiet states produced a target")
    # Denominator is state x card: three cards are queried per state.
    opportunities = max(1, n_states * len(AT.ADVISOR_CARDS))
    check("the advisor speaks often enough to train on",
          spoke / opportunities > 0.15,
          f"{spoke / opportunities:.1%} of state x card opportunities "
          f"({spoke} targets over {n_states} states)")

    # Value is reported, not asserted. These states come from a no-op-only
    # rollout where nobody defends, not the distribution the advisor is used
    # on, and the sample is small. prove_placement.py establishes the
    # advisor-vs-random claim properly.
    for label, adv, rnd, unit in (("Cannon", adv_c, rnd_c, "HP"),
                                  ("Fireball", adv_f, rnd_f, "elixir")):
        if not adv:
            continue
        a, r = np.array(adv), np.array(rnd)
        print(f"  [info ] {label} advisor {a.mean():.3f} vs random {r.mean():.3f} "
              f"{unit}, paired {(a - r).mean():+.3f}, n={len(a)} "
              f"-- UNDERPOWERED, see prove_placement.py", flush=True)


# --- 4. search mechanics ---
def validate_search(iters=400):
    """Snapshot/step/scoring costs, against what the search design rests on: the
    engine must stay far cheaper than the network that scores it.
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

    # The snapshot's aliasing hazards; both are silent wrong numbers, not
    # crashes.
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


# --- 5. the spell-value anneal ---
def validate_spell_anneal():
    """The spell-value anneal, pinned end to end."""
    banner("5. Spell-value shaping anneal")
    w0 = shaping.spell_value_weight(0)
    wmid = shaping.spell_value_weight(weights.SPELL_VALUE_ANNEAL_EPISODES // 2)
    wend = shaping.spell_value_weight(weights.SPELL_VALUE_ANNEAL_EPISODES)
    check("weight anneals START -> FINAL",
          w0 > wmid > wend and abs(wend - weights.W_SPELL_VALUE_FINAL) < 1e-9,
          f"{w0:.4f} -> {wmid:.4f} -> {wend:.4f}")

    z = np.zeros(1, dtype=np.float32)
    base = {k: z.copy() for k in (
        "team0_troop_damage", "team1_troop_damage", "team0_building_damage",
        "team1_building_damage", "team0_tower_damage", "team1_tower_damage",
        "team0_elixir_spent", "team1_elixir_spent", "spell_in_hand",
        "spell_value_killed", "spell_elixir_spent")}
    # A fixture 4-cost spell whatever the deck holds: this checks the anneal's
    # wiring, and a spell-less deck would zero the term.
    base["spell_damage"] = np.array([689.0], dtype=np.float32)
    base["spell_cost"] = np.array([4.0], dtype=np.float32)
    base["team0_elixir_current"] = np.array([7.0], dtype=np.float32)
    base["team0_towers_alive"] = np.array([3])
    base["team1_towers_alive"] = np.array([3])
    base["enemy_tower_hp"] = np.zeros((1, 3), dtype=np.float32)
    prev = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base.items()}
    cur["spell_value_killed"] = np.array([8.0], dtype=np.float32)
    cur["spell_elixir_spent"] = np.array([4.0], dtype=np.float32)

    hot = float(shaping.compute_shaping(cur, prev, PPOConfig.gamma, w_spell=w0)[0])
    cold = float(shaping.compute_shaping(cur, prev, PPOConfig.gamma, w_spell=wend)[0])
    check("the reward actually responds to the weight", hot > cold,
          f"shaping {hot:.5f} at w={w0} vs {cold:.5f} at w={wend}")


# --- 6. the side null ---
def validate_side_null(net_path, episodes=300):
    """A policy against a bit-exact copy of itself must score ~0.50 as team 0.

    The only diagnostic that catches an observation-shaped fault: the trainee
    is always team 0, so a blind opponent looks healthy in every training
    metric. Re-run after any change to the observation, the board or
    stepSelfPlay.
    """
    banner("6. Side null (policy vs a bit-exact copy of itself)")
    dev = torch.device("cpu")
    if os.path.exists(net_path):
        from python_ai.models.policy_io import load_net
        a = load_net(net_path, dev, verbose=False)
        b = load_net(net_path, dev, verbose=False)
    else:
        # Without a checkpoint, use a seeded random-init net: it has no
        # side-specific skill, so any deviation from 0.50 is structural.
        from python_ai.deck import DEFAULT_DECK
        from python_ai.envs.gym_wrapper import DEFAULT_DECK_ABILITY_SLOTS
        from python_ai.models.net import MicroRoyaleNet
        print(f"    no {net_path}; using a seeded random-init net (a sharper "
              f"subject for a structural asymmetry)", flush=True)
        torch.manual_seed(0)
        a = MicroRoyaleNet(num_ability_slots=DEFAULT_DECK_ABILITY_SLOTS).to(dev).eval()
        b = MicroRoyaleNet(num_ability_slots=DEFAULT_DECK_ABILITY_SLOTS).to(dev).eval()
        b.load_state_dict(a.state_dict())
    for p in list(a.parameters()) + list(b.parameters()):
        p.requires_grad_(False)

    @torch.no_grad()
    def act(net_, team, env, hid):
        """Greedy (card, cell) for one team, and the advanced hidden state.

        One LSTM step per team per environment step (see
        prove_placement.step_net).
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
        return gi, float(cell % BOARD_W), float(cell // BOARD_W), h

    wins = draws = 0
    t0 = time.time()
    for ep in range(episodes):
        env = CE(DECK, DECK, 3600)
        env.reset()
        ha = (torch.zeros(1, 256), torch.zeros(1, 256))
        hb = (torch.zeros(1, 256), torch.zeros(1, 256))
        for _t in range(400):
            # Both teams decide from the same board, then the engine applies
            # both; deciding sequentially would hand team 1 a fresher board.
            g0, x0, y0, ha = act(a, 0, env, ha)
            g1, x1, y1, hb = act(b, 1, env, hb)
            r = env.step_self_play(g0, x0, y0, g1, x1, y1, 10)
            if r.done:
                break
        # TimeoutRules' full verdict: count-only scoring turns equal-count
        # finishes into draws and hides a weakest-tower advantage.
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
    # +-3 sigma: catches a real side asymmetry while tolerating sampling noise.
    check("team 0 has no side advantage against itself", abs(z) < 3.0,
          f"score {score:.3f} over {episodes} eps, z = {z:+.2f}")


# --- 7. the C++ suite ---
# Where to look for ClashRoyaleTests.exe, in preference order. `build_python`
# is the live directory the .pyd and the suite are built from. A green run of a
# stale binary proves nothing, so the binary's identity is checked alongside
# its exit code.
CPP_BUILD_DIRS = ("build_python", "build", "build_test")

#: Source trees whose content the test binary must post-date.
CPP_SOURCE_DIRS = ("include", "src", "tests")


def _newest_source_mtime():
    newest = 0.0
    for d in CPP_SOURCE_DIRS:
        for root, _dirs, files in os.walk(os.path.join(python_ai.REPO_ROOT, d)):
            for f in files:
                if f.endswith((".h", ".hpp", ".cpp")):
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
    return newest


def _sources_changed_since(built, root=None):
    """Engine sources whose content changed after `built` (a POSIX mtime).

    Mtime alone gives false alarms (tooling can rewrite a file byte for byte),
    so a file counts only if it is newer than the build and either differs from
    HEAD, is untracked, or was last committed after the build. Falls back to
    mtime when git is unavailable.
    """
    root = root or python_ai.REPO_ROOT
    changed = []
    for d in CPP_SOURCE_DIRS:
        for dirpath, _dirs, files in os.walk(os.path.join(root, d)):
            for f in files:
                if not f.endswith((".h", ".hpp", ".cpp")):
                    continue
                full = os.path.join(dirpath, f)
                if os.path.getmtime(full) <= built:
                    continue
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                try:
                    dirty = subprocess.run(
                        ["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root,
                        capture_output=True).returncode != 0
                    untracked = subprocess.run(
                        ["git", "ls-files", "--error-unmatch", rel], cwd=root,
                        capture_output=True).returncode != 0
                    committed = subprocess.run(
                        ["git", "log", "-1", "--format=%ct", "--", rel], cwd=root,
                        capture_output=True, text=True).stdout.strip()
                except OSError:
                    changed.append(rel)
                    continue
                if dirty or untracked or (committed and float(committed) > built):
                    changed.append(rel)
    return sorted(changed)


def _find_cpp_suite():
    """Newest ClashRoyaleTests.exe in the most-preferred dir that holds one."""
    for d in CPP_BUILD_DIRS:
        found = []
        for root, _dirs, files in os.walk(os.path.join(python_ai.REPO_ROOT, d)):
            for f in files:
                if f.lower() == "clashroyaletests.exe":
                    found.append(os.path.join(root, f))
        if found:
            return max(found, key=os.path.getmtime), d
    return None, None


def validate_cpp():
    banner("7. C++ engine test suite")
    exe, where = _find_cpp_suite()
    if exe is None:
        check("C++ suite", False,
              "ClashRoyaleTests.exe not found in " + "/".join(CPP_BUILD_DIRS))
        return

    built = os.path.getmtime(exe)
    stamp = lambda t: time.strftime("%m-%d %H:%M", time.localtime(t))  # noqa: E731
    stale = _sources_changed_since(built)
    check("the test binary post-dates the engine source", not stale,
          f"{where}/ built {stamp(built)}; "
          + (f"changed since: {', '.join(stale[:5])}" if stale
             else "no source content changed since"))

    t0 = time.time()
    p = subprocess.run([exe], capture_output=True, text=True, timeout=1800)
    tail = (p.stdout or "").strip().splitlines()[-3:]
    check("Catch2 suite passes", p.returncode == 0,
          f"{where}/ | {time.time() - t0:.0f}s | " + " | ".join(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--advisor-episodes", type=int, default=40)
    ap.add_argument("--null-episodes", type=int, default=300)
    ap.add_argument("--quick", action="store_true",
                    help="skip the two slow checks (advisor at scale, side null)")
    args = ap.parse_args()

    here = python_ai.PACKAGE_DIR
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
