"""Health daemon for an unattended training run.

    python_ai/venv/Scripts/python.exe python_ai/tools/monitor_run.py --dir _runs/main --once

Prints one line per check and exits non-zero on any ALARM. Reads the
TensorBoard event file and a copy of the checkpoint, never the trainer's
memory, so it cannot perturb the run.

  * NaN / inf in any loss          a dead run that keeps printing
  * win rate, floor alarm, ladder  a run that has stopped winning (every other check reads healthy on it)
  * ratio self-check               rollout and update disagreeing about the policy
  * parameter norm                 weight collapse or blow-up; also confirms checkpoints are being written
  * RSS of python processes        memory leaks on a long run
  * Shaping/SpellValueWeight       that the anneal actually moves
  * entropy vs its target          the controller fighting rather than tracking
  * Advisor/KL with Advisor/Rows   never KL alone: it also falls when the advisor stops speaking
  * per-card modal share + top-1 p the placement collapse detector; a good head is sharp but moves its mode with the board
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import torch

# Run as a script, the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401
from python_ai.engine_constants import BOARD_W  # noqa: E402

ALARM, WARN, OK = "ALARM", "warn", "ok"


def _scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator)
    cands = glob.glob(os.path.join(run_dir, "runs", "*"))
    if not cands:
        return {}
    ea = EventAccumulator(max(cands, key=os.path.getmtime))
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in ev]
    return out


def _last(series, n=1):
    if not series:
        return None
    vals = [v for _s, v in series[-n:]]
    return float(np.mean(vals))


def _trainer_rss_mb():
    """Total RSS of every python process, in MB. Windows: via tasklist."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    total = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip('" ') for p in line.split('","')]
        if len(parts) >= 5:
            kb = parts[4].replace(",", "").replace(" K", "").strip()
            try:
                total += float(kb) / 1024.0
            except ValueError:
                pass
    return total or None


def check_placement(ckpt_path, episodes, opp_elixir):
    """Per-card modal share + top-1 probability over greedy play.

    Behavioural, because "the head returns one cell regardless of the board" is
    invisible in any norm and in aggregate entropy.
    """
    import clash_royale_env as E
    from python_ai.envs import gym_wrapper
    from python_ai.advisors import tactics
    from python_ai.models.policy_io import load_net

    CE = E.ClashRoyaleEnv
    dev = torch.device("cpu")
    net = load_net(ckpt_path, dev, verbose=False)
    deck = list(gym_wrapper.DEFAULT_DECK)
    # Every card of the trainee's deck.
    watch = {int(c): E.get_card_info(int(c))["name"] for c in deck}
    cells = {c: [] for c in watch}
    top1 = {c: [] for c in watch}

    for _ep in range(episodes):
        env = CE(deck, deck, 3600)
        env.set_opponent_elixir_multiplier(opp_elixir)
        env.reset()
        hid = (torch.zeros(1, 256), torch.zeros(1, 256))
        obs = env.get_observation_for_team(0)
        for _t in range(400):
            ot = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                f, emb, sp = net.extract_features(ot)
                lg, _, _, _, hid = net.step_lstm_and_card(
                    f, hid, net.affordability_mask(ot))
                hand = net.hand_card_ids(ot)[0].tolist()
                for cid in watch:
                    if cid not in hand:
                        continue
                    slot = torch.tensor([hand.index(cid)])
                    pl = net.placement_given_card(hid[0], emb, slot, ot, sp)
                    p = torch.softmax(pl, -1)[0]
                    cells[cid].append(int(p.argmax()))
                    top1[cid].append(float(p.max()))
                gi = int(lg.argmax(-1).item())
                gp = net.placement_given_card(
                    hid[0], emb, torch.tensor([gi]), ot, sp)
                cell = int(gp.argmax(-1).item())
            r = env.step(gi, float(cell % BOARD_W), float(cell // BOARD_W), 10)
            obs = r.observation
            if r.done:
                break

    rows = []
    for cid, name in watch.items():
        if not cells[cid]:
            rows.append((name, None, None, 0))
            continue
        counts = np.bincount(cells[cid], minlength=612)
        rows.append((name, counts.max() / len(cells[cid]),
                     float(np.mean(top1[cid])), len(cells[cid])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the run's working directory")
    ap.add_argument("--ckpt", default="model_weights.pth")
    ap.add_argument("--placement-episodes", type=int, default=12)
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--rss-limit-mb", type=float, default=14000.0)
    ap.add_argument("--skip-placement", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.dir)
    findings = []

    def say(level, what, detail):
        findings.append((level, what, detail))
        print(f"  [{level:5s}] {what}: {detail}", flush=True)

    print(f"\n=== health @ {time.strftime('%H:%M:%S')} -- {run_dir}", flush=True)
    sc = _scalars(run_dir)
    if not sc:
        say(ALARM, "tensorboard", "no event file -- is the run writing at all?")

    # --- liveness + NaN ---
    for tag in ("Loss/Actor", "Loss/Critic", "Loss/Entropy"):
        v = _last(sc.get(tag, []), 5)
        if v is None:
            say(WARN, tag, "no data yet")
        elif not np.isfinite(v):
            say(ALARM, tag, f"non-finite ({v})")
        else:
            say(OK, tag, f"{v:.5f}")

    steps = [s for s, _v in sc.get("Loss/Actor", [])]
    if steps:
        say(OK, "episodes", f"latest update at episode {steps[-1]}")

    # --- is the agent winning anything ---
    # Every other check here reads healthy on a run that has lost thousands of
    # straight games, and rung 0 has no curriculum valve of its own.
    wr = sc.get("Training/Win_Rate_100", [])
    stage = _last(sc.get("Training/Curriculum_Stage", []), 1)
    if wr:
        last = _last(wr, 3)
        span = wr[-1][0] - wr[0][0]
        best = max(v for _s, v in wr)
        if last <= 0.05 and span >= 1000 and best <= 0.05:
            level = ALARM
        elif last <= 0.15:
            level = WARN
        else:
            level = OK
        say(level, "win rate",
            f"{last:.3f} now, best {best:.3f}, over {span} episodes of history"
            + (f", rung {int(stage)}" if stage is not None else ""))
    else:
        say(WARN, "win rate", "no Training/Win_Rate_100 yet")
    rew = sc.get("Training/Avg_Reward_50", [])
    if len(rew) >= 2:
        say(OK, "reward", f"{_last(rew[:3], 3):+.3f} -> {_last(rew, 3):+.3f}")
    if sc.get("Training/Curriculum_FloorAlarm"):
        n = len(sc["Training/Curriculum_FloorAlarm"])
        say(ALARM, "floor alarm",
            f"fired {n}x -- the run sat at rung 0 without winning; the teacher "
            f"cannot get easier, so this will not correct itself")
    plateau = _last(sc.get("Training/Curriculum_PlateauAdvances", []), 1)
    if stage is not None and plateau is not None:
        say(OK if plateau < max(1.0, stage) else WARN, "ladder",
            f"rung {int(stage)}, {int(plateau)} rung(s) left by PLATEAU rather "
            f"than by the gate")
    rdev = _last(sc.get("Loss/Ratio_Dev_First_Minibatch", []), 5)
    if rdev is not None:
        say(ALARM if rdev > 1e-3 else OK, "ratio self-check",
            f"max |ratio-1| at epoch 0 = {rdev:.2e} -- above 1e-3 the rollout and "
            f"the update disagree about what the policy did")
    modal = sc.get("Placement/ModalShare_Max", [])
    if modal:
        m = _last(modal, 3)
        worst = max(((tag.split("/")[-1], v[-1][1]) for tag, v in sc.items()
                     if tag.startswith("Placement/ModalShare/") and v),
                    key=lambda kv: kv[1], default=("?", m))
        say(WARN if m > 0.60 else OK, "placement modal share",
            f"max {m:.2f} ({worst[0]} {worst[1]:.2f}) -- above 0.60 a card is "
            f"landing on one cell whatever the board")
    dmin = _last(sc.get("Decks/WinRate_Min", []), 1)
    if dmin is not None:
        say(OK, "worst deck", f"win rate {dmin:.2f}")

    # --- the spell anneal ---
    w = sc.get("Shaping/SpellValueWeight", [])
    if len(w) >= 2:
        first, last = w[0][1], w[-1][1]
        moved = abs(first - last) > 1e-9
        say(OK if moved else WARN, "spell weight",
            f"{first:.5f} -> {last:.5f} over {len(w)} updates"
            + ("" if moved else "  (flat: check CLASH_SPELL_ANNEAL_* )"))
    elif w:
        say(OK, "spell weight", f"{w[-1][1]:.5f} (one sample)")

    # --- advisor: KL and Rows, always together ---
    kl, rows_ = sc.get("Advisor/KL", []), sc.get("Advisor/Rows", [])
    coef = _last(sc.get("Advisor/Coef", []), 1)
    if kl and rows_:
        k0, k1 = _last(kl[:5], 5), _last(kl, 5)
        r0, r1 = _last(rows_[:5], 5), _last(rows_, 5)
        if coef is not None and coef <= 0.0:
            # Zero rows is correct for a control arm with the term switched
            # off.
            say(OK, "advisor", f"disabled (coef 0) -- {r1:.1f} rows, as expected")
        elif r1 is not None and r1 < 1.0:
            say(ALARM, "advisor", f"target rows collapsed to {r1:.1f} at "
                                  f"coef {coef} -- the term is not training anything")
        else:
            say(OK, "advisor", f"KL {k0:.3f} -> {k1:.3f}   rows {r0:.1f} -> {r1:.1f}")

    # --- entropy controller ---
    meas = _last(sc.get("Entropy/Placement_Measured", []), 5)
    targ = _last(sc.get("Entropy/Placement_Target", []), 5)
    if meas is not None and targ is not None:
        gap = abs(meas - targ)
        say(OK if gap < 0.25 else WARN, "placement entropy",
            f"measured {meas:.3f} vs target {targ:.3f} (gap {gap:.3f})")
    cov = _last(sc.get("Entropy/Placement_Coverage", []), 5)
    if cov is not None:
        say(OK, "coverage entropy", f"{cov:.3f} of max (fallback rows)")

    # --- weights + memory ---
    ck = os.path.join(run_dir, args.ckpt)
    if os.path.exists(ck):
        age_min = (time.time() - os.path.getmtime(ck)) / 60.0
        # Load a copy: on Windows the trainer's checkpoint replace fails while
        # any handle holds the file.
        import shutil
        import tempfile
        tmp_ck = os.path.join(tempfile.gettempdir(), f"monitor_{os.getpid()}.pth")
        shutil.copy2(ck, tmp_ck)
        blob = torch.load(tmp_ck, map_location="cpu", weights_only=False)
        model = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        norms = {k: float(v.float().norm()) for k, v in model.items()
                 if hasattr(v, "float")}
        bad = [k for k, n in norms.items() if not np.isfinite(n)]
        total = float(np.sqrt(sum(n * n for n in norms.values())))
        if bad:
            say(ALARM, "weights", f"non-finite tensors: {bad[:4]}")
        elif total < 1e-3:
            say(ALARM, "weights", f"collapsed to zero (norm {total:.2e})")
        else:
            say(OK, "weights", f"global norm {total:.2f}, "
                               f"checkpoint {age_min:.0f} min old")
        if age_min > 90:
            say(WARN, "checkpoint", f"{age_min:.0f} min since last write")
    else:
        say(WARN, "checkpoint", f"{ck} not written yet")

    rss = _trainer_rss_mb()
    if rss is not None:
        say(ALARM if rss > args.rss_limit_mb else OK, "memory",
            f"{rss:.0f} MB across all python processes")

    # --- the placement collapse detector ---
    if not args.skip_placement and os.path.exists(ck):
        t0 = time.time()
        for name, modal, p1, n in check_placement(
                ck, args.placement_episodes, args.opp_elixir):
            if n == 0:
                say(WARN, f"placement {name}", "never in hand")
                continue
            # High modal share at a near-uniform top-1 (1/612) is a dissolved
            # head with a meaningless mode; at a high top-1 it is a frozen
            # cell, the collapse being watched for.
            level = ALARM if (modal > 0.60 and p1 > 0.20) else (
                WARN if modal > 0.60 or p1 < 0.005 else OK)
            say(level, f"placement {name}",
                f"modal share {modal:5.1%}  top-1 p {p1:.4f}  n={n}")
        say(OK, "placement probe", f"{time.time() - t0:.0f}s")

    alarms = [f for f in findings if f[0] == ALARM]
    print(f"\n  {len(alarms)} alarm(s), "
          f"{len([f for f in findings if f[0] == WARN])} warning(s), "
          f"{len(findings)} checks", flush=True)
    return 1 if alarms else 0


if __name__ == "__main__":
    sys.exit(main())
