"""Keep a long unattended phase-1 run alive.

WHY. A 25-hour run that dies at hour 3 costs 22 hours of nothing, and the ways
it dies are quiet: an OOM kill leaves no traceback, and a wedged worker leaves
the log simply not advancing. Neither is visible to a log tail that greps for
"Traceback".

Two independent liveness tests, because they fail differently:

  * the PROCESS is gone            -> crashed or was killed
  * the process is alive but the
    EPISODE COUNTER has not moved  -> wedged; a deadlocked worker keeps the
    for `--stall-minutes`             parent alive indefinitely

Restarting is cheap and safe by construction: the trainer checkpoints every
CLASH_SAVE_EVERY episodes, and since 2026-09-04 the PFSP deck estimates ride in
the checkpoint too, so a resume no longer throws away what the run learned about
the pool. Before that fix an automatic restarter would have quietly degraded the
run every time it fired.

REFUSES TO RESTART-LOOP. If more than `--max-restarts` fire inside one hour the
watchdog stops and says so: something is repeatably broken and relaunching it
faster is not the answer.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402


def trainer_pids():
    """PIDs of live trainer processes, matched on the module they run."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
             "Where-Object { $_.CommandLine -like '*trainers.train*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


def last_episode(log_path):
    """Highest episode number the log has printed, or None."""
    try:
        with open(log_path, errors="replace") as fh:
            tail = fh.readlines()[-400:]
    except OSError:
        return None
    for line in reversed(tail):
        if line.startswith("Episodes:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def launch(log_path, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    fh = open(log_path, "a", buffering=1, errors="replace")
    fh.write(f"\n=== watchdog relaunch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "python_ai.trainers.train"],
        stdout=fh, stderr=subprocess.STDOUT, cwd=python_ai.REPO_ROOT, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--weights", default="model_weights_phase7.pth")
    ap.add_argument("--logdir", default="runs/phase7")
    ap.add_argument("--save-every", default="250")
    ap.add_argument("--num-envs", default=None)
    ap.add_argument("--check-every", type=float, default=120.0)
    ap.add_argument("--stall-minutes", type=float, default=25.0,
                    help="no episode progress for this long counts as wedged. "
                         "Generous on purpose: a PPO update is ~47s and a "
                         "checkpoint write pauses the loop, so a tight bound "
                         "would kill a healthy run mid-update.")
    ap.add_argument("--max-restarts", type=int, default=4)
    args = ap.parse_args()

    env_extra = {"CLASH_WEIGHTS": args.weights,
                 "CLASH_LOGDIR": args.logdir,
                 "CLASH_SAVE_EVERY": args.save_every}
    if args.num_envs:
        env_extra["CLASH_NUM_ENVS"] = args.num_envs

    restarts = []
    last_ep, last_move = last_episode(args.log), time.time()
    print(f"watchdog up: episode {last_ep}, checking every "
          f"{args.check_every:.0f}s, stall bound {args.stall_minutes:.0f}m",
          flush=True)

    while True:
        time.sleep(args.check_every)
        pids = trainer_pids()
        ep = last_episode(args.log)
        now = time.time()

        if ep is not None and ep != last_ep:
            last_ep, last_move = ep, now
            continue

        stalled = (now - last_move) > args.stall_minutes * 60
        if pids and not stalled:
            continue        # alive and recently moved, or mid-update

        why = "process gone" if not pids else (
            f"no episode progress for {(now - last_move)/60:.0f}m")
        restarts = [t for t in restarts if now - t < 3600]
        if len(restarts) >= args.max_restarts:
            print(f"!! {len(restarts)} restarts inside an hour -- refusing to "
                  f"loop. Something is repeatably broken; stopping the "
                  f"watchdog so it is visible.", flush=True)
            return 1

        print(f"!! {time.strftime('%H:%M:%S')} restarting trainer ({why}, "
              f"last episode {ep})", flush=True)
        for pid in pids:                       # wedged: clear it out first
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        time.sleep(5)
        launch(args.log, env_extra)
        restarts.append(now)
        last_move = now
        time.sleep(90)                         # let it boot before judging it


if __name__ == "__main__":
    sys.exit(main() or 0)
