"""Keep a long unattended run alive.

Runs die quietly: an OOM kill leaves no traceback and a wedged worker just
stops the log. Two liveness tests, since they fail differently:

  * the process is gone                                   -> crashed or killed
  * alive, but the episode counter has not moved for `--stall-minutes` -> wedged

Restarting is safe: the trainer checkpoints every CLASH_SAVE_EVERY episodes,
PFSP deck estimates included.

Phase 1 ends by launching phase 2 and exiting, so the watchdog recognises both
phases in every launch form, relaunches the phase whose checkpoint is furthest
along, and watches that phase's log.

More than `--max-restarts` inside an hour stops the watchdog: something is
repeatably broken.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402


def classify(cmdline):
    """"phase1", "phase2" or None for one process command line.

    Separator-agnostic: a module launch spells it `trainers.train`, a path
    launch `trainers\\train.py` or `trainers/train.py`.
    """
    c = (cmdline or "").replace("\\", "/")
    if "trainers.train_selfplay" in c or "trainers/train_selfplay" in c:
        return "phase2"
    if ("trainers/train.py" in c or c.rstrip().endswith("trainers.train")
            or "trainers.train " in c):
        return "phase1"
    return None


def trainer_pids():
    """{pid: phase} for live trainer processes of either phase."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
             "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return {}
    found = {}
    for line in out.stdout.splitlines():
        pid, _, cmd = line.partition("|")
        phase = classify(cmd)
        if phase and pid.strip().isdigit():
            found[int(pid)] = phase
    return found


def phase_to_resume():
    """The phase whose checkpoint is furthest along: phase 2 once one exists."""
    from python_ai.trainers.train_selfplay import selfplay_paths
    weights, _bootstrap, _logdir = selfplay_paths()
    return "phase2" if os.path.exists(weights) else "phase1"


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


def launch(log_path, env_extra, phase="phase1"):
    env = dict(os.environ)
    env.update(env_extra)
    fh = open(log_path, "a", buffering=1, errors="replace")
    fh.write(f"\n=== watchdog relaunch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    module = ("python_ai.trainers.train_selfplay" if phase == "phase2"
              else "python_ai.trainers.train")
    return subprocess.Popen(
        [sys.executable, "-u", "-m", module],
        stdout=fh, stderr=subprocess.STDOUT, cwd=python_ai.REPO_ROOT, env=env)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True,
                    help="phase 1's console log; phase 2's is found from --logdir")
    # No default checkpoint: a watchdog started with defaults must not resume
    # an old run.
    ap.add_argument("--weights", default=None,
                    help="CLASH_WEIGHTS for the trainer; unset = the default path")
    ap.add_argument("--logdir", default=None,
                    help="CLASH_LOGDIR for the trainer; unset = the default")
    ap.add_argument("--save-every", default="250")
    ap.add_argument("--num-envs", default=None)
    ap.add_argument("--check-every", type=float, default=120.0)
    ap.add_argument("--stall-minutes", type=float, default=25.0,
                    help="no episode progress for this long counts as wedged; "
                         "generous, since PPO updates and checkpoint writes "
                         "pause the loop")
    ap.add_argument("--max-restarts", type=int, default=4)
    return ap


def main():
    args = build_parser().parse_args()

    env_extra = {"CLASH_SAVE_EVERY": args.save_every}
    # Set here too, so phase_to_resume() resolves the same paths as the
    # trainer.
    if args.weights:
        env_extra["CLASH_WEIGHTS"] = os.environ["CLASH_WEIGHTS"] = args.weights
    if args.logdir:
        env_extra["CLASH_LOGDIR"] = os.environ["CLASH_LOGDIR"] = args.logdir
    if args.num_envs:
        env_extra["CLASH_NUM_ENVS"] = args.num_envs

    restarts = []
    last_ep, last_move = last_episode(args.log), time.time()
    print(f"watchdog up: episode {last_ep}, checking every "
          f"{args.check_every:.0f}s, stall bound {args.stall_minutes:.0f}m",
          flush=True)

    from python_ai.rl.checkpointing import run_path
    # Where launch_pipeline2 writes phase 2's console log.
    phase2_log = os.path.join(
        os.path.dirname(os.path.abspath(
            run_path(args.logdir or "runs/clash_royale_experiment"))),
        "training_selfplay_pfsp.log")

    while True:
        time.sleep(args.check_every)
        pids = trainer_pids()
        phase = phase_to_resume()
        log = phase2_log if phase == "phase2" else args.log
        ep = last_episode(log)
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
        launch(log, env_extra, phase)
        restarts.append(now)
        last_move = now
        time.sleep(90)                         # let it boot before judging it


if __name__ == "__main__":
    sys.exit(main() or 0)
