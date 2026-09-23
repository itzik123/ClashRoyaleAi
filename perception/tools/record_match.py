"""Record a live match to lossless frames, for offline fitting.

The live loop is detector-bound (~1.5 fps), too coarse to correlate a placement
with the elixir bar's drop. Pure capture runs at the window's own rate, and
every analysis runs offline, where the detector can take as long as it likes
and be re-run without another match.

PNG, not video:

  * Lossless. The badge and HP readers key on exact colours (ally fill
    (111,208,252) against its track (63,79,112); team hue bands ~20 wide),
    which lossy codecs move.
  * Honest timestamps. Live capture is variable-rate (WGC delivers on window
    updates); a video at nominal fps would bake in a constant-rate fiction.
    Each frame's real capture time goes in the manifest.

Stops at `--seconds`, or when the manifest's `stop_file` is created, so a
background recording can be ended cleanly from another process.
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.window import WindowSource  # noqa: E402

# PNG compression on a 549x976 frame: level 0 costs 68 ms, 1 costs 177, 3 costs
# 247, 6 costs 552. At 10 fps the budget is 100 ms, so encoding runs on workers
# and level 1 is affordable.
PNG_COMPRESSION = 1

# cv2.imwrite releases the GIL, so these run in parallel: three at 177 ms
# sustain ~17 fps against a 10 fps target, leaving room for the emulator on the
# same machine.
WRITER_THREADS = 3

# Bounded, so a slow disk applies backpressure instead of growing a queue into
# an out-of-memory kill.
QUEUE_DEPTH = 48

# Refuse to start without room for the whole recording plus a margin: filling
# the system disk mid-match loses the match and endangers a training run on the
# same machine.
BYTES_PER_FRAME_ESTIMATE = 700_000
DISK_MARGIN_BYTES = 2 << 30      # 2 GiB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "assets"
                    / "live" / time.strftime("match_%Y%m%d_%H%M%S"))
    ap.add_argument("--fps", type=float, default=10.0,
                    help="target sampling rate (capture runs at ~13)")
    ap.add_argument("--seconds", type=float, default=420.0,
                    help="hard stop; a 3-minute match plus overtime and lobby")
    ap.add_argument("--window", default="BlueStacks App Player")
    args = ap.parse_args()

    expect = int(args.fps * args.seconds)
    need = expect * BYTES_PER_FRAME_ESTIMATE + DISK_MARGIN_BYTES
    free = shutil.disk_usage(args.out.anchor).free
    if free < need:
        print(f"REFUSING: need ~{need / 2**30:.1f} GiB free "
              f"(={expect} frames + margin), have {free / 2**30:.1f} GiB")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stop_file = args.out / "STOP"
    print(f"output    : {args.out}")
    print(f"target    : {args.fps} fps for up to {args.seconds:.0f}s "
          f"(~{expect} frames, ~{expect * BYTES_PER_FRAME_ESTIMATE / 2**30:.1f} GiB)")
    print(f"stop early: create {stop_file}")

    source = WindowSource(args.window)
    print(f"game frame: {source.size[0]} x {source.size[1]}   "
          f"capture {source.fps:.1f} fps")

    work: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    dropped = 0
    stalled = 0

    def writer():
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            path, image = item
            cv2.imwrite(str(path), image,
                        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            work.task_done()

    threads = [threading.Thread(target=writer, daemon=True)
               for _ in range(WRITER_THREADS)]
    for t in threads:
        t.start()

    interval = 1.0 / args.fps
    rows = []
    t0 = time.perf_counter()
    next_due = t0
    last_report = t0
    try:
        while True:
            now = time.perf_counter()
            if now - t0 >= args.seconds or stop_file.exists():
                break
            if now < next_due:
                time.sleep(min(0.005, next_due - now))
                continue
            # Anchored to the frame actually taken, so a stall cannot cause a
            # catch-up burst.
            next_due = now + interval

            # read_new, not read: a duplicated surface would enter the manifest
            # as two observations of one instant.
            frame = source.read_new(timeout_s=1.0)
            if frame is None:
                stalled += 1
                continue
            name = f"f{len(rows):05d}.png"
            try:
                work.put_nowait((args.out / name, frame.image))
            except queue.Full:
                # Recorded as a gap rather than blocked on: a stalled sampler
                # would skew every later timestamp.
                dropped += 1
                continue
            rows.append({"file": name, "index": frame.index,
                         "wall_time_ms": round(frame.wall_time_ms, 3),
                         "t_since_start_s": round(now - t0, 4)})

            if now - last_report >= 10.0:
                got = len(rows) / (now - t0)
                print(f"  {now - t0:6.1f}s  {len(rows):5d} frames  "
                      f"{got:.1f} fps  capture {source.fps:.1f}  "
                      f"queue {work.qsize()}  dropped {dropped}  stalled {stalled}")
                last_report = now
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        source.close()
        for _ in threads:
            work.put(None)
        work.join()

    elapsed = time.perf_counter() - t0
    manifest = {
        "frames": rows,
        "count": len(rows),
        "dropped": dropped,
        "stalled": stalled,
        "elapsed_s": round(elapsed, 3),
        "achieved_fps": round(len(rows) / elapsed, 3) if elapsed else 0.0,
        "target_fps": args.fps,
        "size": list(source.size) if rows else None,
        "window": args.window,
        "stop_file": str(stop_file),
        "note": "variable-rate by nature; use wall_time_ms, never index/fps",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                            encoding="utf-8")
    print(f"\n{len(rows)} frames in {elapsed:.1f}s "
          f"({manifest['achieved_fps']:.1f} fps) -> {args.out}")
    if rows and manifest["achieved_fps"] < args.fps * 0.8:
        print(f"WARNING: achieved {manifest['achieved_fps']:.1f} fps against a "
              f"{args.fps} target -- placement instants will be coarser than "
              "planned. Check what else is using the CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
