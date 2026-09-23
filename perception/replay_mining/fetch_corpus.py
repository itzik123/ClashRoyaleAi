"""Cache the fast_hog_2.6 corpus locally: 221 episodes, ~1.3 GB compressed. Not
committed: third-party data with no declared licence, and the extracted prior
(~254 KB) is the artifact this project needs.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/wty-yy/Clash-Royale-Replay-Dataset/master"
API = ("https://api.github.com/repos/wty-yy/Clash-Royale-Replay-Dataset"
       "/git/trees/master?recursive=1")
LABELS = ("https://raw.githubusercontent.com/wty-yy/KataCR/master"
          "/katacr/constants/label_list.py")


def main(out_dir: str, prefix: str = "fast_hog_2.6/") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab = out / "label_list.py"
    if not lab.exists():
        urllib.request.urlretrieve(LABELS, lab)
    tree = json.loads(urllib.request.urlopen(API).read())["tree"]
    names = sorted(x["path"] for x in tree
                   if x["type"] == "blob" and x["path"].startswith(prefix))
    print(f"{len(names)} episodes under {prefix}", flush=True)
    done = 0
    for i, name in enumerate(names, 1):
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(f"{RAW}/{name}", dest)
            done += 1
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED {name}: {exc}", flush=True)
        if i % 20 == 0:
            print(f"  {i}/{len(names)} ...", flush=True)
    total = sum(p.stat().st_size for p in out.rglob("*.xz"))
    print(f"cached {done}/{len(names)} episodes, {total/1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "./_replay_cache"))
