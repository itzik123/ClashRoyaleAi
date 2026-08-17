"""Does a reconstructed board's WRONG tower HP change which action search picks?

The live loop cannot write tower HP into the engine -- perception measures it
(hp_fraction 0.87 on our right princess, 0.81 on the enemy king in one sampled
frame) but there is no setter, so a reconstructed board always reports both
sides untouched and symmetric.

Whether that MATTERS is a separate question from whether it is wrong. Search
compares candidates on one board, so a bias shared by every candidate cancels;
only a bias that reorders them costs anything. This measures the reordering
directly, in the SIMULATOR where the true tower HP is known:

    ranking A   candidates scored on the real observation
    ranking B   the same candidates, tower scalars overwritten to FULL
    metric      how often argmax(A) != argmax(B)

Everything else is held identical -- same states, same candidate set, same net,
same hidden state. So a disagreement is attributable to the six tower scalars
and nothing else.

Deliberately measured on states where the towers are ACTUALLY damaged. Early in
a match the real board is already near-full and the corruption is nearly a
no-op, which would dilute the estimate toward zero and understate the problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PY_AI = _ROOT.parent / "python_ai"
if str(_PY_AI) not in sys.path:
    sys.path.append(str(_PY_AI))

import engine as _engine_build  # noqa: F401,E402 -- fresh build first

# The six tower-HP scalars sit inside the appended extra scalars: index 0 is
# elapsed time, 1-2 the two cumulative elixir spends, 3-8 the tower HPs.
TOWER_SCALAR_SLICE = slice(-6, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=_PY_AI / "model_weights_selfplay.pth")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--min-damage", type=float, default=0.02,
                        help="skip states whose towers are still ~full")
    args = parser.parse_args()

    import clash_royale_env as cre
    from model import MicroRoyaleNet

    import shutil
    import tempfile
    # The live phase-2 run rewrites this file; read a copy.
    frozen = Path(tempfile.gettempdir()) / "towerhp_probe.pth"
    shutil.copy2(args.checkpoint, frozen)
    blob = torch.load(frozen, map_location="cpu", weights_only=False)

    net = MicroRoyaleNet(num_ability_slots=0)
    net.load_state_dict(blob["model"] if "model" in blob else blob, strict=False)
    net.eval()

    # Not `from gym_wrapper import DEFAULT_DECK`: that pulls in gymnasium,
    # which perception's venv deliberately does not carry. Same single source
    # of truth, read the same way the live loop reads it.
    from live.mvp_loop import _training_deck_ids  # noqa: PLC0415
    DEFAULT_DECK = _training_deck_ids()
    # 3600 explicitly -- bindings.cpp defaults max_ticks to 1800, HALF the
    # length the policy trained at.
    env = cre.ClashRoyaleEnv(DEFAULT_DECK, DEFAULT_DECK, 3600)

    changed = total = skipped = 0
    value_deltas = []

    for ep in range(args.episodes):
        env.reset()
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        for _step in range(300):
            obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
            towers = obs[TOWER_SCALAR_SLICE].copy()

            # "Full" means what a FRESH env reports, not 1.0 -- the scalars are
            # normalised by max building HP, so a princess at full reads 0.6322.
            # Hardcoding 1.0 here would measure a board that cannot exist.
            if not hasattr(main, "_full"):
                probe = cre.ClashRoyaleEnv(DEFAULT_DECK, DEFAULT_DECK, 3600)
                probe.reset()
                main._full = np.asarray(
                    probe.get_observation_for_team(0),
                    dtype=np.float32)[TOWER_SCALAR_SLICE].copy()
            full = main._full

            damage = float(np.abs(towers - full).max())
            if damage < args.min_damage:
                skipped += 1
            else:
                total += 1
                corrupted = obs.copy()
                corrupted[TOWER_SCALAR_SLICE] = full

                with torch.no_grad():
                    batch = torch.from_numpy(np.stack([obs, corrupted]))
                    feats, _, _ = net.extract_features(batch)
                    # The SAME hidden state for both rows, so the only
                    # difference reaching the heads is the tower scalars.
                    h2 = (hx.expand(2, 256).contiguous(),
                          cx.expand(2, 256).contiguous())
                    out = net.step_lstm_and_card(feats, h2)
                    card_logits, values = out[0], out[3]
                    a = int(card_logits[0].argmax().item())
                    b = int(card_logits[1].argmax().item())
                    value_deltas.append(
                        float(values[1].item() - values[0].item()))
                if a != b:
                    changed += 1

            # Advance the match with the real observation. Hidden state is
            # carried so the states sampled are the ones a recurrent policy
            # actually reaches, rather than a reflex policy's trajectory.
            with torch.no_grad():
                o = torch.from_numpy(obs).unsqueeze(0)
                feats, _, _ = net.extract_features(o)
                out = net.step_lstm_and_card(feats, (hx, cx))
                slot = int(out[0].argmax().item())
                hx, cx = net.lstm(feats, (hx, cx))
            result = env.step(slot, 9.0, 8.0)
            if result.done:
                break

    print(f"states scored          {total}")
    print(f"states skipped (~full) {skipped}")
    if not total:
        print("\nno damaged states sampled -- raise --episodes")
        return 1
    print(f"\nARGMAX CARD CHANGED    {changed} / {total} "
          f"({100*changed/total:.1f}%)")
    d = np.array(value_deltas)
    print(f"critic value shift     mean {d.mean():+.4f}  "
          f"absmean {np.abs(d).mean():.4f}  max|.| {np.abs(d).max():.4f}")
    print("\nBoth arms are the same net on the same state with the same hidden")
    print("state; only the six tower scalars differ. Any disagreement is caused")
    print("by them alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
