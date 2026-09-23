"""Per-opponent-deck matchup measurement. Read-only; never touches a live run.

THE QUESTION
------------
The 2026-08-29 deck autopsy established that three of `DEFAULT_DECK`'s eight
cards sit at P(play | in hand) <= 0.009 and that the card head is RIGHT to price
them there: forced through `env.step` (which charges the elixir, unlike
`inject`), a Cannon at the policy's own cell was worth +185 tower HP over the
policy's own action -- better in 6 of 14 states. A coin flip.

Every one of those measurements held the OPPONENT'S DECK fixed at the 2.6 mirror.
This harness varies it, because the three dead cards are precisely 2.6's answers
to threats a 2.6 mirror does not produce:

    Cannon    a tank walking at your tower       mirror offers: one Hog
    Fireball  a medium-HP cluster                mirror offers: one Musketeer
    The Log   a ground swarm / bait              mirror offers: 1-elixir Skeletons

If the dead-card verdict is a property of the AGENT, these numbers barely move
across decks. If it is a property of the OPPONENT DISTRIBUTION we chose, they
move a lot, and the fix is the deck pool rather than any pressure on the policy.

WHAT IS REPORTED, AND WHY BOTH HALVES ARE NEEDED
------------------------------------------------
  win / loss / draw      the WINNABILITY gate. `gym_wrapper`'s own comment
                         records that random registry decks beat a cycle deck
                         "by tens of win-rate points" in this engine, so a deck
                         pool has to be screened or it recreates the zero-
                         gradient state the curriculum pivot exists to avoid.

  fb_catch / log_catch   OPPORTUNITY: the best enemy value a Fireball / Log
                         aimed anywhere could catch, per decision. This is the
                         instrument CLAUDE.md already uses for the Fireball
                         question ("median ONE unit" in the mirror), read here
                         as an upper bound with perfect information -- it says
                         what the BOARD offers, independent of whether this
                         policy could find it.

  threat_hp              enemy troop HP on our half: the Cannon's value driver,
                         via `tactics.threat_level`, the same definition the
                         solvency gate and the deck-coverage gate already use.

  take-up                what the policy actually DID with the opportunity.

Opportunity and take-up must be read as a PAIR. Opportunity up with take-up flat
is an environment that now rewards the card and a policy that has not noticed --
which is a training target. Opportunity flat is the mirror hypothesis refuted.

Usage:

    python_ai/venv/Scripts/python.exe -m python_ai.eval.measure_deck_matchups \\
        --weights model_weights_phase5.pth --episodes 24 --stage 3
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.models.net import MicroRoyaleNet  # noqa: E402
from python_ai.models.policy_io import load_state_dict_flexible  # noqa: E402
from python_ai.opponents import deck_pool  # noqa: E402

#: The Log's corridor, from CardRegistry via CLAUDE.md's rolling-spell block:
#: 3.9 wide x 10.1 long, swept forward from the cast point. A disc of the same
#: AREA would be the wrong shape -- the whole point of the 2026-08-28 change is
#: that the Log is a narrow moving rectangle, not the 7.8-wide circle it was.
LOG_WIDTH = 3.9        # CardRegistry.h: The Log's roll width  (not bound)
LOG_RANGE = 10.1       # CardRegistry.h: The Log's roll range  (not bound)
#: The Log's damage, for the overkill cap -- MEASURED, not restated. This was a
#: literal 240.0 labelled "CardRegistry.h" while the registry says 269, so every
#: Log-opportunity figure measured with it capped each body 11% low. Numbers
#: recorded before 2026-09-23 (CLAUDE.md's 145 / 273 / 427) used the old cap.
LOG_DAMAGE = None


def _log_damage():
    global LOG_DAMAGE
    if LOG_DAMAGE is None:
        from python_ai.advisors import card_probes
        log_id = next(c for c in E.get_all_card_ids()
                      if E.get_card_info(c)["name"] == "The Log")
        LOG_DAMAGE = card_probes.roller_damage(log_id)
    return LOG_DAMAGE


def log_catch_map(obs):
    """(34,18) map of enemy value a Log cast at each cell would sweep.

    Deliberately NOT `spell_catch_map` with a fudged radius: the corridor is
    3.9 x 10.1 and forward-only, so a disc gets both the lateral reach and the
    longitudinal reach wrong in opposite directions. Same damage-cap convention
    as `spell_catch_map` (overkill is wasted), for the same reason.

    Computed as a SEPARABLE box filter rather than the obvious per-cell double
    loop, because a rectangle is separable and the loop version was the
    dominant cost of this whole harness: ~200 Python-level iterations per
    occupied enemy cell per decision, against ~5,000 decisions per deck.
    `_log_catch_map_reference` keeps the loop as the test oracle.
    """
    hp = tactics.enemy_hp_map(obs)
    count = np.maximum(1.0, tactics.spatial(obs)[tactics.CH_ENEMY_COUNT]
                       * tactics.MAX_CELL_UNITS)
    effective = np.minimum(hp, _log_damage() * count).astype(np.float64)

    # Lateral: a cast at column ax catches column x when |ax - x| <= 1.95, i.e.
    # dx in {-1, 0, +1} on the integer grid.
    half_w = int(LOG_WIDTH / 2.0)          # 1
    lat = np.zeros_like(effective)
    for dx in range(-half_w, half_w + 1):
        lat += np.roll(effective, dx, axis=1) * _col_valid(dx)

    # Longitudinal: a cast at row ay sweeps FORWARD to ay + LOG_RANGE, so it
    # catches rows [ay, ay + reach]. A forward window sum is one cumsum.
    reach = int(LOG_RANGE)                 # 10
    csum = np.cumsum(lat, axis=0)
    padded = np.vstack([csum, np.repeat(csum[-1:], reach + 1, axis=0)])
    lo = np.vstack([np.zeros((1, lat.shape[1])), csum[:-1]])
    return (padded[reach:reach + lat.shape[0]] - lo).astype(np.float32)


def _col_valid(dx):
    """Column mask killing the wrap `np.roll` introduces at the board edge."""
    m = np.ones((1, tactics.BOARD_W))
    if dx > 0:
        m[0, :dx] = 0.0
    elif dx < 0:
        m[0, dx:] = 0.0
    return m


def _log_catch_map_reference(obs):
    """The literal double loop `log_catch_map` replaces. Test oracle only.

    The row bound is `ceil(y - LOG_RANGE)` and NOT `int(y - LOG_RANGE)`. A cast
    at row `ay` reaches `ay + 10.1`, so it catches row y exactly when
    `ay >= y - 10.1`, and on the integer grid that is a CEILING. `int()`
    truncates toward zero, which admits one row too many and quietly makes the
    Log 11.1 tiles long instead of 10.1 -- the same half-tile convention error
    that put the arena off-centre, one axis over.
    """
    hp = tactics.enemy_hp_map(obs)
    count = np.maximum(1.0, tactics.spatial(obs)[tactics.CH_ENEMY_COUNT]
                       * tactics.MAX_CELL_UNITS)
    effective = np.minimum(hp, _log_damage() * count)
    out = np.zeros((tactics.BOARD_H, tactics.BOARD_W), dtype=np.float64)
    half_w = LOG_WIDTH / 2.0
    ys, xs = np.nonzero(effective)
    for y, x in zip(ys, xs):
        v = effective[y, x]
        lo = max(0, int(np.ceil(y - LOG_RANGE)))
        for ay in range(lo, y + 1):
            for ax in range(tactics.BOARD_W):
                if abs(ax - x) <= half_w:
                    out[ay, ax] += v
    return out


def build_net(weights):
    here = python_ai.PACKAGE_DIR
    path = weights if os.path.isabs(weights) else os.path.join(here, weights)
    net = MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    ep = "?"
    if os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        load_state_dict_flexible(net, sd, os.path.basename(path))
        ep = ck.get("episodes_completed", "?") if isinstance(ck, dict) else "?"
    else:
        print(f"!! {path} not found -- measuring an UNTRAINED net")
    net.eval()
    return net, ep


def run_deck(net, deck, *, episodes, stage, greedy, seed0, max_decisions=400):
    """One deck's arm. Returns a dict of aggregates."""
    ai_deck = list(gym_wrapper.DEFAULT_DECK)
    env = gym_wrapper.MicroRoyaleEnv({
        "ai_deck": ai_deck,
        "opp_deck": list(deck.card_ids),
        "opponent": "teacher",
        "teacher_stage": int(stage),
    })

    played, affordable = Counter(), Counter()
    fb, lg, thr, units = [], [], [], []
    wins = losses = draws = 0
    steps = 0

    for ep_i in range(episodes):
        # Seeded so every deck arm sees the SAME sequence of opening hands.
        # Unpaired win-rate comparisons at this n are dominated by the opening
        # shuffle otherwise -- CLAUDE.md's own note that ~1,568 episodes/arm are
        # needed to resolve 5 points UNPAIRED is the reason this is not optional.
        env.game.seed(seed0 + ep_i)
        obs, _ = env.reset()
        hx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        cx = torch.zeros(1, MicroRoyaleNet.LSTM_HIDDEN)
        term = trunc = False

        for _t in range(max_decisions):
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mask = net.affordability_mask(t)
                feats, embeds, spatial_f = net.extract_features(t)
                logits, _, _, _, (hx, cx) = net.step_lstm_and_card(
                    feats, (hx, cx), mask)
                idx = (logits.argmax(-1) if greedy
                       else Categorical(logits=logits).sample())
                place = net.placement_given_card(hx, embeds, idx, t, spatial_f)
                cell = (place.argmax(-1) if greedy
                        else Categorical(logits=place).sample())
            x, y = net.cell_to_xy(cell)

            # --- OPPORTUNITY, read off the same observation the decision saw.
            o = np.asarray(obs, dtype=np.float32)
            fb.append(float(tactics.spell_catch_map(o).max()))
            lg.append(float(log_catch_map(o).max()))
            thr.append(float(tactics.threat_level(o)))
            units.append(float((tactics.enemy_hp_map(o) > 0).sum()))

            hand_ids = net.hand_card_ids(t)[0].tolist()
            for slot_i, cid in enumerate(hand_ids):
                if cid >= 0 and bool(mask[0, slot_i]):
                    affordable[cid] += 1
            slot = int(idx.item())
            if slot < net.hand_size and hand_ids[slot] >= 0:
                played[hand_ids[slot]] += 1
            steps += 1

            obs, _r, term, trunc, info = env.step({
                "card_index": np.array([slot]),
                "target_x": x.numpy().reshape(1, 1),
                "target_y": y.numpy().reshape(1, 1),
                "activate_ability_slot1": np.zeros(1, dtype=np.int64),
                "activate_ability_slot2": np.zeros(1, dtype=np.int64),
            })
            if term or trunc:
                break

        # Outcome from the engine's own crown count, never from the shaped
        # reward -- the reward carries shaping terms and a draw penalty.
        a0 = env.game.get_towers_alive(0)
        a1 = env.game.get_towers_alive(1)
        if a0 > a1:
            wins += 1
        elif a1 > a0:
            losses += 1
        else:
            draws += 1

    n = max(1, episodes)
    total_plays = max(1, sum(played.values()))
    takeup = {}
    for c in ai_deck:
        share = played[c] / total_plays
        aff = affordable[c] / max(1, steps)
        takeup[c] = (share / aff) if aff > 0 else float("nan")

    return {
        "deck": deck.name,
        "archetype": deck.archetype,
        "tags": deck.tags,
        "avg_elixir": round(deck.avg_elixir, 2),
        "episodes": episodes,
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "decisions": steps,
        "fb_catch_mean": float(np.mean(fb)) if fb else 0.0,
        "fb_catch_median": float(np.median(fb)) if fb else 0.0,
        "fb_catch_p90": float(np.percentile(fb, 90)) if fb else 0.0,
        "log_catch_mean": float(np.mean(lg)) if lg else 0.0,
        "log_catch_median": float(np.median(lg)) if lg else 0.0,
        "threat_mean": float(np.mean(thr)) if thr else 0.0,
        "threat_p90": float(np.percentile(thr, 90)) if thr else 0.0,
        "enemy_units_mean": float(np.mean(units)) if units else 0.0,
        "takeup": {E.get_card_info(c)["name"]: round(takeup[c], 3)
                   for c in ai_deck},
        "play_share": {E.get_card_info(c)["name"]:
                       round(played[c] / total_plays, 4) for c in ai_deck},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model_weights_phase5.pth")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--sample", dest="greedy", action="store_false")
    ap.add_argument("--decks", default="", help="comma-separated subset")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    net, ep = build_net(args.weights)
    decks = deck_pool.load_pool()
    if args.decks:
        want = {s.strip() for s in args.decks.split(",") if s.strip()}
        decks = [d for d in decks if d.name in want]
        missing = want - {d.name for d in decks}
        if missing:
            raise SystemExit(f"unknown deck(s): {sorted(missing)}")

    print(f"weights={args.weights} (episode {ep})  stage={args.stage}  "
          f"episodes/deck={args.episodes}  mode="
          f"{'greedy' if args.greedy else 'sampled'}\n")

    rows = []
    for d in decks:
        t0 = time.time()
        row = run_deck(net, d, episodes=args.episodes, stage=args.stage,
                       greedy=args.greedy, seed0=args.seed)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"{row['deck']:<26} win {row['win_rate']:.3f}  "
              f"draw {row['draw_rate']:.2f}  "
              f"fb {row['fb_catch_mean']:7.1f}  log {row['log_catch_mean']:7.1f}  "
              f"threat {row['threat_mean']:7.1f}  ({row['seconds']}s)")
        sys.stdout.flush()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"weights": args.weights, "episode": ep,
                       "stage": args.stage, "episodes_per_deck": args.episodes,
                       "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
