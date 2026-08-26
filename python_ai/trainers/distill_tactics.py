"""Teach the placement head the dead cards' geometry from the advisor.

WHY THE ENTROPY COVERAGE TERM WAS NOT ENOUGH -- MEASURED, NEGATIVE
------------------------------------------------------------------
`PLACEMENT_COVERAGE_COEF` restores a gradient to cards the policy never plays,
and a 3-arm run (80 PPO updates, matched episodes, byte-identical code) shows it
does exactly what it says and that this is not sufficient:

    Cannon, ep~64800    modal cell   modal share   top-1 p   engine-scored
    seed                  (11,0)        88.7%       0.920      135.9 HP
    control (coef 0)      (11,0)        55.4%       0.550      116.4 HP
    treatment (coef .02)   (6,0)        79.4%       0.051      182.0 HP
    random legal cell        --            --          --      394.7 HP

Top-1 probability 0.051 for Cannon and 0.006 for Fireball is a near-UNIFORM
distribution (uniform is 1/612 = 0.0016). Entropy went up exactly as designed --
and the head still returns one fixed cell, because **argmax of a flat map is an
arbitrary constant**. The lock moved from (11,0) to (6,0); it did not break.

The lesson is that entropy is a MARGINAL objective. It says "be spread out", not
"depend on the board", and a card with no other gradient has nothing telling it
WHICH cell is right in WHICH state. Closing a coverage hole needs a target, not
just noise.

WHAT THIS DOES
--------------
Supplies that target from `tactics.py`, whose cells are measured against the
engine's own accounting rather than assumed good, and distils it into the
placement head for the dead cards only.

Two properties make this safe rather than a blunt overwrite:

* Everything except the placement pathway is FROZEN. That follows the recipe
  expert_iteration.py already established on this codebase -- frozen beat full
  fine-tuning there (+0.1027 vs +0.1415 lift but zero critic drift and a sharper
  selectivity ratio), and the critic must not move because it is the scorer that
  decision-time search depends on.
* The cards that are ALIVE are anchored to their own current distribution with a
  KL term. `place_ctx` and `place_up` are shared across every card, so training
  Cannon and Fireball alone would silently drag the five working cards with them.
  The anchor is what makes this a targeted repair instead of a trade.

    python_ai/venv/Scripts/python.exe python_ai/distill_tactics.py \
        --net model_weights_selfplay.pth --out model_weights_tactical.pth
"""
import argparse
import os
import sys

import numpy as np

from python_ai.rl.seeding import seed_everything
import torch

from python_ai.rl.checkpointing import atomic_save

from python_ai.rl.optim_step import clip_and_step
import torch.nn.functional as F

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env as E  # noqa: E402
from python_ai.envs import gym_wrapper  # noqa: E402
from python_ai.advisors import tactics  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402

CE = E.ClashRoyaleEnv
CANNON, FIREBALL = tactics.CANNON_ID, tactics.FIREBALL_ID
# The cards measured dead: never played, placement head a constant function.
TARGET_CARDS = (CANNON, FIREBALL)


@torch.no_grad()
def collect(net, episodes, opp_elixir, device, want_maps=False):
    """Play with `net` and record (obs, per-card advisor cell) at each step.

    States come from the net's OWN trajectory on purpose: that is the
    distribution the placement head will be queried on in play, so it is the one
    it should be correct on.

    With `want_maps`, also returns the advisor's full per-cell SCORE MAP for
    each card (illegal cells -inf), which is the target a distribution-valued
    distillation needs. The argmax alone throws away the margin -- it cannot
    say whether the runner-up was equally good or a blunder -- and for the
    Cannon it is actively misleading, because `building_score_map` is built
    from flat discs whose top is a large exact-tie plateau resolved by
    row-major order (see that function, and prove_hires.py).
    """
    deck = list(gym_wrapper.DEFAULT_DECK)
    legal = {c: net._placement_legal[c].numpy().astype(bool) for c in TARGET_CARDS}
    obs_rows, hx_rows, targets = [], [], {c: [] for c in TARGET_CARDS}
    maps = {c: [] for c in TARGET_CARDS}

    for ep in range(episodes):
        env = CE(deck, deck, 3600)
        env.set_opponent_elixir_multiplier(opp_elixir)
        env.reset()
        hx, cx = torch.zeros(1, 256, device=device), torch.zeros(1, 256, device=device)
        obs = env.get_observation_for_team(0)
        for _t in range(400):
            o = np.asarray(obs, dtype=np.float32)
            t = torch.tensor(o, device=device).unsqueeze(0)

            cx_, cy_, cover = tactics.best_building_cell(o, legal=legal[CANNON])
            fx_, fy_, catch = tactics.best_spell_cell(o, legal=legal[FIREBALL])
            # Only keep states where the advisor has something to say. With an
            # empty board its "target" is a default pocket/arbitrary cell, and
            # training on those would teach a constant -- the exact failure this
            # module exists to undo.
            feats, embeds, sp = net.extract_features(t)
            mask = net.affordability_mask(t)
            logits, _, _, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), mask)

            # Recorded AFTER the LSTM step, which is the state
            # placement_given_card is actually called with at inference. Training
            # on a zero hidden state instead would fit the head under a context
            # it never sees in play.
            if cover > 0.0 or catch > 0.0:
                obs_rows.append(o)
                hx_rows.append(hx[0].detach().cpu().numpy().copy())
                targets[CANNON].append(int(cy_) * tactics.BOARD_W + int(cx_))
                targets[FIREBALL].append(int(fy_) * tactics.BOARD_W + int(fx_))
                if want_maps:
                    # Same masking the advisor itself applies, so the surface
                    # being distilled and the cell being played come from one
                    # set of legal cells. -inf (not -1) on illegal: the target
                    # is consumed as logits, and a finite floor would leave
                    # real probability mass on a cell the engine refuses.
                    cm = tactics.building_score_map(o, legal=legal[CANNON])
                    fm = tactics.spell_catch_map(o).reshape(-1).copy()
                    fm[~legal[FIREBALL].reshape(-1)] = -np.inf
                    maps[CANNON].append(cm.reshape(-1).astype(np.float32))
                    maps[FIREBALL].append(fm.astype(np.float32))
            gi = int(logits.argmax(-1).item())
            place = net.placement_given_card(hx, embeds, torch.tensor([gi], device=device), t, sp)
            cell = int(place.argmax(-1).item())
            r = env.step(gi, float(cell % 18), float(cell // 18), 10)
            obs = r.observation
            if r.done:
                break
        print(f"  ep{ep}: {len(obs_rows)} usable states", flush=True)
    out = (np.asarray(obs_rows, dtype=np.float32),
           np.asarray(hx_rows, dtype=np.float32),
           {c: np.asarray(v) for c, v in targets.items()})
    if want_maps:
        return out + ({c: np.asarray(v, dtype=np.float32)
                       for c, v in maps.items()},)
    return out


def masked_kl(new_logits, old_logits):
    """KL(old || new) over the LEGAL cells only.

    `placement_mask` fills illegal cells with -inf, and torch's kl_div then
    evaluates 0 * (-inf - -inf) = 0 * nan = nan at every one of them, which
    poisons the whole loss -- observed as `loss nan` from the first batch, with
    the freeze verified intact, so the defect was arithmetic and not the model.
    Zeroing the non-finite terms is correct rather than a patch: those cells
    carry no probability mass in either distribution, so their true
    contribution to the divergence is exactly zero.
    """
    ln = F.log_softmax(new_logits, -1)
    lo = F.log_softmax(old_logits, -1)
    term = lo.exp() * (lo - ln)
    term = torch.where(torch.isfinite(term), term, torch.zeros_like(term))
    return term.sum(-1).mean()


def slot_of(net, obs_t, card_id):
    """Hand slot holding `card_id`, or -1. (B,) long."""
    hand = net.hand_card_ids(obs_t)
    return torch.where((hand == card_id).any(dim=1),
                       (hand == card_id).float().argmax(dim=1),
                       torch.full((obs_t.shape[0],), -1, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model_weights_selfplay.pth")
    ap.add_argument("--out", default="model_weights_tactical.pth")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=1.0,
                    help="weight on keeping the ALIVE cards where they are")
    ap.add_argument("--opp-elixir", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=None,
                   help="Seed every RNG so this run reproduces. Off by "
                        "default: seeding by default would change what "
                        "every existing invocation does.")
    args = ap.parse_args()
    # Applied BEFORE any data is loaded or any net is built: the
    # per-epoch shuffle below runs on the GLOBAL RNG, and the net's
    # initialisation is itself a draw. Seeding after either would leave
    # the run half-reproducible, which is worse than not at all.
    seed_everything(args.seed)

    here = python_ai.PACKAGE_DIR
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    net = load_net(os.path.join(here, args.net), dev)
    frozen = load_net(os.path.join(here, args.net), dev)   # the anchor reference
    for p in frozen.parameters():
        p.requires_grad_(False)

    print(f"\ncollecting {args.episodes} episodes at {args.opp_elixir}x ...")
    obs, hxs, targets = collect(net, args.episodes, args.opp_elixir, dev)
    print(f"collected {len(obs)} states\n")

    # Only the placement pathway trains. card_id_embed is included because it is
    # the per-card parameter the coverage hole starves -- see
    # test_python_ai.py.
    trainable = []
    for name, p in net.named_parameters():
        train_it = name.startswith(("place_ctx", "place_up", "card_id_embed"))
        p.requires_grad_(train_it)
        if train_it:
            trainable.append(p)
    n_train = sum(p.numel() for p in trainable)
    print(f"training {n_train:,} of {sum(p.numel() for p in net.parameters()):,} params "
          f"(placement pathway only)\n")
    opt = torch.optim.Adam(trainable, lr=args.lr)

    deck_slots = list(range(net.hand_size))
    n = len(obs)
    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        tot, seen, hit = 0.0, 0, {c: 0 for c in TARGET_CARDS}
        cnt = {c: 0 for c in TARGET_CARDS}
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            ob = torch.tensor(obs[idx], device=dev)
            B = ob.shape[0]
            hx = torch.tensor(hxs[idx], device=dev)

            feats, embeds, sp = net.extract_features(ob)
            with torch.no_grad():
                f2, e2, s2 = frozen.extract_features(ob)

            loss = torch.zeros((), device=dev)
            # --- the repair: dead cards learn the advisor's geometry --------
            for cid in TARGET_CARDS:
                slot = slot_of(net, ob, cid)
                m = slot >= 0
                if not bool(m.any()):
                    continue
                logits = net.placement_given_card(
                    hx[m], embeds[m], slot[m].clamp(min=0), ob[m], sp[m])
                tgt = torch.tensor(targets[cid][idx][m.numpy()], dtype=torch.long, device=dev)
                loss = loss + F.cross_entropy(logits, tgt)
                hit[cid] += int((logits.argmax(-1) == tgt).sum())
                cnt[cid] += int(m.sum())

            # --- the guard: alive cards stay where they are -----------------
            if args.anchor > 0:
                anchor = torch.zeros((), device=dev)
                for s in deck_slots:
                    si = torch.full((B,), s, dtype=torch.long, device=dev)
                    new = net.placement_given_card(hx, embeds, si, ob, sp)
                    with torch.no_grad():
                        old = frozen.placement_given_card(hx, e2, si, ob, s2)
                    # Skip the slots currently holding a target card: those are
                    # supposed to move.
                    keep = torch.ones(B, dtype=torch.bool, device=dev)
                    for cid in TARGET_CARDS:
                        keep &= slot_of(net, ob, cid) != s
                    if bool(keep.any()):
                        anchor = anchor + masked_kl(new[keep], old[keep])
                loss = loss + args.anchor * anchor

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_and_step(opt, trainable, 0.5)
            tot += loss.item() * B
            seen += B
        acc = "  ".join(f"{'Cannon' if c == CANNON else 'Fireball'} "
                        f"{hit[c]/max(1,cnt[c]):.1%}" for c in TARGET_CARDS)
        print(f"  epoch {epoch}: loss {tot/max(1,seen):.4f}   argmax match: {acc}")

    out = os.path.join(here, args.out)
    atomic_save({"model": net.state_dict()}, out)
    print(f"\nsaved {out}")

    # Verify the freeze actually held -- the same check expert_iteration.py
    # makes, because a silently-drifting critic degrades every future search.
    with torch.no_grad():
        ob = torch.tensor(obs[:64], device=dev)
        hx = torch.tensor(hxs[:64], device=dev)
        a = net.value_head(hx)
        b = frozen.value_head(hx)
        print(f"critic drift |dV| = {float((a - b).abs().max()):.6f} (must be 0.000000)")


if __name__ == "__main__":
    main()
