"""Behaviour cloning: supervised warm-start for MicroRoyaleNet from
demonstrations.

    demonstrations  ->  .npz  ->  behaviour cloning  ->  warm-started net

Self-play discovers strategies but not the distribution humans play; AlphaStar
started from supervised learning on human replays. The two halves meet at a
file with a fixed schema (`DATASET_SCHEMA`), so human demonstrations from
`perception/` can be dropped in unchanged. Today the scripted opponents supply
demonstrations, which validates the pipeline against a known teacher; cloning
them is a warm start at best, not a source of skill, and the clone quality
measured so far does not justify using it. Not wired into either trainer (see
the note above `__main__`).
"""

import os
import sys

import numpy as np

from python_ai.rl.seeding import seed_everything
import torch

from python_ai.rl.checkpointing import atomic_save

from python_ai.rl.optim_step import clip_and_step
import torch.nn.functional as F

# Run as a script, the repo root is not on sys.path; importing the package also
# makes `clash_royale_env` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import python_ai  # noqa: E402,F401

import clash_royale_env  # noqa: E402
from python_ai.envs import gym_wrapper
from python_ai.models.net import MicroRoyaleNet

DATASET_SCHEMA = """
Arrays in the .npz, all with the same leading dimension N (one row per
DECISION STEP, in chronological order):

    obs        float32 (N, obs_dim)   observation from the ACTING player's own
                                      point of view -- i.e. what
                                      ClashEnv::getObservationForTeam(team)
                                      returns for whichever side is acting.
                                      Never the raw team-0 frame for a team-1
                                      actor; the network always believes it is
                                      team 0 and the mirroring is what makes
                                      that valid.
    card       int64   (N,)           chosen hand slot in [0, HAND_SIZE], where
                                      HAND_SIZE itself means "no-op / wait".
    cell       int64   (N,)           chosen board cell = row * BOARD_WIDTH +
                                      col, in the SAME mirrored frame as obs.
                                      Ignored by the loss on no-op rows.
    episode    int64   (N,)           episode index; rows sharing a value are
                                      one continuous trajectory, in order. The
                                      LSTM is replayed per episode, so this
                                      must be right or the recurrence is
                                      trained on spliced states.

obs_dim must equal ClashEnv::observationSize() of the engine build being
trained against. A dataset recorded against an older observation layout is not
silently usable -- load_dataset refuses it rather than letting the mismatch
turn into garbage features.
"""

BC_TEACHERS = ["Rusher", "Defender", "Cycler", "Counter"]


def _cell_from_xy(x, y, board_width, board_height):
    """Continuous engine coordinates -> the discrete cell the placement head
    predicts.

    Truncation, not rounding: the engine bins positions with
    `static_cast<int>`. Rounding put a placement on the last own-half row plus
    0.5 into the next row, outside the legal region, where its target was
    silently dropped.
    """
    col = int(np.clip(int(float(x)), 0, board_width - 1))
    row = int(np.clip(int(float(y)), 0, board_height - 1))
    return row * board_width + col


def collect_demonstrations(n_episodes=40, teachers=None, seed=0, verbose=True):
    """Generate demonstrations from the scripted teachers.

    Team 1 is the teacher (recorded, already in its own mirrored frame, which
    is the frame the network trains in); team 0 plays randomly to vary the
    board.
    """
    from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv

    teachers = teachers or BC_TEACHERS
    rng = np.random.default_rng(seed)
    CE = clash_royale_env.ClashRoyaleEnv
    W, H = CE.BOARD_WIDTH, CE.BOARD_HEIGHT

    obs_rows, card_rows, cell_rows, ep_rows = [], [], [], []
    env = MicroRoyaleSelfPlayEnv({"scenarios_enabled": False})

    for ep in range(n_episodes):
        env.set_scripted_opponent(teachers[ep % len(teachers)])
        env.game.reset()
        for _t in range(400):
            obs1 = np.array(env.game.get_observation_for_team(1), dtype=np.float32)
            slot, tx, ty, _a1, _a2 = env._scripted_opponent_action(obs1)

            obs_rows.append(obs1)
            card_rows.append(int(slot))
            cell_rows.append(_cell_from_xy(tx, ty, W, H))
            ep_rows.append(ep)

            # Team 0 plays at random: a state generator, not a demonstrator.
            c0 = int(rng.integers(0, CE.HAND_SIZE + 1))
            x0 = float(rng.integers(0, W))
            y0 = float(rng.integers(0, CE.BOARD_HEIGHT))
            res = env.game.step_self_play(c0, x0, y0, slot, float(tx), float(ty), 10,
                                          False, False, False, False)
            if res.done:
                break
        if verbose and (ep + 1) % 10 == 0:
            print(f"  collected {ep + 1}/{n_episodes} episodes, {len(obs_rows)} steps")

    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "card": np.asarray(card_rows, dtype=np.int64),
        "cell": np.asarray(cell_rows, dtype=np.int64),
        "episode": np.asarray(ep_rows, dtype=np.int64),
    }


def save_dataset(data, path):
    np.savez_compressed(path, **data)
    return path


def load_dataset(path):
    """Load and validate against the current engine's observation size, refusing a
    dataset recorded against another layout.
    """
    z = np.load(path)
    data = {k: z[k] for k in ("obs", "card", "cell", "episode")}
    expected = clash_royale_env.ClashRoyaleEnv(
        list(gym_wrapper.DEFAULT_DECK), list(gym_wrapper.DEFAULT_DECK), 100).observation_size()
    got = data["obs"].shape[1]
    if got != expected:
        raise ValueError(
            f"{path}: observation size {got} does not match this engine build's "
            f"{expected}. The dataset was recorded against a different observation "
            f"layout and cannot be used without re-recording. See DATASET_SCHEMA.")
    n = len(data["card"])
    for k, v in data.items():
        if len(v) != n:
            raise ValueError(f"{path}: array '{k}' has {len(v)} rows, expected {n}")
    return data


def train_bc(data, net=None, epochs=6, lr=1e-3, batch_episodes=8, device=None,
             placement_weight=1.0, verbose=True, sample_weights=None,
             episode_filter=None):
    """Fit a policy to the demonstrations by maximum likelihood.

    `sample_weights`: optional per-row floats (None keeps the unweighted path),
    for upweighting the rows where an expert disagrees with the policy.
    `episode_filter`: train on a subset of episodes, keeping a held-out split.

    Trained through the LSTM in episode order: a recurrent policy fitted on
    shuffled frames behaves differently when rolled out with real hidden state.
    The placement loss is masked on no-op rows, where the engine ignores
    placement.
    """
    device = device or torch.device("cpu")
    net = net or MicroRoyaleNet(num_ability_slots=gym_wrapper.DEFAULT_DECK_ABILITY_SLOTS)
    net = net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    ep_ids = np.unique(data["episode"])
    if episode_filter is not None:
        keep = set(int(e) for e in episode_filter)
        ep_ids = np.asarray([e for e in ep_ids if int(e) in keep])
    obs_all = torch.tensor(data["obs"]).to(device)
    card_all = torch.tensor(data["card"]).to(device)
    cell_all = torch.tensor(data["cell"]).to(device)
    weight_all = (torch.tensor(np.asarray(sample_weights, dtype=np.float32)).to(device)
                  if sample_weights is not None else None)

    history = []
    for epoch in range(epochs):
        np.random.shuffle(ep_ids)
        tot_loss = tot_card = tot_cell = 0.0
        n_batches = 0
        # Count and report dropped (mask-illegal) targets rather than skipping
        # them silently: a dataset whose discretization disagrees with the mask
        # is otherwise dropped wholesale while training appears to run.
        dropped_card = dropped_cell = seen_card = seen_cell = 0
        for b in range(0, len(ep_ids), batch_episodes):
            chunk = ep_ids[b:b + batch_episodes]
            loss = 0.0
            card_hits = cell_hits = steps = 0
            for e in chunk:
                idx = np.where(data["episode"] == e)[0]
                o = obs_all[idx]
                ca = card_all[idx]
                ce = cell_all[idx]
                feats, embeds, spatial = net.extract_features(o)
                mask = net.affordability_mask(o)
                hx = torch.zeros(1, 256, device=device)
                cx = torch.zeros(1, 256, device=device)
                card_lp, cell_lp = [], []
                for t in range(len(idx)):
                    cl, _, _, _, (hx, cx) = net.step_lstm_and_card(
                        feats[t:t + 1], (hx, cx), mask[t:t + 1])
                    pl = net.placement_given_card(
                        hx, embeds[t:t + 1], ca[t:t + 1], o[t:t + 1], spatial[t:t + 1])
                    card_lp.append(cl)
                    cell_lp.append(pl)
                card_lp = torch.cat(card_lp)
                cell_lp = torch.cat(cell_lp)

                # A target the affordability mask forbids has -inf logit and
                # infinite cross-entropy; drop it.
                legal = torch.isfinite(card_lp.gather(1, ca.view(-1, 1)).squeeze(1))
                seen_card += int(legal.numel())
                dropped_card += int((~legal).sum())
                if legal.sum() == 0:
                    continue
                # None keeps the original unweighted call exactly.
                w = weight_all[idx] if weight_all is not None else None
                if w is None:
                    l_card = F.cross_entropy(card_lp[legal], ca[legal])
                else:
                    per = F.cross_entropy(card_lp[legal], ca[legal], reduction="none")
                    wl = w[legal]
                    l_card = (per * wl).sum() / wl.sum().clamp_min(1e-8)

                played = legal & (ca != net.hand_size)
                if played.sum() > 0:
                    cell_legal = torch.isfinite(
                        cell_lp[played].gather(1, ce[played].view(-1, 1)).squeeze(1))
                    seen_cell += int(cell_legal.numel())
                    dropped_cell += int((~cell_legal).sum())
                    if cell_legal.sum() > 0:
                        sel = torch.where(played)[0][cell_legal]
                        if w is None:
                            l_cell = F.cross_entropy(cell_lp[sel], ce[sel])
                        else:
                            per_c = F.cross_entropy(cell_lp[sel], ce[sel], reduction="none")
                            wc = w[sel]
                            l_cell = (per_c * wc).sum() / wc.sum().clamp_min(1e-8)
                        cell_hits += int((cell_lp[sel].argmax(1) == ce[sel]).sum())
                    else:
                        l_cell = torch.zeros((), device=device)
                else:
                    l_cell = torch.zeros((), device=device)

                loss = loss + l_card + placement_weight * l_cell
                card_hits += int((card_lp[legal].argmax(1) == ca[legal]).sum())
                steps += int(legal.sum())
                tot_card += float(l_card.detach())
                tot_cell += float(l_cell.detach())

            if not torch.is_tensor(loss):
                continue
            opt.zero_grad()
            loss.backward()
            clip_and_step(opt, net.parameters(), 1.0)
            tot_loss += float(loss)
            n_batches += 1

        acc = card_hits / max(1, steps)
        card_drop = dropped_card / max(1, seen_card)
        cell_drop = dropped_cell / max(1, seen_cell)
        history.append({"epoch": epoch, "loss": tot_loss / max(1, n_batches),
                        "card_acc": acc,
                        "card_drop_rate": card_drop, "cell_drop_rate": cell_drop})
        if verbose:
            warn = ""
            # Not tuned: a systematic mismatch drops ~100% of a teacher's
            # placements, not a few percent.
            if cell_drop > 0.2 or card_drop > 0.2:
                warn = ("   <-- WARNING: a large share of demonstrations is being "
                        "discarded as mask-illegal; check the dataset's "
                        "discretization against the engine's")
            print(f"  epoch {epoch}: loss {tot_loss / max(1, n_batches):.4f}  "
                  f"card-match {acc:.3f}  "
                  f"dropped card/cell {card_drop:.1%}/{cell_drop:.1%}{warn}")
    return net, history


def action_match_rate(net, data, device=None, limit_episodes=10):
    """Fraction of held-out teacher decisions the policy reproduces greedily,
    reported separately for card and placement.

    Judge by `card_match_decisions`: plain `card_match` includes forced no-op
    steps, where any net scores 100%.
    """
    device = device or torch.device("cpu")
    net = net.to(device).eval()
    ep_ids = np.unique(data["episode"])[:limit_episodes]
    card_ok = card_n = cell_ok = cell_n = 0
    dec_ok = dec_n = 0
    with torch.no_grad():
        for e in ep_ids:
            idx = np.where(data["episode"] == e)[0]
            o = torch.tensor(data["obs"][idx]).to(device)
            ca = torch.tensor(data["card"][idx]).to(device)
            ce = torch.tensor(data["cell"][idx]).to(device)
            feats, embeds, spatial = net.extract_features(o)
            mask = net.affordability_mask(o)
            hx = torch.zeros(1, 256, device=device)
            cx = torch.zeros(1, 256, device=device)
            for t in range(len(idx)):
                cl, _, _, _, (hx, cx) = net.step_lstm_and_card(
                    feats[t:t + 1], (hx, cx), mask[t:t + 1])
                pred_card = int(cl.argmax(1))
                hit = int(pred_card == int(ca[t]))
                card_ok += hit
                card_n += 1
                if int(mask[t].sum()) > 1:      # a real choice existed here
                    dec_ok += hit
                    dec_n += 1
                if int(ca[t]) != net.hand_size:
                    pl = net.placement_given_card(
                        hx, embeds[t:t + 1], ca[t:t + 1], o[t:t + 1], spatial[t:t + 1])
                    cell_ok += int(int(pl.argmax(1)) == int(ce[t]))
                    cell_n += 1
    return {"card_match": card_ok / max(1, card_n),
            "card_match_decisions": dec_ok / max(1, dec_n),
            "cell_match": cell_ok / max(1, cell_n),
            "card_n": card_n, "decision_n": dec_n, "cell_n": cell_n}


# Deliberately not called from the trainers. Cloning the scripted bots is of
# unclear and possibly negative value (they are weak and narrow); the module
# exists for the pipeline, so that real human demonstrations in DATASET_SCHEMA
# format can be used without changing it.
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collect", type=int, metavar="N_EPISODES",
                    help="generate demonstrations from the scripted teachers")
    ap.add_argument("--data", default="bc_demos.npz", help="dataset path")
    ap.add_argument("--train", action="store_true", help="fit a policy to --data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="bc_warmstart.pth")
    ap.add_argument("--seed", type=int, default=None,
                   help="Seed every RNG so this run reproduces. Off by "
                        "default: seeding by default would change what "
                        "every existing invocation does.")
    args = ap.parse_args()
    # Before any data is loaded or net built: the shuffle and the
    # initialisation both draw from the global RNG.
    seed_everything(args.seed)

    if args.collect:
        d = collect_demonstrations(n_episodes=args.collect)
        save_dataset(d, args.data)
        print(f"wrote {len(d['card'])} steps to {args.data}")
    if args.train:
        d = load_dataset(args.data)
        model, hist = train_bc(d, epochs=args.epochs)
        atomic_save({"model": model.state_dict()}, args.out)
        print(f"wrote warm-start weights to {args.out}")
        print("  ", action_match_rate(model, d))
