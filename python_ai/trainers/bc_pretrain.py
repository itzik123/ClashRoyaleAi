"""Behaviour cloning: supervised warm-start for MicroRoyaleNet from demonstrations.

Why this exists
---------------
Self-play discovers strategies. It does not discover *the distribution of
strategies humans actually play*, and it starts from a policy that does not
know the game exists. AlphaStar's headline result was not pure self-play -- it
began with supervised learning on human replays, and the league was built on
top of that. Without the supervised stage it did not get anywhere.

This module is the supervised stage, plus the dataset format that stage runs
on. It is deliberately split into two halves that meet at a file:

    demonstrations  ->  .npz  ->  behaviour cloning  ->  warm-started net

WHAT WORKS TODAY: the demonstration side can be filled from the scripted
teachers already in `train_selfplay.py` (Rusher / Defender / Cycler / Counter).
That is genuinely useful on its own -- a net that starts at scripted-bot level
instead of random skips the phase where it is learning that cards exist -- and
it makes the whole pipeline testable right now.

WHAT IS STILL MISSING: real human demonstrations. `perception/` is the module
meant to produce those from recorded matches, and it is blocked on actually
having a recording. Nothing here needs to change when that lands: perception
just has to emit the same .npz schema documented in `DATASET_SCHEMA` below,
and `train_bc` consumes it unchanged. That is the entire reason the format is
pinned down in a docstring instead of being implicit in the collector.

Honest limitation: cloning a scripted teacher can only ever reach that
teacher's level. It is a warm start, not a source of skill. The value of doing
it now is that it validates the pipeline end to end against a teacher whose
behaviour is known, so that when human data arrives the only new variable is
the data.

MEASURED STATE, 2026-07-29 -- the pipeline works, the CLONE QUALITY does not
yet justify using it as a warm start:

    dataset       60 episodes / 4082 steps / 1799 placements over 75 cells
    mask drops    0.0% (after the truncation fix, see _cell_from_xy)
    train loss    51.96 -> 44.21 over 10 epochs, monotone
    held-out card 0.300 -> 0.319 on real decisions
    held-out cell 0.000 -> 0.073

The number that matters is the last one against the right baseline. Random
guessing on 612 cells is 0.0016, so 0.073 looks like a 45x win -- but the
teachers only ever use 75 distinct cells, and ALWAYS GUESSING THE SINGLE MOST
COMMON CELL scores 0.273. The clone is at 0.27x that. It has not yet learned
state-dependent placement at all; it has not even learned the marginal.

Almost certainly underfitting rather than a defect: four different teachers are
cycled into one dataset, and Defender/Counter place in response to where the
opponent's unit is, which needs far more than ~1400 placement examples. Fix by
scaling data and epochs, and check against the marginal baseline (not random)
before trusting any improvement.

This is exactly why nothing here is wired into train.py or train_selfplay.py --
see the note above the __main__ block.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Run as a script the repo root is not on sys.path, so `python_ai.*` cannot
# resolve; importing the package is also what makes `clash_royale_env` (an
# unpackaged .pyd in python_ai/) importable. See python_ai/__init__.py.
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
    """Continuous engine coordinates -> the discrete cell index the policy's
    placement head predicts.

    TRUNCATION, not rounding, and that distinction is load-bearing. The engine
    bins a position into a board cell with `static_cast<int>(position.y)` (see
    ClashEnv::extractObservationForTeam), so truncation is what "which cell is
    this unit in" already means everywhere else in the system.

    Rounding broke on the exact case the scripted teachers hit constantly: a
    Rusher places at `self.MAX_Y`, which was getOwnHalfMaxY() = 15.5 AT THE
    TIME (it is 15.0 since the 2026-07-29 river re-centring -- this paragraph
    is kept as the historical account of why truncation was chosen, not as a
    current statement of the constant), and
    round(15.5) = 16 -- one row PAST the last legal own-half row (0..15). The
    placement mask then marks that cell illegal, its logit is -inf, and the
    cross-entropy target can never be matched. Because train_bc drops
    mask-illegal targets rather than letting them produce inf loss, this
    failed silently: every Rusher demonstration was discarded and placement
    imitation sat at ~0.9%, barely above the 0.16% of random guessing.
    """
    col = int(np.clip(int(float(x)), 0, board_width - 1))
    row = int(np.clip(int(float(y)), 0, board_height - 1))
    return row * board_width + col


def collect_demonstrations(n_episodes=40, teachers=None, seed=0, verbose=True):
    """Generate demonstrations from the scripted teachers.

    Team 1 is the teacher (its actions are what we record); team 0 plays
    randomly, purely to produce varied board states for the teacher to react
    to. Recording team 1 rather than team 0 is deliberate -- the scripted bots
    only exist on the opponent side, and their observation is already mirrored
    into "my own point of view" form, which is exactly the frame the network
    trains in.
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

            # Team 0 plays at random -- a state generator, not a demonstrator.
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
    """Loads and VALIDATES against the current engine's observation size.

    The check is not defensive padding: this project has been bitten more than
    once by a layout change silently reinterpreting an old tensor, and a
    behaviour-cloning dataset recorded before an observation change would train
    the net on features that mean something else entirely, with no error and no
    obvious symptom beyond a policy that mysteriously fails to learn.
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

    `sample_weights` is an optional per-ROW float array (same length as
    data["card"]). None -- the default -- takes the original unweighted code
    path unchanged, so nothing about existing callers moves. It exists for
    expert-iteration distillation, where ~90% of labels already agree with what
    the policy does and the gradient is otherwise dominated by "keep doing what
    you already do"; upweighting the disagreement rows is the lever that tests
    whether the policy can learn the CONDITIONAL rule rather than the marginal.

    `episode_filter` restricts training to a subset of episode ids, so a
    held-out split can be kept back. Without one, "the policy fits the labels"
    and "the policy generalises" are the same number and neither is trustworthy.

    Trained through the LSTM in episode order rather than on shuffled
    independent frames: the network is recurrent, and a policy fitted on
    shuffled frames learns a feed-forward mapping that then behaves differently
    the moment it is rolled out with real hidden-state carryover.

    The placement loss is masked on no-op rows. On those steps the engine
    ignores the placement entirely (see ClashEnv::step's cardIndex bounds
    check), so the teacher's "position" is meaningless there and regressing on
    it would inject pure noise into the placement head.
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
        # Counted and REPORTED, not just silently skipped. Dropping targets the
        # action mask forbids is necessary (their logit is -inf, so their
        # cross-entropy is +inf and would destroy the batch), but a dataset
        # whose discretization disagrees with the mask gets *entirely* dropped
        # this way while training still appears to run. That is exactly what
        # happened once here: round() vs the engine's truncation put every
        # Rusher placement one row outside the legal own-half, and the only
        # symptom was a placement match rate that would not rise. A high drop
        # rate is now visible instead of invisible -- which matters most for
        # the case this module exists for, a human-replay dataset produced by
        # a separate pipeline that could discretize differently.
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

                # A teacher action the affordability mask marks illegal has
                # -inf logit, so its cross-entropy is +inf. Drop those rows
                # instead of poisoning the batch: they are teacher/engine
                # disagreements about affordability at the exact step
                # boundary, not something the policy should imitate.
                legal = torch.isfinite(card_lp.gather(1, ca.view(-1, 1)).squeeze(1))
                seen_card += int(legal.numel())
                dropped_card += int((~legal).sum())
                if legal.sum() == 0:
                    continue
                # weight_all is None on the original path, and the branch below
                # then calls F.cross_entropy exactly as before -- identical
                # numerics, not merely equivalent ones.
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
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
            # 20% is not a tuned threshold, just "clearly more than the
            # occasional boundary disagreement" -- a systematic discretization
            # mismatch drops ~100% of a teacher's placements, not a few percent.
            if cell_drop > 0.2 or card_drop > 0.2:
                warn = ("   <-- WARNING: a large share of demonstrations is being "
                        "discarded as mask-illegal; check the dataset's "
                        "discretization against the engine's")
            print(f"  epoch {epoch}: loss {tot_loss / max(1, n_batches):.4f}  "
                  f"card-match {acc:.3f}  "
                  f"dropped card/cell {card_drop:.1%}/{cell_drop:.1%}{warn}")
    return net, history


def action_match_rate(net, data, device=None, limit_episodes=10):
    """Fraction of held-out teacher decisions the policy reproduces greedily.

    Reported separately for card choice and placement -- a policy can look good
    on card choice alone simply by learning the teacher's no-op rate, so the
    two are never merged into one number.

    `card_match_decisions` is the one to judge by. Plain `card_match` counts
    every step including those where the affordability mask left only the
    no-op, and on those the "prediction" is forced -- both an untrained net and
    a perfect one score 100%. Measured here: the teacher no-ops on ~60% of
    steps, which puts the trivial baseline for plain card_match around 0.75 and
    makes it nearly useless for detecting learning. Restricting to steps with
    >=2 legal options is the same correction train.py's `decision` mask already
    applies to the actor loss, for the same reason.
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


# Deliberately NOT called from train.py or train_selfplay.py.
#
# Warm-starting the main agent by cloning the SCRIPTED teachers is of genuinely
# unclear value and could easily be negative: those bots are weak and narrow,
# and seeding the policy inside their behaviour is the opposite of the
# exploration the league is trying to buy. There is no measurement here saying
# it helps, so it does not run by default.
#
# What this module IS for right now is the pipeline: the dataset format, the
# recurrent trainer, and the evidence that both work end to end against a
# teacher whose behaviour is known. When perception/ can emit real human
# demonstrations in the DATASET_SCHEMA format, warm-starting from THOSE is the
# well-evidenced move (it is how AlphaStar started), and nothing in this file
# has to change to do it.
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collect", type=int, metavar="N_EPISODES",
                    help="generate demonstrations from the scripted teachers")
    ap.add_argument("--data", default="bc_demos.npz", help="dataset path")
    ap.add_argument("--train", action="store_true", help="fit a policy to --data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="bc_warmstart.pth")
    args = ap.parse_args()

    if args.collect:
        d = collect_demonstrations(n_episodes=args.collect)
        save_dataset(d, args.data)
        print(f"wrote {len(d['card'])} steps to {args.data}")
    if args.train:
        d = load_dataset(args.data)
        model, hist = train_bc(d, epochs=args.epochs)
        torch.save({"model": model.state_dict()}, args.out)
        print(f"wrote warm-start weights to {args.out}")
        print("  ", action_match_rate(model, d))
