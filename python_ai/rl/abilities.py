"""Champion / Hero ability actions: which engine slot, when legal, how sampled.

One place for the three things that go silently wrong (see
tests/test_champion_abilities.py):

* ENGINE SLOTS ARE DECK INDICES 1 AND 2. `GameManager` binds a Champion to
  `championSlots[slot]` by `deckConfig[slot]`, and a Champion is only legal at
  deck index 1 or 2. The network builds one head per Champion in the deck, so
  head k must drive `ability_engine_slots(deck)[k]` -- a lone Champion at index 2
  is head 0 driving engine slot 2.
* THE ACTIVATE ARM IS MASKED BY READINESS (`is_champion_ability_ready`, surfaced
  per slot in the env's info). An activation the engine will refuse must not
  carry probability, or its log-prob is noise in the PPO ratio.
* Each head is a 2-way Categorical (decline, activate). Log-probs and entropies
  are computed from the SAME masked logits at rollout and at update.
"""
import numpy as np
import torch
from torch.distributions import Categorical

import clash_royale_env as E

_NEG_INF = float("-inf")


def ability_engine_slots(deck):
    """[engine slot, ...] for each Champion/Hero in the deck, in slot order."""
    out = []
    for slot in (1, 2):
        if slot < len(deck):
            info = E.get_card_info(int(deck[slot]))
            if info.get("is_champion") or info.get("is_hero"):
                out.append(slot)
    return out


def ready_from_infos(infos, num_envs, engine_slots):
    """(num_envs, k) bool readiness for the next decision. Missing -> not ready,
    which is the value that can never fabricate an activation."""
    cols = []
    for slot in engine_slots:
        v = infos.get(f"champion_ability_slot{slot}_ready")
        cols.append(np.zeros(num_envs, dtype=bool) if v is None
                    else np.asarray(v, dtype=bool).reshape(num_envs))
    if not cols:
        return torch.zeros((num_envs, 0), dtype=torch.bool)
    return torch.as_tensor(np.stack(cols, axis=1))


def _masked(logits, ready_col):
    """Activate arm -inf where not ready; decline is always legal."""
    mask = torch.stack([torch.ones_like(ready_col), ready_col], dim=-1)
    return logits.masked_fill(~mask, _NEG_INF)


def log_prob_and_entropy(logits_list, ready, actions):
    """(sum over heads of log-probs, per-head entropies) for given actions.

    logits_list: [ (..., 2) ] one per head; ready/actions: (..., k).
    """
    total = None
    ents = []
    for k, logits in enumerate(logits_list):
        dist = Categorical(logits=_masked(logits, ready[..., k].bool()))
        lp = dist.log_prob(actions[..., k].long())
        total = lp if total is None else total + lp
        ents.append(dist.entropy())
    if total is None:
        shape = actions.shape[:-1]
        return torch.zeros(shape), torch.zeros(shape + (0,))
    return total, torch.stack(ents, dim=-1)


def sample(logits_list, ready):
    """(actions (N,k) long, log-prob (N,), entropy (N,k))."""
    acts = []
    for k, logits in enumerate(logits_list):
        acts.append(Categorical(logits=_masked(logits, ready[:, k].bool())).sample())
    n = ready.shape[0]
    actions = (torch.stack(acts, dim=1) if acts
               else torch.zeros((n, 0), dtype=torch.long))
    logprob, ent = log_prob_and_entropy(logits_list, ready, actions)
    return actions, logprob, ent


def action_dict(actions, engine_slots, num_envs):
    """The env's two action keys. Both are always present: AsyncVectorEnv's Dict
    space requires every key."""
    out = {"activate_ability_slot1": np.zeros(num_envs, dtype=np.int64),
           "activate_ability_slot2": np.zeros(num_envs, dtype=np.int64)}
    a = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
    for k, slot in enumerate(engine_slots):
        out[f"activate_ability_slot{slot}"] = a[:, k].astype(np.int64)
    return out
