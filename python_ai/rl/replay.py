"""Recording a demo replay and stamping the agent's internals onto it."""
import json

import numpy as np
import torch
from torch.distributions import Categorical

from python_ai.models.policy_io import LSTM_HIDDEN

#: Ticks per decision in a recorded replay: the training rollout's
#: `skip_frames`.
REPLAY_SKIP_FRAMES = 10

def annotate_replay_with_agent_info(filepath, decisions, skip_frames):
    """Merge per-decision agent internals (state value, chosen action) into a
    saved replay JSON, one skip_frames-wide window per decision, so the viewer
    can show them without the model.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    card_names = data.get("cardNames", {})
    for i, tick in enumerate(data["ticks"]):
        decision = decisions[min(i // skip_frames, len(decisions) - 1)]
        tick["stateValue"] = decision["stateValue"]
        tick["actionCardId"] = decision["actionCardId"]
        tick["actionCardName"] = card_names.get(str(decision["actionCardId"]), "No-op")
        tick["actionX"] = decision["actionX"]
        tick["actionY"] = decision["actionY"]

    with open(filepath, "w") as f:
        json.dump(data, f)


def record_greedy_replay(net, env, device, path, skip_frames=REPLAY_SKIP_FRAMES):
    """Play one episode with `env`, save its log to `path`, and annotate it.

    Sampled, not argmax: it shows the policy that is actually training.
    """
    obs, _ = env.reset()
    hx = torch.zeros(1, LSTM_HIDDEN).to(device)
    cx = torch.zeros(1, LSTM_HIDDEN).to(device)
    done = False
    decisions = []
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = net.affordability_mask(obs_t)
            features, card_embeds, spatial = net.extract_features(obs_t)
            logits, _, _, value, (hx, cx) = net.step_lstm_and_card(
                features, (hx, cx), mask)
            idx = Categorical(logits=logits).sample()
            place_logits = net.placement_given_card(
                hx, card_embeds, idx, obs_t, spatial)
            cell = Categorical(logits=place_logits).sample()
            x, y = net.cell_to_xy(cell)
        card_idx = int(idx.item())
        hand = env.game.get_hand()
        card_id = hand[card_idx] if card_idx < len(hand) else -1
        action = {
            "card_index": np.array([card_idx]),
            "target_x": np.array([x.item()]),
            "target_y": np.array([y.item()]),
            "activate_ability_slot1": np.array([0]),
            "activate_ability_slot2": np.array([0]),
        }
        decisions.append({
            "stateValue": value.item(),
            "actionCardId": card_id,
            "actionX": float(action["target_x"][0]),
            "actionY": float(action["target_y"][0]),
        })
        obs, _, terminated, truncated, _ = env.step(action, skip_frames=skip_frames)
        done = terminated or truncated
    env.game.save_log(path)
    annotate_replay_with_agent_info(path, decisions, skip_frames)
    return path
