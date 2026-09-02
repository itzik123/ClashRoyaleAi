"""SearchCfg and the checkpoint destinations -- two small pieces, both moved.

`SearchCfg` used to live in a 1,152-line experiment harness, and the checkpoint
directories in `train.py` where nothing but that trainer could reach them.
"""
import os

import pytest
import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.checkpointing import (
    HISTORICAL_CHECKPOINT_DIR, HISTORICAL_CHECKPOINT_INTERVAL_EPISODES,
    STAGE_CHECKPOINT_DIR, save_historical_snapshot, save_stage_snapshot,
)
from python_ai.search.config import SearchCfg


# ------------------------------------------------------------- SearchCfg --
def test_the_shipping_horizon_is_the_one_the_sweep_confirmed():
    from python_ai import shipping
    cfg = shipping.search_cfg()
    assert cfg.horizon == 12
    assert cfg.k_cards == 3 and cfg.k_cells == 2
    assert cfg.terminal_weight == 10.0


def test_max_candidates_is_greedy_plus_the_WIDENED_expansion_grid():
    """K <= 1 + k_cards * max(k_cells, WIDE_PROPOSAL_MAX_CELLS).

    This asserted `== 7` (greedy plus k_cards*k_cells) until 2026-09-03. That
    stopped being the bound when `propose_cells` began widening a flat
    placement head: the real search emitted up to 150 candidates while this
    still read 7, so the padding-width test it feeds was passing for a schema
    that could not hold a row. The harness used to compute the same expression
    inline in an f-string, which is exactly where such a mismatch hides.

    Written against the FORMULA rather than a literal, so raising the per-card
    cap moves the bound and the padding-width guard together instead of
    silently decoupling them again.
    """
    from python_ai.search.config import WIDE_PROPOSAL_MAX_CELLS

    per_card = max(2, WIDE_PROPOSAL_MAX_CELLS)
    assert SearchCfg(k_cards=3, k_cells=2).max_candidates == 1 + 3 * per_card
    assert SearchCfg(k_cards=1, k_cells=1).max_candidates ==         1 + max(1, WIDE_PROPOSAL_MAX_CELLS)


def test_the_padding_width_is_wide_enough_for_the_shipping_config():
    """`K_MAX` pads the recorded candidate set. If the shipping search could
    emit more candidates than the dataset has room for, labels would be silently
    truncated."""
    from python_ai import shipping
    from python_ai.trainers.expert_collect import K_MAX
    assert shipping.search_cfg().max_candidates <= K_MAX


def test_the_config_is_frozen_so_two_callers_cannot_diverge():
    """The whole reason it is a class and not an argparse namespace: labels are
    only expert labels for the configuration that produced them."""
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        SearchCfg().horizon = 99


def test_it_is_importable_without_dragging_in_a_trainer():
    """The coupling this move removed. Asserted on the module's own import
    graph rather than on sys.modules, which any earlier test could pollute."""
    import ast
    import pathlib

    import python_ai
    src = pathlib.Path(python_ai.PACKAGE_DIR, "search", "config.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    # The rule is about DEPENDENCY WEIGHT, not a literal one-module list: this
    # file must stay the cheapest import in the tree, which is why shipping.py
    # can read it without pulling a 1,100-line experiment harness. Stdlib is
    # free (already resident at interpreter start); a first-party import is not,
    # and is what this guard actually exists to refuse. Widened from
    # `== {"dataclasses"}` on 2026-09-03 when the widened-search constants moved
    # here and brought `os` for their env overrides.
    STDLIB_OK = {"dataclasses", "os", "math", "typing", "enum"}
    assert not any(m.startswith("python_ai") for m in imported), imported
    assert imported <= STDLIB_OK, imported


# -------------------------------------------------------- checkpointing --
def test_the_three_destinations_are_separate_directories():
    """Dropping stage snapshots into the PFSP pool would silently change which
    opponents self-play samples and how often -- the pool is read as a
    weakest-to-strongest ladder by save order."""
    assert HISTORICAL_CHECKPOINT_DIR != STAGE_CHECKPOINT_DIR


def test_the_snapshot_interval_and_the_age_gate_stay_coupled():
    """MIN_OPPONENT_AGE_EPISODES is kept at 3x the snapshot interval. Changing
    one alone silently changes which snapshots are eligible."""
    from python_ai.trainers.league import MIN_OPPONENT_AGE_EPISODES
    assert MIN_OPPONENT_AGE_EPISODES == 3 * HISTORICAL_CHECKPOINT_INTERVAL_EPISODES


def test_a_historical_snapshot_is_weights_only_and_names_its_pipeline(tmp_path,
                                                                     monkeypatch):
    """Weights only on purpose: these are never resumed from, only loaded as
    opponents, so carrying Adam's buffers would triple the file size for
    nothing. The pipeline tag is what the age gate matches on."""
    net = MicroRoyaleNet(num_ability_slots=0)
    path = save_historical_snapshot(net, 4321, "pipeline2",
                                    directory=str(tmp_path))
    assert "_pipeline2_ep00004321.pth" in os.path.basename(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == {"model"}


def test_snapshots_sort_oldest_first_by_save_order_not_by_filename(tmp_path,
                                                                  monkeypatch):
    """Both pipelines drop snapshots into one folder on two unrelated episode
    scales, so parsing episode numbers out of the filename would not give a
    meaningful weakest-to-strongest order. mtime does."""
    import time

    from python_ai.trainers.league import discover_historical_checkpoints
    net = MicroRoyaleNet(num_ability_slots=0)
    first = save_historical_snapshot(net, 99999999, "pipeline1",
                                     directory=str(tmp_path))
    time.sleep(0.02)
    second = save_historical_snapshot(net, 1, "pipeline1",
                                      directory=str(tmp_path))
    found = discover_historical_checkpoints(directory=str(tmp_path))
    assert [os.path.basename(p) for p in found] == [
        os.path.basename(first), os.path.basename(second)], (
        "a high episode number in the filename must not reorder the pool")


def test_pipeline_2s_own_recent_snapshots_are_age_gated_out(tmp_path,
                                                            monkeypatch):
    """Otherwise the pool fills with coin-flip mirrors of the current trainee
    and PFSP has nothing weak left to weight toward."""
    from python_ai.trainers.league import (
        MIN_OPPONENT_AGE_EPISODES, discover_historical_checkpoints,
    )
    net = MicroRoyaleNet(num_ability_slots=0)
    d = str(tmp_path)
    save_historical_snapshot(net, 50_000, "pipeline2", directory=d)  # too young
    save_historical_snapshot(net, 1_000, "pipeline2", directory=d)   # old enough
    save_historical_snapshot(net, 50_000, "pipeline1", directory=d)  # never gated
    eligible = discover_historical_checkpoints(
        current_episode=50_000 + MIN_OPPONENT_AGE_EPISODES // 2,
        directory=d)
    names = [os.path.basename(p) for p in eligible]
    assert not any("_pipeline2_ep00050000" in n for n in names)
    assert any("_pipeline2_ep00001000" in n for n in names)
    assert any("_pipeline1_ep00050000" in n for n in names), (
        "pipeline 1's snapshots live on a different episode scale and are "
        "always eligible")


def test_a_stage_snapshot_records_the_curriculum_it_was_taken_at(tmp_path,
                                                                 monkeypatch):
    """The metadata is what makes a later probe reproducible: it records which
    opponent strength this policy was actually trained against, so a comparison
    can replay it against a different rung and attribute the difference."""
    net = MicroRoyaleNet(num_ability_slots=0)
    path = save_stage_snapshot(net, str(tmp_path), stage=3,
                               episodes_completed=1234, teacher_stage=3,
                               reason="cleared stage 3 gate")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["curriculum_stage"] == 3
    assert payload["episodes_completed"] == 1234
    assert payload["teacher_stage"] == 3
    assert "cleared stage 3" in payload["reason"]
    assert "optimizer" not in payload, "stage snapshots are never resumed from"


# --- what a rollout that ENDED is worth -----------------------------------
#
# `search_action` scored a finished rollout as `reward * terminal_weight`, and
# the engine pays ~0 for a draw. So:
#
#     loss -> -1.0 * 10 = -10.0
#     draw ->  0.0 * 10 =   0.0
#
# while the reward the POLICY is trained on prices the two identically:
#
#     loss -> raw -1.0, no penalty        = -1.0
#     draw -> raw  0.0 - DRAW_PENALTY 1.0 = -1.0
#
# DRAW_PENALTY exists precisely to stop a timeout being the safe outcome, and
# the search was handing a drawn line a 10-point advantage over a lost one.
# That is not a small mis-weighting: it means the search is maximising a
# DIFFERENT objective from the one the policy is trained on, which is exactly
# the claim that makes it a policy-improvement operator at all.
#
# It only bites when a rollout can actually reach the clock -- the horizon is
# 4-12 decision steps against a ~360 s match -- so it is an ENDGAME defect, and
# the endgame is where running the clock out is tempting in the first place.

def test_a_drawn_rollout_is_priced_like_a_loss_not_like_a_neutral_outcome():
    from python_ai.rewards.weights import DRAW_PENALTY
    from python_ai.search.search import terminal_score
    assert terminal_score(0.0, weight=10.0) == terminal_score(-1.0, weight=10.0)
    assert terminal_score(0.0, weight=10.0) == -DRAW_PENALTY * 10.0


def test_a_win_still_dominates_and_a_loss_still_sinks():
    from python_ai.search.search import terminal_score
    assert terminal_score(1.0, weight=10.0) == 10.0
    assert terminal_score(-1.0, weight=10.0) == -10.0
    assert terminal_score(1.0, weight=10.0) > terminal_score(0.0, weight=10.0)


def test_the_terminal_weight_still_dominates_any_critic_value():
    """The reason the weight exists: a finished game must outrank anything the
    critic can say about an unfinished one."""
    from python_ai.search.search import terminal_score
    plausible_critic_range = 2.0
    for reward in (1.0, -1.0, 0.0):
        assert abs(terminal_score(reward, weight=10.0)) > plausible_critic_range


def test_search_action_uses_the_shared_terminal_score():
    """A static check: the scoring must not be re-inlined in the search loop,
    which is how it drifted from the reward function in the first place."""
    import inspect
    import re

    from python_ai.search import search
    body = inspect.getsource(search.search_action)
    assert "terminal_score" in body
    assert not re.search(r"reward\s*\*\s*cfg\.terminal_weight", body), (
        "terminal scoring re-inlined in search_action")
