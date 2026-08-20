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


def test_max_candidates_is_greedy_plus_the_expansion_grid():
    """K <= 1 + k_cards * k_cells. The harness used to compute this inline in an
    f-string, which is where a mismatch with the padding width would hide."""
    assert SearchCfg(k_cards=3, k_cells=2).max_candidates == 7
    assert SearchCfg(k_cards=1, k_cells=1).max_candidates == 2


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
    assert imported == {"dataclasses"}, imported


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
    monkeypatch.chdir(tmp_path)
    net = MicroRoyaleNet(num_ability_slots=0)
    path = save_historical_snapshot(net, 4321, "pipeline2")
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
    monkeypatch.chdir(tmp_path)
    net = MicroRoyaleNet(num_ability_slots=0)
    first = save_historical_snapshot(net, 99999999, "pipeline1")
    time.sleep(0.02)
    second = save_historical_snapshot(net, 1, "pipeline1")
    found = discover_historical_checkpoints()
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
    monkeypatch.chdir(tmp_path)
    net = MicroRoyaleNet(num_ability_slots=0)
    save_historical_snapshot(net, 50_000, "pipeline2")          # too young
    save_historical_snapshot(net, 1_000, "pipeline2")           # old enough
    save_historical_snapshot(net, 50_000, "pipeline1")          # never gated
    eligible = discover_historical_checkpoints(
        current_episode=50_000 + MIN_OPPONENT_AGE_EPISODES // 2)
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
    monkeypatch.chdir(tmp_path)
    net = MicroRoyaleNet(num_ability_slots=0)
    path = save_stage_snapshot(net, STAGE_CHECKPOINT_DIR, stage=3,
                               episodes_completed=1234, teacher_stage=3,
                               reason="cleared stage 3 gate")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["curriculum_stage"] == 3
    assert payload["episodes_completed"] == 1234
    assert payload["teacher_stage"] == 3
    assert "cleared stage 3" in payload["reason"]
    assert "optimizer" not in payload, "stage snapshots are never resumed from"
