"""CPU-only synthetic fixtures; no benchmark predictions or GPU side effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hift_research.evaluation import evaluate_sequence
from hift_research.evaluation.assets import sha256_file
from hift_research.wave3_evaluation.audit import (
    crop_summary, independent_sequence_metrics, paired_cluster_interval,
    verify_pretracking_contract,
)
from hift_research.wave3_evaluation.preparation import PANEL, panel_estimate, write_native_inputs


def test_independent_metrics_matches_primary_with_absence_and_boundaries():
    gt = [[0, 0, 3, 2], [0, 0, 100, 100], [np.nan] * 4, [-100, -100, 2, 2]]
    pred = [[1, 0, 3, 2], [20, 0, 100, 100], [0, 0, 10, 10], [100, 100, 2, 2]]
    expected = evaluate_sequence(gt, pred, invalid_policy="count_as_failure")
    actual = independent_sequence_metrics(gt, pred)
    for key in ("success_auc", "precision_at_20px"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-14)
    assert actual["frame_count"] == 4
    assert actual["absent_GT_frames_counted_as_failure"] == 1
    assert actual["zero_overlap_frames_excluding_init"] == 2
    assert actual["measured_zero_overlap_frames_valid_GT_only"] == 1
    assert actual["zero_overlap_episodes_excluding_init"] == 1


def test_independent_random_seeded_positive_fixture():
    rng = np.random.default_rng(1401)
    gt = np.column_stack([rng.uniform(-20, 100, (100, 2)), rng.uniform(2, 30, (100, 2))])
    pred = gt.copy()
    pred[:, :2] += rng.normal(size=(100, 2)) * 30
    actual, expected = independent_sequence_metrics(gt, pred), evaluate_sequence(gt, pred)
    assert actual["success_auc"] == pytest.approx(expected["success_auc"], abs=1e-14)
    assert actual["precision_at_20px"] == expected["precision_at_20px"]


@pytest.mark.parametrize("gt,pred", [([], []), ([[0, 0, 1, 1]], []),
    ([[0, 0, 1, 1]], [[0, 0, 0, 1]]), ([[0, 0, 1, 1]], [[np.nan, 0, 1, 1]])])
def test_independent_metrics_reject_missing_or_invalid_predictions(gt, pred):
    with pytest.raises(ValueError):
        independent_sequence_metrics(gt, pred)


def test_paired_bootstrap_resamples_whole_pairs_not_frames_or_separate_baselines():
    # Individual scores vary enormously, paired deltas do not.
    result = paired_cluster_interval([0, 100, 200], [1, 101, 201], replicates=500)
    assert result["mean_delta_B1_minus_B0"] == 1
    assert result["percentile_95_interval"] == [1, 1]
    assert result["clusters"] == 3
    assert result["unit"] == "paired_sequence_cluster"
    assert paired_cluster_interval([0, 0], [1, -1])["percentile_95_interval"] is None


def test_bootstrap_deterministic_and_preserves_unfavorable_deltas():
    a, b = [0, 0, 0, 0], [.5, -.8, -.3, .2]
    first = paired_cluster_interval(a, b, replicates=1000, seed=47)
    assert first == paired_cluster_interval(a, b, replicates=1000, seed=47)
    assert first["mean_delta_B1_minus_B0"] == pytest.approx(-.1)
    assert first["percentile_95_interval"][0] < 0
    with pytest.raises(ValueError):
        paired_cluster_interval([1], [1, 2])
    with pytest.raises(ValueError):
        paired_cluster_interval([1, np.nan], [1, 2])


def test_crop_padding_is_original_source_geometry_not_resized_detail():
    rows = [{"frame": 0}, {"frame": 1, "crop_geometry": {"source_size": 20, "model_size": 10,
             "origin_x": -5, "origin_y": -5}, "image_shape": [10, 10, 3]}]
    result = crop_summary(rows)
    assert result["crop_source_size_mean_px"] == 20
    assert result["resize_scale_mean"] == .5
    assert result["padding_fraction_mean"] == .75
    assert result["tracked_frames"] == 1
    rows[1]["frame"] = 3
    with pytest.raises(ValueError):
        crop_summary(rows)


def test_source_lineage_overlap_and_wave2_are_rejected():
    def member(folder, offset):
        return {"name": f"UAV123_10fps/data_seq/UAV123_10fps/{folder}/000001.jpg",
                "header_offset": offset, "compressed_size": 10}
    inventory = {"members": [member("source", 10), {"name": "tail", "header_offset": 30}]}
    config = {"a": {"folder": "source", "start_frame": 1, "end_frame": 1, "digits": 6, "extension": "jpg"}}
    assert panel_estimate(inventory, config, sequences=["a"])[0]["bounded_contiguous_span_bytes"] == 20
    config["b"] = config["a"]
    with pytest.raises(ValueError, match="lineage"):
        panel_estimate(inventory, config, sequences=["a", "b"])
    config["a"] = {**config["a"], "folder": "uav6"}
    with pytest.raises(ValueError, match="lineage"):
        panel_estimate(inventory, config, sequences=["a"])


def prepared_fixture(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    paths = ["configs/B0.json", "configs/B1-score.json", "scripts/run_sequence.py"]
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    sequences = []
    for name in PANEL:
        frame = root / "data" / name / "000001.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"synthetic fixture, file-size/hash check only")
        sequences.append({"name": name, "frame_paths": [f"{name}/000001.jpg"], "ground_truth": [[1, 2, 3, 4]],
                          "init_box": [1, 2, 3, 4], "frame_sha256": [sha256_file(frame)], "frame_sizes": [frame.stat().st_size]})
    dataset = {"schema_version": 1, "kind": "dataset", "root": str((root / "data").resolve()),
               "dataset": "UAV123@10fps", "protocol": "ope_all_frames_sequence_macro", "invalid_policy": "count_as_failure",
               "completeness": "partial", "sequences": sequences}
    plan = {"sequences": [{"name": n, "source_folder_lineage": n, "frame_count": 1} for n in PANEL],
            "invalid_policy": "count_as_failure", "configs": {b: {"sha256": sha256_file(root / f"configs/{b}.json")} for b in ("B0", "B1-score")},
            "critical_inference_source_sha256": {"scripts/run_sequence.py": sha256_file(root / "scripts/run_sequence.py")}}
    dataset_path, plan_path = tmp_path / "dataset.json", tmp_path / "plan.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return root, dataset_path, plan_path


def test_gt_free_native_input_contract_roundtrip_and_no_overwrite(tmp_path):
    root, dataset, plan = prepared_fixture(tmp_path)
    inputs = tmp_path / "inputs"
    receipts = write_native_inputs(dataset, plan, inputs)
    assert len(receipts) == 12
    first = json.loads((inputs / f"{PANEL[0]}.json").read_text())
    assert first["initial_bbox"] == [1.5, 2.5, 3, 4]
    assert first["annotation_coordinate_origin"] is None
    assert first["evaluation_ground_truth_included"] is False
    assert "ground_truth" not in first
    verify_pretracking_contract(root, plan, dataset, inputs)
    with pytest.raises(ValueError, match="new"):
        write_native_inputs(dataset, plan, inputs)
    (root / "configs/B0.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        verify_pretracking_contract(root, plan, dataset, inputs)


def test_gt_input_tampering_rejected_even_with_rehashed_receipt(tmp_path):
    root, dataset, plan = prepared_fixture(tmp_path)
    inputs = tmp_path / "inputs"
    write_native_inputs(dataset, plan, inputs)
    path = inputs / f"{PANEL[0]}.json"
    value = json.loads(path.read_text())
    value["ground_truth"] = [[1, 2, 3, 4]]
    path.write_text(json.dumps(value), encoding="utf-8")
    receipt_path = inputs / "input_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"][0]["sha256"] = sha256_file(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        verify_pretracking_contract(root, plan, dataset, inputs)


def test_preserves_old_acquisition_signature_defaults():
    from inspect import signature
    from hift_research.data_acquisition.uav import acquire_uav_pilot
    params = signature(acquire_uav_pilot).parameters
    assert params["selection_id"].default == "WAVE2-UAV10-PILOT-001"
    assert params["selection_rule"].default is None
