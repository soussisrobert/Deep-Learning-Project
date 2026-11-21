"""Synthetic deterministic correctness fixtures; never benchmark measurements."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from hift_research.evaluation import ValidationError, aggregate_sequences, evaluate_sequence
from hift_research.evaluation.assets import (
    import_checkpoint, import_dataset, read_boxes, read_json, sha256_file,
    validate_dataset_manifest, write_json_exclusive,
)

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_success_is_20_over_21_not_one():
    score = evaluate_sequence([[0, 0, 10, 10]], [[0, 0, 10, 10]])
    assert score["success_auc"] == pytest.approx(20 / 21)
    assert score["success_curve"][-1] == 0
    assert score["precision_at_20px"] == 1
    assert score["frame_count"] == 1
    assert score["initialization_frame_included"] is True
    assert score["normalized_precision_at_0_20"] is None


def test_threshold_equality_and_exact_iou_half():
    score = evaluate_sequence([[0, 0, 3, 2]], [[1, 0, 3, 2]])
    assert score["success_auc"] == pytest.approx(10 / 21)
    assert score["success_curve"][10] == 0  # strictly greater than 0.5
    score = evaluate_sequence([[0, 0, 100, 100]] * 2,
                              [[20, 0, 100, 100], [20.00001, 0, 100, 100]])
    assert score["precision_at_20px"] == 0.5


def test_disjoint_and_negative_center_not_automatically_successful():
    score = evaluate_sequence([[-100, -100, 2, 2]], [[100, 100, 2, 2]])
    assert score["success_auc"] == 0
    assert score["precision_at_20px"] == 0


def test_normalized_precision_is_explicit_diagnostic_point_not_auc():
    score = evaluate_sequence([[0, 0, 10, 20]] * 2,
                              [[2, 0, 10, 20], [2.01, 0, 10, 20]], normalized_precision=True)
    assert score["normalized_precision_at_0_20"] == 0.5
    assert score["normalized_precision_applicability"] == "supplemental_diagnostic"
    assert len(score["normalized_precision_curve"]) == 51


def test_macro_not_frame_pooling_includes_init_rows():
    good = evaluate_sequence([[0, 0, 10, 10]], [[0, 0, 10, 10]])
    bad = evaluate_sequence([[0, 0, 10, 10]] * 9, [[100, 100, 10, 10]] * 9)
    result = aggregate_sequences({"short": good, "long": bad})
    assert result["precision_at_20px"] == 0.5
    assert result["success_auc"] == pytest.approx(10 / 21)
    assert result["frame_count"] == 10
    assert result["aggregation"] == "sequence_macro"
    with pytest.raises(ValidationError):
        aggregate_sequences({})
    with pytest.raises(ValidationError):
        aggregate_sequences({"a": good, "b": evaluate_sequence([[0, 0, 1, 1]], [[0, 0, 1, 1]], normalized_precision=True)})


@pytest.mark.parametrize("gt,pred", [([], []), ([[1, 2, 3]], [[1, 2, 3]]),
    ([[0, 0, 1, 1]], [[0, 0, 1, 1]] * 2), ([[0, 0, 0, 1]], [[0, 0, 1, 1]]),
    ([[0, 0, 1, 1]], [[0, 0, -1, 1]]), ([[np.nan, 0, 1, 1]], [[0, 0, 1, 1]]),
    ([[0, 0, 1, 1]], [[np.inf, 0, 1, 1]]), ([["bad", 0, 1, 1]], [[0, 0, 1, 1]])])
def test_invalid_shapes_counts_and_values_rejected(gt, pred):
    with pytest.raises(ValidationError):
        evaluate_sequence(gt, pred)


def test_failure_policy_retains_invalid_gt_and_prediction_denominators():
    score = evaluate_sequence([[0, 0, 10, 10], [np.nan, 0, 10, 10], [0, 0, 10, 10]],
                              [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 10]],
                              invalid_policy="count_as_failure", normalized_precision=True)
    assert score["precision_at_20px"] == pytest.approx(1 / 3)
    assert score["normalized_precision_at_0_20"] == pytest.approx(1 / 3)
    assert score["invalid_ground_truth_frames"] == 1
    assert score["invalid_prediction_frames"] == 1
    assert score["frame_count"] == 3
    with pytest.raises(ValidationError):
        evaluate_sequence([[0, 0, 1, 1]], [], invalid_policy="count_as_failure")
    with pytest.raises(ValidationError):
        evaluate_sequence([[0, 0, 1, 1]], [[0, 0, 1, 1]], invalid_policy="drop")


def test_numerical_extremes_rejected_not_nan_report():
    with pytest.raises(ValidationError):
        evaluate_sequence([[1e308, 1e308, 1e308, 1e308]], [[1e308, 1e308, 1e308, 1e308]])
    with pytest.raises(ValidationError):
        evaluate_sequence([[0, 0, 1e-300, 1e-300]], [[0, 0, 1e-300, 1e-300]])
    with pytest.raises(ValidationError, match="not representable"):
        evaluate_sequence([[1e100, 1e100, 1, 1]], [[1e100, 1e100, 1, 1]])


def test_matches_pinned_upstream_on_valid_positive_center_fixture():
    path = ROOT / "third_party/HiFT/toolkit/utils/statistics.py"
    if not path.exists():
        pytest.skip("pinned upstream submodule not initialized")
    spec = importlib.util.spec_from_file_location("pinned_statistics", path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng = np.random.default_rng(20261023)
    gt = np.column_stack([rng.uniform(50, 200, (32, 2)), rng.uniform(5, 80, (32, 2))])
    pred = gt.copy()
    pred[:, :2] += rng.uniform(-30, 30, (32, 2))
    actual = evaluate_sequence(gt, pred, normalized_precision=True)
    np.testing.assert_allclose(actual["success_curve"], upstream.success_overlap(gt, pred, len(gt)))
    center_gt = gt[:, :2] + (gt[:, 2:] - 1) / 2
    center_pred = pred[:, :2] + (pred[:, 2:] - 1) / 2
    np.testing.assert_allclose(actual["precision_curve"], upstream.success_error(center_gt, center_pred, np.arange(51), len(gt)))
    np.testing.assert_allclose(actual["normalized_precision_curve"], upstream.success_error(
        center_gt / gt[:, 2:], center_pred / gt[:, 2:], np.arange(51) / 100, len(gt)))


def raw_dtb(tmp_path, frames=2):
    root = tmp_path / "DTB70"
    images = root / "Toy/img"
    images.mkdir(parents=True)
    for index in range(frames):
        # Import validation verifies existence/size, NOT image decoding (explicit limitation).
        (images / f"{index + 1:05d}.jpg").write_bytes(b"synthetic-not-decoded-image")
    (root / "Toy/groundtruth_rect.txt").write_text("0,0,10,10\n" * frames, encoding="utf-8")
    return root


def test_dataset_import_partial_explicit_and_revalidation(tmp_path):
    root = raw_dtb(tmp_path)
    with pytest.raises(ValidationError, match="expected 70"):
        import_dataset(root, "DTB70")
    manifest = import_dataset(root, "DTB70", allow_partial=True, hash_frames=True)
    assert manifest["completeness"] == "partial"
    assert not manifest["official_membership_verified"]
    assert len(manifest["sequences"][0]["frame_sha256"]) == 2
    validate_dataset_manifest(manifest)
    frame = root / "Toy/img/00001.jpg"
    frame.write_bytes(b"S" * frame.stat().st_size)
    with pytest.raises(ValidationError, match="hash changed"):
        validate_dataset_manifest(manifest)


def test_dataset_count_mismatch_duplicate_missing_path_escape(tmp_path):
    root = raw_dtb(tmp_path)
    manifest = import_dataset(root, "DTB70", allow_partial=True)
    row = manifest["sequences"][0]
    row["frame_paths"][1] = row["frame_paths"][0]
    with pytest.raises(ValidationError, match="duplicate"):
        validate_dataset_manifest(manifest)
    row["frame_paths"][1] = "../outside.jpg"
    with pytest.raises(ValidationError, match="unsafe"):
        validate_dataset_manifest(manifest)
    row["frame_paths"][1] = "Toy/img/missing.jpg"
    with pytest.raises(ValidationError, match="missing"):
        validate_dataset_manifest(manifest)
    row["frame_paths"].pop()
    with pytest.raises(ValidationError, match="count mismatch"):
        validate_dataset_manifest(manifest)


def test_uav_requires_segment_metadata_and_checks_init(tmp_path):
    root = tmp_path / "UAV123"
    root.mkdir()
    with pytest.raises(ValidationError, match="exact segment"):
        import_dataset(root, "UAV123", allow_partial=True)
    (root / "f.jpg").write_bytes(b"fixture")
    meta = {"Toy": {"img_names": ["f.jpg"], "gt_rect": [[1, 1, 5, 5]], "init_rect": [2, 2, 5, 5]}}
    (root / "UAV123.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValidationError, match="exactly equal"):
        import_dataset(root, "UAV123", allow_partial=True)


@pytest.mark.parametrize("text", ["", "1,2,3\n", "1,2,3,4\n\n", "1,2,no,4\n", "1,2,3,4,5\n", "1,,2,3,4\n", "1,2,,4\n"])
def test_prediction_reader_never_drops_malformed_rows(tmp_path, text):
    path = tmp_path / "pred.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        read_boxes(path)


def test_hash_pinned_checkpoint_no_unpickling_no_overwrite(tmp_path):
    source = tmp_path / "untrusted.pth"
    source.write_bytes(b"not even pickle: hashing only")
    target = tmp_path / "assets/model.pth"
    with pytest.raises(ValidationError, match="does not match"):
        import_checkpoint(source, target, expected_sha256="0" * 64)
    record = import_checkpoint(source, target, expected_sha256=sha256_file(source))
    assert record["state_dict_validated"] is False
    assert sha256_file(target) == sha256_file(source)
    with pytest.raises(FileExistsError):
        import_checkpoint(source, target, expected_sha256=sha256_file(source))


def test_exclusive_json_and_duplicate_key_rejection(tmp_path):
    output = tmp_path / "manifest.json"
    write_json_exclusive(output, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json_exclusive(output, {"a": 2})
    output.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate JSON"):
        read_json(output)


def test_cli_end_to_end_gt_free_export_and_evaluation(tmp_path):
    import_main = runpy.run_path(str(ROOT / "scripts/import_assets.py"))["main"]
    evaluate_main = runpy.run_path(str(ROOT / "scripts/evaluate.py"))["main"]
    root = raw_dtb(tmp_path)
    manifest_path = tmp_path / "dataset.json"
    inference_dir = tmp_path / "inference"
    assert import_main(["dataset", "--root", str(root), "--dataset", "DTB70", "--allow-partial",
                        "--output", str(manifest_path), "--coordinate-origin", "0",
                        "--inference-dir", str(inference_dir)]) == 0
    inference = read_json(inference_dir / "Toy.json")
    assert inference["initial_bbox"] == [0.5, 0.5, 10, 10]
    assert "ground_truth" not in inference
    assert inference["evaluation_ground_truth_included"] is False
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("0.5,0.5,10,10\n0,0,10,10\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    arguments = ["--dataset-manifest", str(manifest_path), "--predictions", str(predictions),
                 "--sequence", "Toy", "--output", str(report_path), "--prediction-coordinate-origin", "zero"]
    assert evaluate_main(arguments) == 0
    report = read_json(report_path)
    assert report["dataset_completeness"] == "partial"
    assert report["sequences"]["Toy"]["success_auc"] < 20 / 21  # submitted init not overwritten
    assert report["aggregate"]["precision_at_20px"] == 1
    assert evaluate_main(arguments) == 2  # immutable output
    predictions.write_text("0,0,10,10\n", encoding="utf-8")
    arguments[arguments.index(str(report_path))] = str(tmp_path / "missing-frame.json")
    assert evaluate_main(arguments) == 2


@pytest.mark.parametrize("dataset", ["UAV123@10fps", "UAV20L"])
def test_raw_uav_variant_order_no_resampling(tmp_path, dataset):
    root = tmp_path / "variant"
    directory = root / "data_seq/Toy"
    directory.mkdir(parents=True)
    (root / "anno").mkdir()
    # Preserve original IDs; numeric sort is 1,4,10, not lexicographic 1,10,4.
    for name in ["10.jpg", "1.jpg", "4.jpg"]:
        (directory / name).write_bytes(b"fixture")
    (root / "anno/Toy.txt").write_text("0,0,10,10\n" * 3, encoding="utf-8")
    manifest = import_dataset(root, dataset, allow_partial=True)
    assert [Path(p).name for p in manifest["sequences"][0]["frame_paths"]] == ["1.jpg", "4.jpg", "10.jpg"]
    assert len(manifest["sequences"][0]["ground_truth"]) == 3


def test_invalid_gt_import_explicit_failure_policy_and_annotation_hash(tmp_path):
    root = raw_dtb(tmp_path)
    annotation = root / "Toy/groundtruth_rect.txt"
    annotation.write_text("0,0,10,10\nnan,0,0,10\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid GT"):
        import_dataset(root, "DTB70", allow_partial=True)
    manifest = import_dataset(root, "DTB70", allow_partial=True, invalid_policy="count_as_failure")
    assert manifest["sequences"][0]["ground_truth"][1][0] is None
    output = tmp_path / "manifest.json"
    write_json_exclusive(output, manifest)  # no nonstandard NaN JSON
    validate_dataset_manifest(read_json(output))
    annotation.write_text("0,0,10,10\n0,0,10,10\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="annotation hash changed"):
        validate_dataset_manifest(manifest)


def test_origin_one_export_and_origin_zero_evaluation(tmp_path):
    import_main = runpy.run_path(str(ROOT / "scripts/import_assets.py"))["main"]
    evaluate_main = runpy.run_path(str(ROOT / "scripts/evaluate.py"))["main"]
    root = raw_dtb(tmp_path, frames=1)
    (root / "Toy/groundtruth_rect.txt").write_text("1,1,10,10\n", encoding="utf-8")
    path = tmp_path / "dataset.json"
    inputs = tmp_path / "inputs"
    args = ["dataset", "--dataset", "DTB70", "--root", str(root), "--output", str(path),
            "--allow-partial", "--inference-dir", str(inputs)]
    assert import_main(args) == 2  # no origin silently guessed
    assert not path.exists()
    assert import_main(args + ["--coordinate-origin", "1"]) == 0
    assert read_json(inputs / "Toy.json")["initial_bbox"] == [0.5, 0.5, 10, 10]
    prediction = tmp_path / "Toy.csv"
    prediction.write_text("0,0,10,10\n", encoding="utf-8")
    output = tmp_path / "result.json"
    assert evaluate_main(["--dataset-manifest", str(path), "--predictions", str(prediction),
                          "--sequence", "Toy", "--prediction-coordinate-origin", "zero", "--output", str(output)]) == 0
    assert read_json(output)["aggregate"]["success_auc"] == pytest.approx(20 / 21)


def test_schema_is_parseable_and_covers_all_artifact_kinds():
    schema = read_json(ROOT / "experiments/schema.json")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert len(schema["oneOf"]) == 5
    assert schema["$defs"]["evaluation"]["properties"]["kind"]["const"] == "evaluation"
    assert schema["$defs"]["inferenceRun"]["properties"]["kind"]["const"] == "inference_run"


def test_evaluator_rejects_unexpected_sequence_files(tmp_path):
    main = runpy.run_path(str(ROOT / "scripts/evaluate.py"))["main"]
    manifest = import_dataset(raw_dtb(tmp_path), "DTB70", allow_partial=True)
    path = tmp_path / "dataset.json"
    write_json_exclusive(path, manifest)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for name in ["Toy", "Unexpected"]:
        (predictions / f"{name}.txt").write_text("0,0,10,10\n" * 2, encoding="utf-8")
    assert main(["--dataset-manifest", str(path), "--predictions", str(predictions),
                 "--output", str(tmp_path / "result.json")]) == 2
