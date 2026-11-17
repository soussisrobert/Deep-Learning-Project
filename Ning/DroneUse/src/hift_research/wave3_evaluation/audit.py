"""Independent CPU mathematics, matched-identity checks, paired sequence uncertainty.

GT joins here are retrospective only. No score, crop, initialization, or inference
controller is changed; unfavorable sequence deltas/failures remain in all means.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from hift_research.diagnostics.matched import compare_sequence
from hift_research.evaluation.assets import read_boxes, read_json, sha256_file, validate_dataset_manifest, write_json_exclusive
from .preparation import PANEL


def independent_sequence_metrics(ground_truth, predictions):
    gt, pred = np.asarray(ground_truth, dtype=float), np.asarray(predictions, dtype=float)
    if gt.ndim != 2 or gt.shape[1] != 4 or gt.shape != pred.shape or not len(gt):
        raise ValueError("complete aligned nonempty xywh trajectories required")
    if not np.isfinite(pred).all() or np.any(pred[:, 2:] <= 0):
        raise ValueError("ordinary predictions must be finite positive-size boxes")
    overlaps, errors, unavailable = [], [], []
    for truth, prediction in zip(gt, pred, strict=True):
        if not np.isfinite(truth).all() or min(truth[2:]) <= 0:
            overlaps.append(-1.)
            errors.append(None)
            unavailable.append(True)
            continue
        x, y, w, h = map(float, truth)
        px, py, pw, ph = map(float, prediction)
        iw = max(0., min(x + w, px + pw) - max(x, px))
        ih = max(0., min(y + h, py + ph) - max(y, py))
        intersection = iw * ih
        union = w * h + pw * ph - intersection
        if not math.isfinite(union) or union <= 0:
            raise ValueError("unrepresentable box arithmetic")
        overlaps.append(intersection / union)
        gt_center = (x + (w - 1) / 2, y + (h - 1) / 2)
        pred_center = (px + (pw - 1) / 2, py + (ph - 1) / 2)
        errors.append(math.hypot(gt_center[0] - pred_center[0], gt_center[1] - pred_center[1]))
        unavailable.append(False)
    thresholds = np.arange(0, 1.05, .05)
    curve = [sum(value > threshold for value in overlaps) / len(gt) for threshold in thresholds]
    tracked = list(range(1, len(gt)))
    failures = [overlaps[i] <= 0 for i in tracked]
    return {"frame_count": len(gt), "success_auc": float(sum(curve) / len(curve)),
            "precision_at_20px": sum(e is not None and e <= 20 for e in errors) / len(gt),
            "zero_overlap_frames_excluding_init": sum(failures),
            "measured_zero_overlap_frames_valid_GT_only": sum(overlaps[i] <= 0 and not unavailable[i] for i in tracked),
            "absent_GT_frames_counted_as_failure": sum(unavailable),
            "zero_overlap_episodes_excluding_init": sum(v and (j == 0 or not failures[j - 1]) for j, v in enumerate(failures)),
            "iou_below_0_1_frames_excluding_init": sum(overlaps[i] < .1 for i in tracked),
            "center_error_over_20px_frames_excluding_init": sum(errors[i] is None or errors[i] > 20 for i in tracked),
            "invalid_policy": "count_as_failure", "initialization_frame_included": True}


def paired_cluster_interval(b0, b1, *, replicates=10000, seed=20261023, minimum_clusters=3):
    a, b = np.asarray(b0, dtype=float), np.asarray(b1, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite paired sequence scores required")
    if replicates < 1:
        raise ValueError("positive bootstrap replicate count required")
    delta = b - a
    result = {"clusters": len(a), "unit": "paired_sequence_cluster", "mean_delta_B1_minus_B0": float(delta.mean()),
              "seed": seed, "replicates": replicates, "percentile_95_interval": None,
              "interpretation": "selected_panel_sensitivity_not_random_population_or_training_uncertainty"}
    if len(a) >= minimum_clusters:
        rng = np.random.default_rng(seed)
        sample = delta[rng.integers(0, len(a), size=(replicates, len(a)))].mean(axis=1)
        result["percentile_95_interval"] = np.quantile(sample, [.025, .975]).tolist()
    else:
        result["interval_status"] = "descriptive_only_fewer_than_three_sequences"
    return result


def verify_pretracking_contract(root, plan_path, dataset_path, input_dir):
    root, input_dir = Path(root), Path(input_dir)
    plan = read_json(plan_path)
    dataset = validate_dataset_manifest(read_json(dataset_path))
    expected = {r["name"]: r for r in plan["sequences"]}
    if set(expected) != set(PANEL) or {r["name"] for r in dataset["sequences"]} != set(PANEL):
        raise ValueError("Wave3 fixed12 panel required; never rerun/relabel Wave2")
    if dataset["invalid_policy"] != "count_as_failure" or plan["invalid_policy"] != "count_as_failure":
        raise ValueError("explicit failure-denominator protocol required")
    for baseline, record in plan["configs"].items():
        if sha256_file(root / f"configs/{baseline}.json") != record["sha256"]:
            raise ValueError("baseline configuration changed after preregistration")
    for relative, digest in plan["critical_inference_source_sha256"].items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"ordinary inference source changed after preregistration: {relative}")
    receipts = read_json(input_dir / "input_receipt.json")
    if receipts["dataset_manifest_sha256"] != sha256_file(dataset_path) or receipts["plan_sha256"] != sha256_file(plan_path):
        raise ValueError("input receipt no longer matches dataset/preregistration")
    input_hashes = {r["name"]: r["sha256"] for r in receipts["inputs"]}
    for row in dataset["sequences"]:
        if len(row["frame_paths"]) != expected[row["name"]]["frame_count"]:
            raise ValueError("official segment frame count changed")
        path = input_dir / f"{row['name']}.json"
        if sha256_file(path) != input_hashes[row["name"]]:
            raise ValueError("GT-free input changed")
        value = read_json(path)
        forbidden = {"ground_truth", "gt_rect", "gt_boxes", "annotations"} & value.keys()
        initial = np.asarray(row["init_box"], dtype=float).copy()
        initial[:2] += .5
        if forbidden or value.get("evaluation_ground_truth_included") is not False:
            raise ValueError("GT leakage in ordinary input")
        if not np.array_equal(initial, value["initial_bbox"]):
            raise ValueError("initialization differs from literal upstream native+.5")
        if value["frames"] != [str((Path(dataset["root"]) / p).resolve()) for p in row["frame_paths"]]:
            raise ValueError("ordinary frames/order differ from hashed evaluation dataset")
    return plan, dataset


def crop_summary(logs):
    """Geometry/padding from ordinary telemetry only, with no GT-derived events."""
    sizes, scales, padding = [], [], []
    for index, log in enumerate(logs):
        if log["frame"] != index:
            raise ValueError("diagnostics not complete ordered frames")
        if index == 0:
            continue
        crop = log["crop_geometry"]
        size = float(crop["source_size"])
        if size <= 0 or crop["model_size"] <= 0:
            raise ValueError("invalid crop geometry")
        ih, iw = log["image_shape"][:2]
        visible_w = max(0., min(iw, crop["origin_x"] + size) - max(0., crop["origin_x"]))
        visible_h = max(0., min(ih, crop["origin_y"] + size) - max(0., crop["origin_y"]))
        sizes.append(size)
        scales.append(crop["model_size"] / size)
        padding.append(1 - visible_w * visible_h / size ** 2)
    if not sizes:
        return {"tracked_frames": 0}
    return {"tracked_frames": len(sizes), "crop_source_size_mean_px": float(np.mean(sizes)),
            "crop_source_size_median_px": float(np.median(sizes)), "crop_source_size_max_px": max(sizes),
            "resize_scale_mean": float(np.mean(scales)), "padding_fraction_mean": float(np.mean(padding)),
            "fully_outside_image_crops": sum(p >= 1 - 1e-12 for p in padding),
            "interpretation": "ordinary_geometry_only_resizing_adds_no_source_detail"}


def _logs(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def audit_panel(root, plan_path, dataset_path, runs, output):
    plan = read_json(plan_path)
    dataset = validate_dataset_manifest(read_json(dataset_path))
    runs, output = Path(runs).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("immutable new audit output required")
    by_name = {r["name"]: r for r in dataset["sequences"]}
    if set(by_name) != set(PANEL):
        raise ValueError("audit requires every fixed new sequence")
    recorded_plan = read_json(runs / "subset_plan.json")
    if recorded_plan["dataset_manifest"]["sha256"] != sha256_file(dataset_path):
        raise ValueError("driver did not use exact immutable dataset manifest")
    cohort = []
    for selected in plan["sequences"]:
        name, row = selected["name"], by_name[selected["name"]]
        result, _, physical = compare_sequence(row, runs / name / "B0", runs / name / "B1-score",
                                               annotation_origin=plan["physical_diagnostic_origin"], invalid_policy="count_as_failure")
        item = {"sequence": name, "source_folder_lineage": selected["source_folder_lineage"],
                "frames": len(row["frame_paths"]), "official_attributes": plan["official_attributes"][name]["flags"],
                "paired_common_audit": result, "baselines": {}}
        for index, baseline in enumerate(("B0", "B1-score")):
            directory = runs / name / baseline
            manifest = read_json(directory / "manifest.json")
            if manifest["config_sha256"] != plan["configs"][baseline]["sha256"] or manifest["seed"] != plan["seed"]:
                raise ValueError("run config/seed differs from frozen plan")
            for relative, digest in plan["critical_inference_source_sha256"].items():
                if manifest["source"]["source_files_sha256"].get(relative) != digest:
                    raise ValueError("run critical source differs from preregistration")
            pred = read_boxes(directory / f"{name}.csv")
            expected_init = np.asarray(row["init_box"], dtype=float).copy()
            expected_init[:2] += .5
            if not np.allclose(pred[0], expected_init, rtol=0, atol=1e-7):
                raise ValueError("saved submitted init not literal upstream native+.5")
            metrics = independent_sequence_metrics(row["ground_truth"], pred)
            saved = read_json(directory / "evaluation.json")["sequences"][name]
            for metric in ("success_auc", "precision_at_20px"):
                if not np.isclose(metrics[metric], saved[metric], rtol=0, atol=1e-12):
                    raise ValueError(f"independent metric disagreement: {name}/{baseline}/{metric}")
            common_failures = result[baseline]["failures_excluding_init"]
            for independent_key, common_key in (
                ("zero_overlap_frames_excluding_init", "zero_overlap_failure_frames"),
                ("iou_below_0_1_frames_excluding_init", "iou_below_0_1_frames"),
                ("center_error_over_20px_frames_excluding_init", "center_error_over_20px_frames")):
                if metrics[independent_key] != common_failures[common_key]:
                    raise ValueError(f"independent failure-count disagreement: {name}/{baseline}/{independent_key}")
            logs = _logs(directory / "diagnostics.jsonl")
            if len(logs) != len(pred):
                raise ValueError("diagnostic frame count mismatch")
            available = [r for r in physical[index][1:] if not r["ground_truth_unavailable"]]
            item["baselines"][baseline] = {"metrics": metrics, "ordinary_crop_stats": crop_summary(logs),
                "retrospective_physical_containment_fraction_mean": float(np.mean([r["crop_containment_fraction"] for r in available])) if available else None,
                "physical_origin_assumption": 1, "runtime_descriptive_only": manifest["metrics"],
                "manifest_sha256": sha256_file(directory / "manifest.json"),
                "prediction_sha256": sha256_file(directory / f"{name}.csv"),
                "diagnostics_sha256": sha256_file(directory / "diagnostics.jsonl"),
                "policy_grids_sha256": sha256_file(directory / "policy_grids.npz")}
        cohort.append(item)
    macro, deltas, intervals = {}, {}, {}
    for metric in ("success_auc", "precision_at_20px"):
        a = [r["baselines"]["B0"]["metrics"][metric] for r in cohort]
        b = [r["baselines"]["B1-score"]["metrics"][metric] for r in cohort]
        for baseline, values in (("B0", a), ("B1-score", b)):
            macro.setdefault(baseline, {})[metric] = float(np.mean(values))
        deltas[metric] = float(np.mean(np.asarray(b) - a))
        intervals[metric] = paired_cluster_interval(a, b, replicates=plan["bootstrap"]["replicates"], seed=plan["bootstrap"]["seed"])
    driver_report = read_json(runs / "comparison.json")
    driver_names = [r["sequence"] for r in driver_report["sequence_results"]]
    if driver_report["status"] != "completed" or len(driver_names) != len(PANEL) or set(driver_names) != set(PANEL):
        raise ValueError("shared driver report does not contain exactly the complete fixed cohort")
    for baseline in ("B0", "B1-score"):
        for metric in ("success_auc", "precision_at_20px"):
            if not np.isclose(driver_report["aggregate"][baseline][metric], macro[baseline][metric], rtol=0, atol=1e-12):
                raise ValueError("driver aggregate disagrees with independent equal-sequence macro")
    strata = {}
    for flag in plan["predeclared_strata"]:
        rows = [r for r in cohort if (r["frames"] >= 500 if flag == "long_sequence_500plus" else r["official_attributes"][flag])]
        strata[flag] = {"sequences": [r["sequence"] for r in rows], "sequence_count": len(rows), "metrics": {}}
        for metric in ("success_auc", "precision_at_20px"):
            if rows:
                a = [r["baselines"]["B0"]["metrics"][metric] for r in rows]
                b = [r["baselines"]["B1-score"]["metrics"][metric] for r in rows]
                strata[flag]["metrics"][metric] = {"B0_macro": float(np.mean(a)), "B1_macro": float(np.mean(b)),
                    **paired_cluster_interval(a, b, replicates=plan["bootstrap"]["replicates"], seed=plan["bootstrap"]["seed"])}
    report = {"schema_version": 1, "kind": "wave3_independent_ordinary_paired_CPU_audit", "gate": "G3",
              "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "plan": {"path": str(Path(plan_path).resolve()), "sha256": sha256_file(plan_path)},
              "dataset_manifest": {"path": str(Path(dataset_path).resolve()), "sha256": sha256_file(dataset_path)},
              "runs": str(runs), "sequence_count": len(cohort), "frame_count": sum(r["frames"] for r in cohort),
              "aggregation": "equal_sequence_macro_not_frame_pool", "macro": macro,
              "macro_delta_B1_minus_B0": deltas, "paired_cluster_uncertainty": intervals,
              "condition_strata": strata, "sequences": cohort,
              "limitations": ["Selected12 clusters, not a random/full benchmark sample; percentile intervals are panel sensitivity only", "Overlapping sequence-level attributes are not independent causal strata; no multiplicity-adjusted hypothesis claim", "No frame-IID inference; all paired sequence failures and adverse deltas retained", "Physical GT-origin1 is an offline MATLAB-consistency assumption, not primary coordinate conversion", "Latency is descriptive serialized desktop GPU telemetry, not a speed advantage test", "No training, no score retuning, no oracle outputs in ordinary inference"]}
    write_json_exclusive(output, report)
    return report
