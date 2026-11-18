"""Strict CPU NumPy OPE evaluation of precomputed predictions; no inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

# Direct checkout invocation works without an editable install or PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hift_research.evaluation import (
    SUCCESS_THRESHOLDS, PRECISION_THRESHOLDS, NORMALIZED_THRESHOLDS,
    ValidationError, aggregate_sequences, evaluate_sequence,
)
from hift_research.evaluation.assets import (
    UPSTREAM_SHA, read_boxes, read_json, sha256_file,
    validate_dataset_manifest, write_json_exclusive,
)


def _git_provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=root, check=True,
                              capture_output=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                                  cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
        return {"revision": revision, "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "untracked_files": untracked, "limitation": "diff hash excludes untracked content; evaluator sources hashed separately"}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "limitation": "git revision unavailable (possibly unborn repository)"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Directory containing exactly name.txt/name.csv, or a single file with --sequence")
    parser.add_argument("--output", required=True, type=Path, help="New immutable JSON report (must not exist)")
    parser.add_argument("--sequence", help="Explicit sequence subset; output is labeled partial")
    parser.add_argument("--invalid-policy", choices=["error", "count_as_failure"], default="error")
    parser.add_argument("--normalized-precision", action="store_true", help="Supplemental diagnostic, not primary UAV/DTB metric")
    parser.add_argument("--prediction-coordinate-origin", choices=["annotation_native", "zero"], default="annotation_native")
    parser.add_argument("--run-manifest", type=Path, help="Optional inference run manifest to hash and link, not automatically trust")
    args = parser.parse_args(argv)
    start = time.perf_counter()
    try:
        if args.output.exists():
            raise ValidationError("output already exists; choose a new run path")
        manifest = validate_dataset_manifest(read_json(args.dataset_manifest))
        if args.invalid_policy != manifest["invalid_policy"]:
            raise ValidationError("CLI invalid policy must exactly match dataset import policy")
        selected = [r for r in manifest["sequences"] if args.sequence is None or r["name"] == args.sequence]
        if not selected:
            raise ValidationError("requested sequence does not exist")
        if args.predictions.is_file() and args.sequence is None:
            raise ValidationError("a single prediction file requires explicit --sequence")
        prediction_paths = {}
        if args.predictions.is_dir():
            for row in selected:
                candidates = [args.predictions / (row["name"] + suffix) for suffix in (".txt", ".csv")]
                found = [p for p in candidates if p.is_file()]
                if len(found) != 1:
                    raise ValidationError(f"{row['name']}: expected exactly one .txt or .csv prediction file")
                prediction_paths[row["name"]] = found[0]
            actual = {p.resolve() for p in args.predictions.iterdir() if p.suffix.lower() in {".txt", ".csv"}}
            expected = {p.resolve() for p in prediction_paths.values()}
            if args.sequence is None and actual != expected:
                raise ValidationError("unexpected prediction sequence files; no silent exclusion")
        elif args.predictions.is_file():
            prediction_paths[selected[0]["name"]] = args.predictions
        else:
            raise ValidationError("prediction path missing")
        offset = 0
        if args.prediction_coordinate_origin == "zero":
            offset = manifest.get("annotation_coordinate_origin")
            if offset not in {0, 1}:
                raise ValidationError("zero-origin predictions require declared annotation origin at dataset import")
        scores, input_hashes = {}, {}
        for row in selected:
            gt = np.asarray(row["ground_truth"], dtype=float).copy()
            gt[:, :2] -= offset
            path = prediction_paths[row["name"]]
            predictions = read_boxes(path)
            scores[row["name"]] = evaluate_sequence(gt, predictions, invalid_policy=args.invalid_policy,
                                                    normalized_precision=args.normalized_precision)
            input_hashes[row["name"]] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        root = Path(__file__).resolve().parents[1]
        source_paths = [Path(__file__).resolve()] + sorted((root / "src/hift_research/evaluation").glob("*.py"))
        report = {
            "schema_version": 1, "kind": "evaluation", "evidence": "measured_predictions_not_checkpoint_reproduction",
            "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": manifest["dataset"],
            "dataset_completeness": "partial" if args.sequence or manifest["completeness"] == "partial" else "expected_count_only",
            "official_membership_verified": manifest.get("official_membership_verified", False),
            "dataset_manifest": {"path": str(args.dataset_manifest.resolve()), "sha256": sha256_file(args.dataset_manifest)},
            "prediction_inputs": input_hashes, "prediction_coordinate_origin": args.prediction_coordinate_origin,
            "upstream_sha": UPSTREAM_SHA, "protocol": "ope_all_frames_sequence_macro",
            "success_thresholds": SUCCESS_THRESHOLDS.tolist(), "success_comparison": ">",
            "success_auc_convention": "arithmetic_mean_of_21_success_ordinates_not_trapezoid",
            "precision_thresholds": PRECISION_THRESHOLDS.tolist(), "precision_comparison": "<=",
            "normalized_thresholds": NORMALIZED_THRESHOLDS.tolist() if args.normalized_precision else None,
            "normalized_precision_applicability": "supplemental_diagnostic" if args.normalized_precision else "not_requested",
            "initialization_frame_included": True, "invalid_policy": args.invalid_policy,
            "aggregate": aggregate_sequences(scores), "sequences": scores,
            "software": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                         "machine": platform.machine(), "processor": platform.processor()},
            "device": "cpu", "seed": 0, "training_recipe": "not_applicable_no_training",
            "git": _git_provenance(),
            "evaluator_source_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in source_paths},
            "runtime_seconds": time.perf_counter() - start,
            "limitations": ["Input boxes evaluated; checkpoint identity and inference correctness are not certified.",
                            "Official sequence membership and frame decoding require separate validation.",
                            "Paths and sizes only are not full image identity unless imported with --hash-frames."],
        }
        if args.run_manifest:
            report["inference_run_manifest"] = {"path": str(args.run_manifest.resolve()), "sha256": sha256_file(args.run_manifest)}
        write_json_exclusive(args.output, report)
        print(f"sequence_macro success_auc={report['aggregate']['success_auc']:.9f} "
              f"precision@20px={report['aggregate']['precision_at_20px']:.9f}; {report['dataset_completeness']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
