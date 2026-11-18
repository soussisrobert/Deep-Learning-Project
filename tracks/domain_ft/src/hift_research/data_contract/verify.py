"""Independent immutable artifact verification; no model forward/SGD/GPU."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys
import hashlib
import cv2
import numpy as np
from .acquisition import write_exclusive_json
from .pipeline import RealPairPipeline, array_digest
from .schema import file_hash, validate_training_manifest, canonical_hash
from .source_ops import ROOT
from .preflight import wilson


def quantiles(rows):
    return dict(zip(("min", "p50", "p95", "max"), map(float, np.quantile(rows, [0,.5,.95,1]))))


def verify(run_path):
    run = Path(run_path)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest_path = Path(summary["manifest_path"])
    manifest = validate_training_manifest(manifest_path)
    if file_hash(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("run frozen manifest differs")
    provenance = manifest["source_provenance"]
    if file_hash(provenance["annotation_path"]) != provenance["annotation_sha256"]:
        raise ValueError("complete source annotation bytes changed")
    acquisition = json.loads(Path(provenance["acquisition_path"]).read_text(encoding="utf-8"))
    transferred = sum(row["bytes"] for row in acquisition["network_ledger"])
    if transferred != acquisition["bytes_received_including_probe"] or transferred > 192*1024*1024:
        raise ValueError("training network ledger budget/count mismatch")
    ledger = run / "candidate_ledger.jsonl"
    if file_hash(ledger) != summary["candidate_ledger_sha256"]:
        raise ValueError("candidate ledger changed")
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if [row["candidate_index"] for row in records] != list(range(1024)):
        raise ValueError("candidate omission/reordering/replacement")
    pipeline = RealPairPipeline(manifest_path)
    archive_count = 0
    for record in records:
        if record["replacement"] is not None or record["replacement_count"] != 0:
            raise ValueError("hidden replacement")
        expected_prior = bool(np.random.RandomState(20261023+record["candidate_index"]).random_sample() < .05)
        if record["candidate_negative"] != expected_prior:
            raise ValueError("negative prior draw modified")
        replay, arrays = pipeline.candidate(record["candidate_index"])
        comparable = {key: val for key, val in record.items() if key not in ("array_archive_path", "array_archive_sha256")}
        if canonical_hash(replay) != canonical_hash(comparable):
            raise ValueError("all-candidate fresh replay mismatch")
        if record["status"] == "accepted":
            path = Path(record["array_archive_path"])
            if file_hash(path) != record["array_archive_sha256"]:
                raise ValueError("accepted array archive changed")
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(record["array_hashes"]):
                    raise ValueError("stored model-input keys differ")
                for key in saved.files:
                    value = saved[key]
                    if array_digest(value) != record["array_hashes"][key] or not np.isfinite(value).all():
                        raise ValueError("stored model-input array mismatch")
                    np.testing.assert_array_equal(value, arrays[key])
            archive_count += 1
        elif arrays is not None:
            raise ValueError("rejected candidate unexpectedly has inputs")
    accepted = [row for row in records if row["status"] == "accepted"]
    reserved = json.loads(Path(manifest["reservations_path"]).read_text(encoding="utf-8"))
    for source in reserved["source_manifests"]:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("reserved eval manifest changed")
    from .torch_adapter import CandidateDataset, reject_aware_collate
    import torch
    dataset = CandidateDataset(manifest_path)
    positive_index = accepted[0]["candidate_index"]
    rejected_index = next(row["candidate_index"] for row in records if row["status"] == "rejected")
    cpu_batch = reject_aware_collate([dataset[positive_index], dataset[rejected_index]])
    if cpu_batch["candidate_count"] != 2 or cpu_batch["replacement_count"] or len(cpu_batch["rejected_records"]) != 1:
        raise ValueError("real reject-aware collate mismatch")
    if not all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in cpu_batch["model_inputs"].values()):
        raise ValueError("collate did not produce CPU FP32 inputs")
    source_pairs = []
    for offset, first in enumerate(manifest["images"]):
        for second in manifest["images"][offset+1:]:
            distance = (int(first["phash64"],16)^int(second["phash64"],16)).bit_count()
            source_pairs.append({"image_ids":[first["image_id"],second["image_id"]], "phash_hamming":distance})
    source_near_pairs = [row for row in source_pairs if row["phash_hamming"] <= 4]
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    result = {"schema": "hift-real-data-independent-verification-v1", "all_checks_passed": True,
              "all_1024_candidates_fresh_replay_equal": True, "all_973_or_accepted_array_archives_verified": archive_count,
              "model_ready_cpu_collate": {"candidate_count": 2, "accepted": 1, "rejected": 1, "replacements": 0,
                                         "shape_dtype": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                                         for key,value in cpu_batch["model_inputs"].items()}},
              "training_network_bytes_including_probe": transferred, "network_budget_bytes": 192*1024*1024,
              "source_images": len(manifest["images"]), "source_instances": len(manifest["instances"]),
              "accepted_fraction_wilson95": wilson(len(accepted),len(records)),
              "source_to_source_phash_pair_count": len(source_pairs),
              "source_to_source_min_phash_distance": min(row["phash_hamming"] for row in source_pairs),
              "source_to_source_near_pairs_quarantine": source_near_pairs,
              "source_to_source_screen_status": "no_near_pairs" if not source_near_pairs else "posthoc_quarantine_pending_new_freeze",
              "reserved_eval_frames": len(reserved["frames"]), "reserved_family_lineages": reserved["reserved_family_lineages"],
              "known_eval_exact_or_near_image_quarantine": sum("reserved_eval" in row.get("reason", "") for row in manifest["quarantine"]),
              "positive_source_width_px": quantiles([row["source_object_size_px"][0] for row in accepted]),
              "positive_source_height_px": quantiles([row["source_object_size_px"][1] for row in accepted]),
              "positive_augmented_width_px": quantiles([row["augmented_object_size_px"][0] for row in accepted]),
              "positive_augmented_height_px": quantiles([row["augmented_object_size_px"][1] for row in accepted]),
              "regression_support_cells": quantiles([row["regression_support_cells"] for row in accepted]),
              "cls1_ignored_cells": quantiles([row["cls1_ignored_cells"] for row in accepted]),
              "cls2_positive_cells": quantiles([row["cls2_positive_cells"] for row in accepted]),
              "represented_template_annotation_count": len({row["template_annotation_id"] for row in records}),
              "represented_template_image_count": len({row["template_image_id"] for row in records}),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
              "critical_data_source_hashes": manifest["source_code_hashes"],
              "upstream_sha": subprocess.check_output(["git", "-C", "third_party/HiFT", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "upstream_status_porcelain": subprocess.check_output(["git", "-C", "third_party/HiFT", "status", "--porcelain"], cwd=ROOT, text=True),
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                              "opencv": cv2.__version__, "torch": torch.__version__, "opencv_threads": cv2.getNumThreads()},
              "checkpoint": None, "model_forward": False, "SGD": False, "GPU": False,
              "manifest_sha256": file_hash(manifest_path), "summary_sha256": file_hash(run/"summary.json")}
    write_exclusive_json(run / "independent_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    print(json.dumps({key: value for key,value in verify(args.run).items() if key in
                      ("all_checks_passed", "source_images", "source_instances", "reserved_eval_frames")}))
