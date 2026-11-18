"""1024-candidate real-input preflight with immutable arrays/ledger, no SGD."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .pipeline import RealPairPipeline
from .schema import file_hash, canonical_hash
from .acquisition import write_exclusive_json


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    fraction = successes / total
    denom = 1 + z*z/total
    center = (fraction + z*z/(2*total))/denom
    half = z*np.sqrt(fraction*(1-fraction)/total + z*z/(4*total*total))/denom
    return [float(center-half), float(center+half)]


def run_preflight(manifest_path, output, *, count=1024):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays_path = output / "arrays"
    arrays_path.mkdir()
    pipeline = RealPairPipeline(manifest_path)
    if count != pipeline.manifest["policy"]["candidate_count"]:
        raise ValueError("candidate count differs from frozen preregistered policy")
    records, rejected = [], Counter()
    with (output / "candidate_ledger.jsonl").open("x", encoding="utf-8", newline="\n") as ledger:
        for index in range(count):
            record, arrays = pipeline.candidate(index)
            if arrays is not None:
                filename = arrays_path / f"{index:06d}.npz"
                with filename.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                record.update(array_archive_path=str(filename.resolve()), array_archive_sha256=file_hash(filename))
            else:
                rejected[record["reason"]] += 1
            records.append(record)
            ledger.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            ledger.flush()
            if (index+1) % 128 == 0:
                print(f"candidates={index+1} accepted={sum(r['status']=='accepted' for r in records)}", flush=True)
    accepted = [row for row in records if row["status"] == "accepted"]
    negative = sum(row["candidate_negative"] for row in records)
    accepted_negative = sum(row["candidate_negative"] for row in accepted)
    replay_indices = sorted(set([0, count-1] + np.random.RandomState(73).choice(count, min(16, count), replace=False).tolist()))
    replay = []
    # New instance discards all caches: input hashes must replay independently.
    fresh = RealPairPipeline(manifest_path)
    for index in replay_indices:
        actual, arrays = fresh.candidate(index)
        expected = {key: val for key, val in records[index].items() if key not in ("array_archive_path", "array_archive_sha256")}
        match = canonical_hash(actual) == canonical_hash(expected)
        replay.append({"candidate_index": index, "record_and_array_hashes_equal": match})
        if not match:
            raise ValueError("deterministic candidate replay mismatch")
    accepted_nonnegative = [row for row in accepted if not row["candidate_negative"]]
    maxima = [row["max_regression_weight"] for row in accepted_nonnegative]
    summary = {"schema": "hift-real-input-preflight-v1", "manifest_path": str(Path(manifest_path).resolve()),
               "manifest_sha256": file_hash(manifest_path), "policy_sha256": pipeline.manifest["policy_sha256"],
               "candidate_count": count, "candidate_negative_count": negative,
               "candidate_negative_fraction": negative/count, "candidate_negative_wilson95": wilson(negative, count),
               "candidate_negative_target_probability": .05, "accepted_count": len(accepted),
               "rejected_count": count-len(accepted), "rejection_reasons": dict(rejected), "replacements": 0,
               "accepted_negative_count": accepted_negative,
               "accepted_negative_fraction": accepted_negative/len(accepted) if accepted else None,
               "accepted_negative_wilson95": wilson(accepted_negative, len(accepted)),
               "positive_collapsed_support_fallback_count": sum(row["collapsed_support_fallback"] for row in accepted_nonnegative),
               "positive_candidates_with_weight64": sum(row["weight64_cells"] > 0 for row in accepted_nonnegative),
               "positive_candidates_with_weight_above1_5": sum(row["regression_weights_above_1_5"] > 0 for row in accepted_nonnegative),
               "positive_max_weight_quantiles": dict(zip(("min", "p50", "p95", "max"), map(float, np.quantile(maxima, [0, .5, .95, 1])))) if maxima else None,
               "template_image_candidate_counts": dict(Counter(str(row["template_image_id"]) for row in records)),
               "search_image_candidate_counts": dict(Counter(str(row["search_image_id"]) for row in records)),
               "same_frame_positive_candidates": sum(not row["candidate_negative"] and row["repeated_frame"] for row in records),
               "temporal_positive_intervals": {"0": sum(not row["candidate_negative"] for row in records)},
               "no_temporal_motion_evidence": True, "replay": replay,
               "candidate_ledger_sha256": file_hash(output / "candidate_ledger.jsonl"),
               "array_archive_count": len(accepted), "no_model_forward_sgd_gpu": True,
               "gates": {"G1": "bounded_real_input_contract_pass_if_replay_and_tests_pass_not_full_source_recipe",
                         "G5": "partial_still_images_alias_quarantine_and_distribution_shift_no_training_authority"}}
    write_exclusive_json(output / "summary.json", summary)
    return summary
