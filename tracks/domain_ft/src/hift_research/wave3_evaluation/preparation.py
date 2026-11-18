"""Fixed pre-inference panel, official attribute proof, GT-free native inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np

from hift_research import OFFICIAL_CHECKPOINT_SHA256, UPSTREAM_COMMIT
from hift_research.data_acquisition import HTTPRangeReader
from hift_research.data_acquisition.uav import (
    SOURCES, _load_inventory_cache, extract_member, parse_sequence_config,
)
from hift_research.evaluation.assets import (
    read_json, sha256_file, validate_dataset_manifest, write_json_exclusive,
)

PANEL = ("bike1", "boat7", "car16_1", "person22", "truck1", "building1",
         "bird1_1", "wakeboard9", "uav1_1", "uav2", "uav4", "uav7")
ATTRIBUTE_ORDER = ("SV", "ARC", "LR", "FM", "FOC", "POC", "OV", "BC", "IV", "VC", "CM", "SOB")
WAVE2_FOLDERS = {"uav6", "wakeboard7", "building5"}
CRITICAL_SOURCE = ("src/hift_research/inference.py", "src/hift_research/geometry.py",
                   "scripts/run_sequence.py", "src/hift_research/diagnostics/policy.py")


def panel_estimate(inventory, configs, sequences=PANEL):
    members = {r["name"]: r for r in inventory["members"]}
    offsets = sorted(r["header_offset"] for r in inventory["members"])
    next_offsets = dict(zip(offsets, offsets[1:]))
    seen, rows = set(), []
    for name in sequences:
        if name not in configs:
            raise ValueError(f"unknown official segment {name}")
        config = configs[name]
        folder = config["folder"]
        if folder in seen or folder in WAVE2_FOLDERS:
            raise ValueError("evaluation panel overlaps a source-folder lineage")
        seen.add(folder)
        paths = [f"UAV123_10fps/data_seq/UAV123_10fps/{folder}/{i:0{config['digits']}d}.{config['extension']}"
                 for i in range(config["start_frame"], config["end_frame"] + 1)]
        selected = [members[p] for p in paths]
        start, final = min(r["header_offset"] for r in selected), max(r["header_offset"] for r in selected)
        rows.append({"name": name, "source_folder_lineage": folder, "official_config": config,
                     "frame_count": len(paths), "compressed_image_bytes": sum(r["compressed_size"] for r in selected),
                     "bounded_contiguous_span_bytes": next_offsets[final] - start,
                     "original_archive_members": paths})
    return rows


def freeze_plan(root, output, inventory_path):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("new preregistration directory required")
    output.mkdir(parents=True)
    source = SOURCES["UAV123@10fps"]
    inventory = read_json(inventory_path)
    reader = HTTPRangeReader(source["url"], source["size"], budget_bytes=512 * 1024,
                             cache_dir=output / "official_metadata_ranges")
    reader.last_modified = source["last_modified_observed"]
    _load_inventory_cache(reader, inventory)
    # The official config/attribute/annotation prefix is tiny; no image is read here.
    first_image = min(r["header_offset"] for r in inventory["members"] if r["name"].endswith(".jpg"))
    reader.prefetch(0, first_image - 1)
    try:
        with zipfile.ZipFile(reader) as archive:
            config_record = extract_member(archive, archive.getinfo("UAV123_10fps/configSeqs.m"), output / "original")
            configs = parse_sequence_config(Path(config_record["path"]).read_text(encoding="utf-8-sig"))
            rows = panel_estimate(inventory, configs)
            attributes = {}
            for row in rows:
                member = f"UAV123_10fps/anno/UAV123_10fps/att/{row['name']}.txt"
                record = extract_member(archive, archive.getinfo(member), output / "original")
                values = np.loadtxt(record["path"], delimiter=",", dtype=int).reshape(-1)
                if values.shape != (12,) or not np.isin(values, [0, 1]).all():
                    raise ValueError("invalid official sequence attribute vector")
                attributes[row["name"]] = {**record, "flags": dict(zip(ATTRIBUTE_ORDER, values.tolist(), strict=True))}
        estimated = sum(r["bounded_contiguous_span_bytes"] for r in rows)
        frames = sum(r["frame_count"] for r in rows)
        if estimated > 256 * 1024 * 1024 or frames > 5000:
            raise ValueError("panel exceeds approved initial image/frame budget; seek lead approval")
        coverage = {flag: [n for n in PANEL if attributes[n]["flags"][flag]] for flag in ATTRIBUTE_ORDER}
        plan = {
            "schema_version": 1, "kind": "wave3_preinference_panel_preregistration",
            "selection_id": "WAVE3-UAV10-12-001", "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "UAV123@10fps", "source": source, "upstream_sha": UPSTREAM_COMMIT,
            "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256, "selected_before_new_inference": True,
            "sequences": rows, "official_attributes": attributes, "attribute_order": list(ATTRIBUTE_ORDER),
            "attribute_coverage": coverage, "selection_rule": "Fixed12 diverse official source-folder lineages; include bike1(1029frames) and uav1_1(519frames) for longer drift, bounded bytes, no model outcomes used",
            "excluded_wave2_source_folders": sorted(WAVE2_FOLDERS), "expected_frame_count": frames,
            "estimated_new_image_payload_bytes": estimated, "approved_image_budget_bytes": 256 * 1024 * 1024,
            "primary_metrics": ["success_auc_21point_strict_gt", "precision_at_20px"],
            "invalid_policy": "count_as_failure", "initialization": "raw_first_GT_plus_half_pixel_no_origin_shift",
            "primary_prediction_coordinate_origin": "annotation_native", "annotation_coordinate_origin": None,
            "physical_diagnostic_origin": 1, "physical_origin_status": "MATLAB_consistency_assumption_not_author_origin_proof",
            "seed": 0, "bootstrap": {"seed": 20261023, "replicates": 10000, "confidence": .95,
                "unit": "paired_sequence_cluster", "method": "paired_percentile_bootstrap", "minimum_stratum_clusters_for_interval": 3},
            "predeclared_strata": ["LR", "FM", "SV", "FOC", "ARC", "SOB", "long_sequence_500plus"],
            "configs": {name: {"path": str(root / f"configs/{name}.json"), "sha256": sha256_file(root / f"configs/{name}.json")}
                        for name in ("B0", "B1-score")},
            "critical_inference_source_sha256": {p: sha256_file(root / p) for p in CRITICAL_SOURCE},
            "official_config": config_record, "metadata_network_payload_bytes": reader.network_bytes,
            "metadata_range_responses": reader.records,
            "planned_run_root": str(root / "artifacts/wave3-evaluation/MATCHED-001"),
            "planned_dataset_manifest": str(root / "artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json"),
            "oracle_output_boundary": "lead-owned separate root; never oracle outputs in ordinary controller",
            "limitations": ["Deliberately chosen12-sequence panel, not full123 or random benchmark sample", "Original source-folder identities exclude known overlap; unverified cross-dataset/transcoded aliases are not disproven", "Attribute labels are sequence-level only and unavailable as inference features", "Paired bootstrap describes selected panel sensitivity, not population or retraining uncertainty"]}
        write_json_exclusive(output / "plan.json", plan)
        return plan
    finally:
        write_json_exclusive(output / "metadata_network_ledger.json", {"bytes": reader.network_bytes, "ranges": reader.records})


def write_native_inputs(dataset_path, plan_path, destination):
    dataset, plan = validate_dataset_manifest(read_json(dataset_path)), read_json(plan_path)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("new GT-free input directory required")
    by_name = {r["name"]: r for r in dataset["sequences"]}
    if set(by_name) != set(PANEL) or dataset["invalid_policy"] != "count_as_failure":
        raise ValueError("acquired cohort/policy differs from preregistration")
    if [r["name"] for r in plan["sequences"]] != list(PANEL):
        raise ValueError("preregistered cohort differs from fixed selection")
    destination.mkdir(parents=True)
    receipts = []
    for expected in plan["sequences"]:
        row = by_name[expected["name"]]
        if len(row["frame_paths"]) != expected["frame_count"]:
            raise ValueError("acquired frame count differs from preregistration")
        initial = np.asarray(row["init_box"], dtype=float).copy()
        initial[:2] += .5
        item = {"name": row["name"], "dataset": dataset["dataset"],
                "frames": [str((Path(dataset["root"]) / p).resolve()) for p in row["frame_paths"]],
                "initial_bbox": initial.tolist(), "coordinate_origin": 0,
                "annotation_coordinate_origin": None,
                "annotation_to_tracker_transform": "upstream_numeric_passthrough_plus_half_pixel_no_origin_shift",
                "initialization_convention": "upstream_tools_test_rectangle_half_pixel",
                "evaluation_ground_truth_included": False}
        path = destination / f"{row['name']}.json"
        write_json_exclusive(path, item)
        receipts.append({"name": row["name"], "path": str(path), "sha256": sha256_file(path),
                         "source_folder_lineage": expected["source_folder_lineage"],
                         "frame_sha256": row["frame_sha256"]})
    write_json_exclusive(destination / "input_receipt.json", {
        "kind": "wave3_GT_free_inputs", "dataset_manifest_sha256": sha256_file(dataset_path),
        "plan_sha256": sha256_file(plan_path), "inputs": receipts,
        "evaluation_ground_truth_included": False})
    return receipts
