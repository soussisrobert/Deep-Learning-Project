"""Segment-scoped UAV123@10fps acquisition (isolated adapter; uav.py stays frozen).

Identical to uav.acquire_uav_pilot except that each segment is extracted under
original/segments/<name>/, so official segments sharing one source folder (e.g. car1_1..car1_3,
which share boundary frames) each validate against their exact official frame set.
DOMAIN-UAV123 ACQUISITION-002 (2026-10-27) was produced by this exact logic while it briefly
lived inside uav.py as an opt-in flag; it was moved here to keep the Wave3-protected file unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile

from hift_research.evaluation.assets import import_dataset, sha256_file, write_json_exclusive
from .range_zip import AcquisitionError, HTTPRangeReader
from .uav import OFFICIAL_PAGE, SOURCES, _load_inventory_cache, extract_member, parse_sequence_config, validate_sequence


def acquire_uav_segments(output_root, provenance_dir, *, sequences, inventory=None, budget_bytes=4*1024**3,
                         annotation_coordinate_origin=None, selection_id, selection_rule):
    source = SOURCES["UAV123@10fps"]
    output_root, provenance_dir = Path(output_root).resolve(), Path(provenance_dir).resolve()
    if output_root.exists() or provenance_dir.exists():
        raise AcquisitionError("new output and provenance directories required; do not overwrite runs")
    if len(set(sequences)) != len(sequences) or not sequences:
        raise AcquisitionError("unique explicit sequence selection required")
    provenance_dir.mkdir(parents=True)
    write_json_exclusive(provenance_dir/"selection.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "sequences": list(sequences),
        "selected_before_tracking": True, "selection_id": selection_id, "selection_rule": selection_rule,
        "dataset": "UAV123@10fps", "network_budget_bytes": budget_bytes, "layout": "segment-scoped originals"})
    reader = HTTPRangeReader(source["url"], source["size"], budget_bytes=budget_bytes, cache_dir=provenance_dir/"ranges")
    reader.last_modified = source["last_modified_observed"]
    if inventory:
        _load_inventory_cache(reader, inventory)
    prefix = source["prefix"]
    records, validations = [], {}
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            names = {i.filename: i for i in infos}
            if len(names) != len(infos):
                raise AcquisitionError("duplicate ZIP member names")
            for suffix in ("configSeqs.m", "ReadMe.txt"):
                records.append(extract_member(archive, names[f"{prefix}/{suffix}"], output_root/"original"))
            configs = parse_sequence_config((output_root/"original"/prefix/"configSeqs.m").read_text(encoding="utf-8-sig"))
            ordered = sorted(infos, key=lambda i: i.header_offset)
            next_offset = {a.header_offset: b.header_offset for a, b in zip(ordered, ordered[1:])}
            for name in sequences:
                if name not in configs:
                    raise AcquisitionError(f"sequence not declared by official config: {name}")
                config = configs[name]
                folder = f"{prefix}/data_seq/{prefix}/{config['folder']}/"
                selected = [names[f"{folder}{index:0{config['digits']}d}.{config['extension']}"]
                            for index in range(config["start_frame"], config["end_frame"]+1)]
                annotation_name = f"{prefix}/anno/{prefix}/{name}.txt"
                original = output_root/"original"/"segments"/name
                records.append(extract_member(archive, names[annotation_name], original))
                start = min(i.header_offset for i in selected)
                end = next_offset.get(max(i.header_offset for i in selected))
                if end is None:
                    raise AcquisitionError("could not bound final selected ZIP member span")
                if end-start > sum(i.compress_size for i in selected)+1024*1024:
                    raise AcquisitionError("selected members not contiguous; refusing unexpectedly large span")
                reader.prefetch(start, end-1)
                for info in selected:
                    records.append(extract_member(archive, info, original))
                validation = validate_sequence(original/folder, original/annotation_name, config)
                validation["authoritative_config"] = config
                validations[name] = validation
                ready_dir = output_root/"prepared/data_seq"/name
                ready_dir.mkdir(parents=True)
                for frame in validation["frames"]:
                    shutil.copyfile(frame["path"], ready_dir/Path(frame["path"]).name)
                prepared_annotation = output_root/"prepared/anno"/f"{name}.txt"
                prepared_annotation.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original/annotation_name, prepared_annotation)
                print(f"Validated complete official sequence {name}: {validation['frame_count']} frames, {reader.network_bytes} new network bytes", flush=True)
        manifest = import_dataset(output_root/"prepared", "UAV123@10fps", allow_partial=True,
                                  invalid_policy="count_as_failure", hash_frames=True)
        manifest["annotation_coordinate_origin"] = annotation_coordinate_origin
        manifest["selected_segment_membership_verified_against_original_config"] = True
        manifest["acquisition_selection_id"] = selection_id
        write_json_exclusive(provenance_dir/"dataset_manifest.json", manifest)
        report = {"schema_version": 1, "kind": "authoritative_partial_dataset_acquisition", "layout": "segment-scoped originals",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "UAV123@10fps",
                  "official_page": OFFICIAL_PAGE, "source": source, "source_archive_full_sha256": None,
                  "selected_sequences": list(sequences), "network_bytes_this_run": reader.network_bytes,
                  "range_responses": reader.records, "members": records, "validation": validations,
                  "original_config_sha256": sha256_file(output_root/"original"/prefix/"configSeqs.m"),
                  "annotation_coordinate_origin": annotation_coordinate_origin,
                  "inference_ground_truth_boundary": "prepared dataset/validation manifest is EVALUATION ONLY",
                  "selection_id": selection_id}
        write_json_exclusive(provenance_dir/"acquisition.json", report)
        return report
    finally:
        write_json_exclusive(provenance_dir/"network_ledger.json", {
            "network_bytes_this_run": reader.network_bytes, "ranges": reader.records,
            "last_modified": reader.last_modified, "budget_bytes": budget_bytes})
