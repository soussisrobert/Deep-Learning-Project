"""Export GT-free tracker inputs for every sequence of a validated dataset manifest.

Same per-sequence JSON as the Wave3 exporter (wave3_evaluation/preparation.py): ordered frames and the
first-frame box with the pinned upstream +0.5 half-pixel convention; no later ground truth.

    python scripts/export_inputs.py --dataset-manifest <manifest.json> --output <new dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from hift_research.evaluation.assets import read_json, sha256_file, validate_dataset_manifest, write_json_exclusive  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = validate_dataset_manifest(read_json(args.dataset_manifest))
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    for row in dataset["sequences"]:
        initial = np.asarray(row["init_box"], dtype=float).copy()
        initial[:2] += .5
        item = {"name": row["name"], "dataset": dataset["dataset"],
                "frames": [str((Path(dataset["root"])/p).resolve()) for p in row["frame_paths"]],
                "initial_bbox": initial.tolist(), "coordinate_origin": 0, "annotation_coordinate_origin": None,
                "annotation_to_tracker_transform": "upstream_numeric_passthrough_plus_half_pixel_no_origin_shift",
                "initialization_convention": "upstream_tools_test_rectangle_half_pixel",
                "evaluation_ground_truth_included": False}
        path = destination/f"{row['name']}.json"
        write_json_exclusive(path, item)
        receipts.append({"name": row["name"], "path": str(path), "sha256": sha256_file(path), "frames": len(item["frames"])})
    write_json_exclusive(destination/"input_receipt.json", {"dataset_manifest": str(args.dataset_manifest.resolve()),
                                                           "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
                                                           "sequences": receipts})
    print(f"exported {len(receipts)} GT-free inputs, {sum(r['frames'] for r in receipts)} frames")


if __name__ == "__main__":
    main()
