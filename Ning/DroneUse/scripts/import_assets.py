"""Validate explicitly supplied local assets; never download or unpickle files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Direct checkout invocation works without an editable install or PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hift_research.evaluation.assets import (
    DATASET_COUNTS, import_checkpoint, import_dataset, read_json,
    sha256_file, validate_dataset_manifest, write_json_exclusive,
)
from hift_research.evaluation import ValidationError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checkpoint = commands.add_parser("checkpoint", help="Copy and hash a local checkpoint; does NOT validate model keys")
    checkpoint.add_argument("--source", required=True, type=Path)
    checkpoint.add_argument("--destination", required=True, type=Path)
    checkpoint.add_argument("--sha256", required=True)
    checkpoint.add_argument("--output", required=True, type=Path)
    dataset = commands.add_parser("dataset", help="Validate local frames/GT and create immutable manifest")
    dataset.add_argument("--root", required=True, type=Path)
    dataset.add_argument("--dataset", required=True, choices=sorted(DATASET_COUNTS))
    dataset.add_argument("--output", required=True, type=Path)
    dataset.add_argument("--allow-partial", action="store_true", help="Explicitly non-benchmark fixtures/subsets")
    dataset.add_argument("--invalid-policy", choices=["error", "count_as_failure"], default="error")
    dataset.add_argument("--hash-frames", action="store_true", help="Read all local image bytes for content identity; no download")
    dataset.add_argument("--coordinate-origin", type=int, choices=[0, 1], help="Required explicit source origin when exporting inference inputs")
    dataset.add_argument("--inference-dir", type=Path, help="Export GT-free per-sequence inputs; x/y converted to zero-origin")
    validate = commands.add_parser("validate-dataset", help="Revalidate manifest frame sizes and available hashes")
    validate.add_argument("manifest", type=Path)
    hash_command = commands.add_parser("hash", help="Hash one explicitly supplied local file, without deserialization")
    hash_command.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "hash":
            print(sha256_file(args.path))
        elif args.command == "validate-dataset":
            manifest = validate_dataset_manifest(read_json(args.manifest))
            print(f"Validated {manifest['dataset']}: {len(manifest['sequences'])} sequences")
        elif args.command == "checkpoint":
            if args.output.exists() or args.destination.exists():
                raise ValidationError("checkpoint/manifest destination already exists")
            record = import_checkpoint(args.source, args.destination, expected_sha256=args.sha256)
            write_json_exclusive(args.output, record)
            print(f"Checkpoint integrity verified: {record['sha256']}; model keys NOT validated")
        else:
            if args.output.exists():
                raise ValidationError("dataset manifest destination already exists")
            if args.inference_dir is not None and args.coordinate_origin is None:
                raise ValidationError("--inference-dir requires explicit --coordinate-origin; no origin guessed")
            manifest = import_dataset(args.root, args.dataset, allow_partial=args.allow_partial,
                                      invalid_policy=args.invalid_policy, hash_frames=args.hash_frames)
            manifest["annotation_coordinate_origin"] = args.coordinate_origin
            # Validate all intended output names before creating any of them.
            outputs = []
            if args.inference_dir is not None:
                for sequence in manifest["sequences"]:
                    target = args.inference_dir / (sequence["name"] + ".json")
                    if target.exists():
                        raise ValidationError(f"inference input already exists: {target}")
                    initial = list(sequence["init_box"])
                    initial[0] += 0.5 - args.coordinate_origin
                    initial[1] += 0.5 - args.coordinate_origin
                    outputs.append((target, {"name": sequence["name"], "dataset": args.dataset,
                        "frames": [str((args.root.resolve() / p).resolve()) for p in sequence["frame_paths"]],
                        "initial_bbox": initial, "coordinate_origin": 0,
                        "annotation_coordinate_origin": args.coordinate_origin,
                        "initialization_convention": "upstream_tools_test_rectangle_half_pixel",
                        "evaluation_ground_truth_included": False}))
            write_json_exclusive(args.output, manifest)
            for target, value in outputs:
                write_json_exclusive(target, value)
            print(f"Imported {args.dataset}: {len(manifest['sequences'])} sequences; official membership NOT verified")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Asset validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
