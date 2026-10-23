"""Freeze/read authentic COCO TRAIN arrays; CPU only, never model/SGD/GPU."""
import argparse
import json
from pathlib import Path
import cv2
from hift_research.data_contract.freeze import freeze_training
from hift_research.data_contract.preflight import run_preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition")
    parser.add_argument("--eval-manifest", action="append", default=[])
    parser.add_argument("--reserved-lineages", default="")
    parser.add_argument("--freeze-output")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    manifest = args.manifest
    if args.acquisition:
        if len(args.eval_manifest) < 2 or not args.freeze_output or not args.reserved_lineages:
            parser.error("freeze requires Wave2+Wave3 eval manifests, all reserved lineages, and exclusive output")
        manifest = freeze_training(args.acquisition, args.eval_manifest,
                                   args.reserved_lineages.split(","), args.freeze_output)
        print("frozen manifest:", manifest, flush=True)
    if not manifest:
        parser.error("provide --manifest or --acquisition")
    if args.freeze_only:
        return
    if not args.output:
        parser.error("--output required for 1024-candidate preflight")
    summary = run_preflight(manifest, args.output)
    print(json.dumps({key: summary[key] for key in ("candidate_count", "accepted_count", "rejected_count",
                                                   "candidate_negative_count", "accepted_negative_count")}))


if __name__ == "__main__":
    main()
