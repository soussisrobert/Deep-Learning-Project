"""Wave3 fixed-panel preparation and independent CPU audit; GPU is explicit and serialized."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hift_research.evaluation.assets import read_json, sha256_file
from hift_research.data_acquisition.uav import acquire_uav_pilot
from hift_research.wave3_evaluation.preparation import freeze_plan, write_native_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="CPU metadata only: preregister official panel before images/inference")
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--inventory", required=True, type=Path)
    acquire = commands.add_parser("acquire", help="CPU selective official retrieval; no inference")
    acquire.add_argument("--plan", required=True, type=Path)
    acquire.add_argument("--inventory", required=True, type=Path)
    acquire.add_argument("--data-root", required=True, type=Path)
    acquire.add_argument("--output", required=True, type=Path)
    inputs = commands.add_parser("inputs", help="CPU: export raw+.5 native-number GT-free ordinary inputs")
    inputs.add_argument("--plan", required=True, type=Path)
    inputs.add_argument("--dataset-manifest", required=True, type=Path)
    inputs.add_argument("--output", required=True, type=Path)
    track = commands.add_parser("track", help="EXPLICIT GPU: only after lead allocation and CPU validation")
    track.add_argument("--plan", required=True, type=Path)
    track.add_argument("--dataset-manifest", required=True, type=Path)
    track.add_argument("--inference-dir", required=True, type=Path)
    track.add_argument("--output", required=True, type=Path)
    track.add_argument("--gpu-token", required=True)
    audit = commands.add_parser("audit", help="CPU: verify paired ordinary outputs independently")
    audit.add_argument("--plan", required=True, type=Path)
    audit.add_argument("--dataset-manifest", required=True, type=Path)
    audit.add_argument("--runs", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            plan = freeze_plan(ROOT, args.output, args.inventory)
            print(f"Frozen {len(plan['sequences'])} sequences / {plan['expected_frame_count']} frames / {plan['estimated_new_image_payload_bytes']} image-range bytes")
        elif args.command == "acquire":
            plan = read_json(args.plan)
            acquire_uav_pilot(args.data_root, args.output,
                sequences=tuple(r["name"] for r in plan["sequences"]), inventory=read_json(args.inventory),
                budget_bytes=plan["approved_image_budget_bytes"], selection_id=plan["selection_id"],
                selection_rule=plan["selection_rule"])
        elif args.command == "inputs":
            print(f"Created {len(write_native_inputs(args.dataset_manifest, args.plan, args.output))} GT-free inputs")
        elif args.command == "track":
            from hift_research.wave3_evaluation.audit import verify_pretracking_contract
            verify_pretracking_contract(ROOT, args.plan, args.dataset_manifest, args.inference_dir)
            subprocess.run([sys.executable, str(ROOT / "scripts/run_matched_subset.py"),
                "--dataset-manifest", str(args.dataset_manifest.resolve()), "--inference-dir", str(args.inference_dir.resolve()),
                "--output", str(args.output.resolve()), "--physical-annotation-origin", "1", "--gpu-token", args.gpu_token],
                cwd=ROOT, check=True)
        else:
            from hift_research.wave3_evaluation.audit import audit_panel
            report = audit_panel(ROOT, args.plan, args.dataset_manifest, args.runs, args.output)
            print(f"Independent paired audit PASS: {report['sequence_count']} sequences / {report['frame_count']} frames; deltaAUC={report['macro_delta_B1_minus_B0']['success_auc']:.9f}")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Wave3 failed closed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
