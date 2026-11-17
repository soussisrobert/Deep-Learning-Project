"""Acquire the preregistered tiny official UAV10 pilot via strict bounded ZIP ranges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hift_research.data_acquisition.uav import PILOT_SEQUENCES, acquire_uav_pilot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--provenance-dir", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, help="Previously acquired official ZIP inventory/range cache to avoid rereading directory")
    parser.add_argument("--budget-mib", type=int, default=480, help="New transfer cap; <=480 leaves room in512MiB total for metadata/probes")
    parser.add_argument("--annotation-coordinate-origin", type=int, choices=[0, 1], help="Leave unset until justified by official annotation documentation")
    args = parser.parse_args(argv)
    if args.budget_mib <= 0 or args.budget_mib > 480:
        parser.error("budget must be in1..480MiB; initial total512MiB requires explicit lead approval to exceed")
    try:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory else None
        report = acquire_uav_pilot(args.output_root, args.provenance_dir, sequences=PILOT_SEQUENCES,
                                  inventory=inventory, budget_bytes=args.budget_mib * 1024 * 1024,
                                  annotation_coordinate_origin=args.annotation_coordinate_origin)
        print(f"Acquired {sum(v['frame_count'] for v in report['validation'].values())} complete official pilot frames")
        print(f"Dataset manifest: {args.provenance_dir / 'dataset_manifest.json'}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Acquisition failed closed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
