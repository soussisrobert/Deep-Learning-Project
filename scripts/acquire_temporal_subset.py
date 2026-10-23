"""Acquire only a frozen, original, bounded MOT17 temporal pilot; CPU/no SGD."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hift_research.temporal_source.acquisition import acquire_temporal_subset
from hift_research.evaluation.assets import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--index-ledger", type=Path, required=True)
    p.add_argument("--reservations", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        result = acquire_temporal_subset(args.selection, args.index_ledger, args.reservations, args.data_root, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Temporal source failed closed: {exc}", file=sys.stderr)
        return 2
    print(f"Frozen {result['validation']['original_frames']} original frames across {len(result['videos'])} source videos")
    print(f"manifest_sha256={sha256_file(args.output / 'temporal_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
