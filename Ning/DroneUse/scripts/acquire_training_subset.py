"""Bounded official TRAIN-only archive inspection/acquisition entry point."""
import argparse
from pathlib import Path
from hift_research.data_contract.acquisition import (
    ANNOTATIONS_URL, IMAGES_URL, BUDGET, inspect_archive, write_exclusive_json, acquire_subset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--probe-evidence")
    parser.add_argument("--data-root")
    parser.add_argument("--budget-mib", type=int, default=128)
    args = parser.parse_args()
    if args.acquire:
        if not args.probe_evidence or not args.data_root:
            parser.error("--acquire requires --probe-evidence and --data-root")
        value = acquire_subset(args.probe_evidence, args.output, args.data_root, budget=args.budget_mib*1024*1024)
        print({"images": value["image_count"], "bytes": value["bytes_received_including_probe"]})
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    ledger, probes = [], []
    remaining = BUDGET
    for name, url in (("annotations", ANNOTATIONS_URL), ("images", IMAGES_URL)):
        metadata, reader, archive, members = inspect_archive(url, root / (name + "-ranges"), budget=remaining)
        record = dict(metadata, members=members if name == "annotations" else None,
                      member_count=len(members), directory_bytes_received=reader.network_bytes)
        if name == "images":
            record["first_sorted_members"] = sorted(members, key=lambda x: x["name"])[:40]
        probes.append(record)
        ledger.extend(reader.records)
        remaining -= reader.network_bytes
        archive.close()
        reader.close()
    write_exclusive_json(root / "probe.json", {"budget_bytes": BUDGET, "bytes_received": BUDGET - remaining,
                                              "archives": probes, "network_ledger": ledger})
    print(root / "probe.json")
    if not args.probe:
        raise SystemExit("Inspection complete; acquisition requires frozen preselection implementation")


if __name__ == "__main__":
    main()
