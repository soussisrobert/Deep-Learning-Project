"""Verify hashes and reproduce Robert's per-video metrics from the saved boxes.

Run with Python and NumPy from any directory. No tracker inference is needed.
"""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {relative}")
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifests/disjoint34.json").read_text(encoding="utf-8"))
    domain_rows = list(csv.DictReader((root / "comparison/domain_B0_REF_VisDrone34.csv").open()))
    domain_names = {row["video"] for row in domain_rows if row["model"] == "B0"}
    assert set(manifest) == domain_names and len(manifest) == 34
    assert len({name.split("_")[0] for name in manifest}) == 22
    split_sources = []
    for split in ("train", "val", "test"):
        rows = json.loads((root / f"manifests/{split}.json").read_text(encoding="utf-8"))
        split_sources.append({name.split("_")[0] for name in rows})
    assert not split_sources[0] & split_sources[1]
    assert not split_sources[0] & split_sources[2]
    assert not split_sources[1] & split_sources[2]
    scorer = load_module("domain_metrics", root / "scoring/domain_metrics.py")
    stats = load_module("team_stats", root / "scoring/team_stats.py")
    with (root / "per_video_VisDrone34.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 68
    labels = {"HiFT_stock_raw": "stock_hift", "HiFT_corner_pipeline": "corner_pipeline"}
    recomputed = {model: {} for model in labels.values()}
    for row in rows:
        name, model = row["video"], labels[row["model"]]
        boxes = np.loadtxt(root / f"predictions/{model}/{name}.csv", delimiter=",")
        score = scorer.evaluate_sequence(manifest[name]["gt_rect"], boxes,
                                         invalid_policy="count_as_failure")
        assert int(row["frames"]) == score["frame_count"]
        assert abs(float(row["success_auc"]) - score["success_auc"]) < 1e-14
        assert abs(float(row["precision20"]) - score["precision_at_20px"]) < 1e-14
        recomputed[model][name] = score["success_auc"]
    names = sorted(manifest)
    deltas = [recomputed["corner_pipeline"][name] - recomputed["stock_hift"][name] for name in names]
    estimate = stats.cluster_bootstrap(deltas, [name.split("_")[0] for name in names], seed=20261105)
    reported = audit["replay_test34"]
    assert np.allclose(estimate, [reported["paired_gain"], *reported["ci95"]], rtol=0, atol=1e-14)
    print(f"Verified {len(lines)} file hashes, the exact 34-video set, split source separation, "
          "68 per-video score rows, and the paired cluster interval.")
    print(f"AUC gain {100 * estimate[0]:+.4f} points, "
          f"95% interval [{100 * estimate[1]:+.4f}, {100 * estimate[2]:+.4f}].")


if __name__ == "__main__":
    main()
