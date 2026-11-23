"""Cross-check the domain track's NumPy evaluator against the official HiFT/pysot success code.

Scores the sealed per-frame predictions in tracks/domain_ft/evidence/stage7/artifacts/b2-eval/ with
pysot's own `success_overlap` (copied verbatim below from tracks/night/toolkit/utils/statistics.py) and
compares each sequence with the success AUC stored in the matching *.evaluation.json.

Needs only the ground-truth annotation files (no images):
  UAV123@10fps: the prepared annotation copy used by stage 7 (hash-matched to its dataset_manifest.json),
                not the official anno/UAV123_10fps folder
  VisDrone2019-SOT-test-dev:       <root>/annotations/<seq>.txt

    python scripts/crosscheck_domain_vs_pysot.py --dataset UAV123 --anno-dir PATH [--n 10]
    python scripts/crosscheck_domain_vs_pysot.py --dataset VISDRONE --anno-dir PATH [--n 10]

Prints per-sequence |difference| and the maximum; equivalence = max |diff| < 1e-9.
"""
import argparse, json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "tracks/domain_ft/evidence/stage7/artifacts/b2-eval"


# ---- verbatim from pysot toolkit/utils/statistics.py (HiFT 992aa7c) ----
def overlap_ratio(rect1, rect2):
    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])
    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou


def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success
# ------------------------------------------------------------------------


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", " ").replace("\t", " ").split()[:4]])
    return np.array(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["UAV123", "VISDRONE"], required=True)
    ap.add_argument("--anno-dir", required=True)
    ap.add_argument("--arm", default="B0", help="B0, TC2-P1-REF-001 or TC2-P1-REF-S2-001")
    ap.add_argument("--n", type=int, default=0, help="first N sequences only (0 = all)")
    a = ap.parse_args()
    tag = f"STAGE2-SEALED-{a.dataset}-{a.arm}-sigmoid"
    ev = json.load(open(EV / f"{tag}.evaluation.json"))["sequences"]
    seqs = sorted(ev)[: a.n or None]
    worst = 0.0
    compared = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in seqs:
            gt_file = pathlib.Path(a.anno_dir) / f"{s}.txt"
            if not gt_file.exists():
                print(f"{s:<28} no GT file"); continue
            gt, pred = load(gt_file), load(EV / tag / f"{s}.csv")
            n = len(gt)
            if len(pred) != n:
                print(f"{s:<28} length mismatch gt {n} pred {len(pred)}"); continue
            gt = np.nan_to_num(gt, nan=0.0)          # pysot treats NaN rows as invalid (w,h not > 0)
            auc = float(np.mean(success_overlap(gt, pred, n)))
            diff = abs(auc - ev[s]["success_auc"])
            worst = max(worst, diff)
            compared += 1
            print(f"{s:<28} pysot {auc:.6f}  DroneUse {ev[s]['success_auc']:.6f}  |diff| {diff:.2e}")
    ok = compared > 0 and worst < 1e-9
    print(f"\n{compared}/{len(seqs)} sequences compared, max |diff| = {worst:.2e}  ->  {'EQUIVALENT' if ok else 'NOT SHOWN EQUIVALENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
