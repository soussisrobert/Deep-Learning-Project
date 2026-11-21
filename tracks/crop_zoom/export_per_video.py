# Paste into a new Kaggle cell after the runs (same notebook session as the results/ folders).
# Writes one per-video CSV in the group's common format, for both datasets.
# Scoring rule matches the paper protocol: frames with missing/invalid GT count as failures
# (IoU 0, centre error infinite), i.e. they stay in the denominator.
import glob, os, csv
import numpy as np

PROJ = "/kaggle/working/proj"
RUNS = {  # results folder -> dataset label, GT folder, subsample 30fps GT to 10fps?
    "UAV10fps": ("UAV123@10fps(subsampled)", "/kaggle/input/datasets/galaxythereal/uav123-tracking-dataset/anno/UAV123", True),
    "UAV20l":   ("UAV20L", "/tmp/test_dataset/UAV123_20L/anno", False),
}
THR = np.linspace(0, 1, 21)

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append([float(x) if x.lower() != "nan" else np.nan for x in line.replace(",", " ").split()[:4]])
    return np.array(rows, dtype=np.float64)

def score(pred, gt):
    n = min(len(pred), len(gt)); pred, gt = pred[:n], gt[:n]
    valid = (~np.isnan(gt).any(1)) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
    iou = np.zeros(n); cle = np.full(n, np.inf)
    p, g = pred[valid], gt[valid]
    x1 = np.maximum(p[:, 0], g[:, 0]); y1 = np.maximum(p[:, 1], g[:, 1])
    x2 = np.minimum(p[:, 0] + p[:, 2], g[:, 0] + g[:, 2]); y2 = np.minimum(p[:, 1] + p[:, 3], g[:, 1] + g[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = p[:, 2] * p[:, 3] + g[:, 2] * g[:, 3] - inter
    iou[valid] = np.where(union > 0, inter / union, 0)
    cle[valid] = np.hypot(p[:, 0] + p[:, 2] / 2 - g[:, 0] - g[:, 2] / 2, p[:, 1] + p[:, 3] / 2 - g[:, 1] - g[:, 3] / 2)
    return n, float(np.mean([(iou > t).mean() for t in THR])), float((cle <= 20).mean())

out = "/kaggle/working/per_video_crop_zoom.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark", "video", "frames", "model", "seed", "success_auc", "precision20", "toolkit"])
    for ds, (label, anno, sub) in RUNS.items():
        for tdir in sorted(glob.glob(f"{PROJ}/results/{ds}/*")):
            model = os.path.basename(tdir)            # e.g. base1.2, z0901.2, interp_auto1.2
            for pf in sorted(glob.glob(f"{tdir}/*.txt")):
                vid = os.path.basename(pf)[:-4]
                gf = os.path.join(anno, vid + ".txt")
                if not os.path.exists(gf):
                    print("no GT for", ds, vid); continue
                pred, gt = load(pf), load(gf)
                if sub and len(gt) > 2 * len(pred):
                    gt = gt[::3]
                n, auc, p20 = score(pred, gt)
                w.writerow([label, vid, n, model, 0, repr(auc), repr(p20), "own"])
print("wrote", out)
import pandas as pd
d = pd.read_csv(out); print(d.groupby(["benchmark", "model"]).success_auc.agg(["mean", "count"]).round(4))
