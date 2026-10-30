"""One-pass evaluation (OPE) metrics, matching the pysot toolkit conventions HiFT uses.

success AUC : mean over IoU thresholds 0, 0.05, ..., 1.0 of the fraction of frames
              whose IoU exceeds the threshold (frames with no valid ground truth count as 0 IoU)
precision   : fraction of frames whose centre error is <= 20 pixels; frames without a valid
              ground-truth box count as misses (strict rule, used by all earlier runs)
*_official  : the rule in HiFT's own toolkit (toolkit/utils/statistics.py): for precision, a frame
              whose ground-truth centre is not positive in both coordinates (target absent / NaN)
              counts as a HIT. Success is identical under both rules. Use *_official when
              comparing with numbers printed in the HiFT paper.
"""
import numpy as np

IOU_THRESHOLDS = np.arange(0, 1.05, 0.05)
PIX_THRESHOLDS = np.arange(0, 51, 1)


def _valid(gt):
    return np.all(np.isfinite(gt), axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)


def iou(a, b):
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def centre_error(a, b):
    ca = a[:, :2] + (a[:, 2:] - 1) / 2
    cb = b[:, :2] + (b[:, 2:] - 1) / 2
    return np.sqrt(((ca - cb) ** 2).sum(axis=1))


def sequence_scores(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    ok = _valid(gt)
    ious = np.full(n, -1.0)
    ious[ok] = iou(pred[ok], gt[ok])
    errs = np.full(n, np.inf)
    errs[ok] = centre_error(pred[ok], gt[ok])
    succ_curve = np.array([(ious > t).mean() for t in IOU_THRESHOLDS])
    prec_curve = np.array([(errs <= t).mean() for t in PIX_THRESHOLDS])
    # official toolkit rule
    gc = gt[:, :2] + (gt[:, 2:] - 1) / 2
    with np.errstate(invalid="ignore"):
        cmask = np.sum(gc > 0, axis=1) == 2
    dist = np.full(n, -1.0)
    dist[cmask] = centre_error(pred[cmask], gt[cmask])
    prec_off = np.array([(dist <= t).mean() for t in PIX_THRESHOLDS])
    return {
        "frames": int(n),
        "success_auc": float(succ_curve.mean()),
        "precision_20": float(prec_curve[20]),
        "mean_iou_valid": float(ious[ok].mean()) if ok.any() else float("nan"),
        "success_curve": succ_curve.tolist(),
        "precision_curve": prec_curve.tolist(),
        "precision_20_official": float(prec_off[20]),
        "precision_curve_official": prec_off.tolist(),
        "absent_frames": int((~ok).sum()),
    }


def aggregate(per_seq):
    """Average over sequences (pysot convention: each sequence weighs the same)."""
    if not per_seq:
        return {}
    succ = np.mean([s["success_curve"] for s in per_seq], axis=0)
    prec = np.mean([s["precision_curve"] for s in per_seq], axis=0)
    extra = {}
    if all("precision_curve_official" in s for s in per_seq):
        po = np.mean([s["precision_curve_official"] for s in per_seq], axis=0)
        extra = {"precision_20_official": float(po[20]), "precision_curve_official": po.tolist()}
    return {
        "sequences": len(per_seq),
        "frames": int(sum(s["frames"] for s in per_seq)),
        "success_auc": float(succ.mean()),
        "precision_20": float(prec[20]),
        "success_curve": succ.tolist(),
        "precision_curve": prec.tolist(),
        **extra,
    }
