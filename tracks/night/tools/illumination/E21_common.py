"""E21 helpers: scoring as toolkit/eval_nat.py (success_overlap: GT w, h must be > 0 else IoU -1, all frames in the denominator) and the E20 size diagnostics."""
import os
import json
import numpy as np

ROOT = '../test_dataset'
THR = np.arange(0, 1.05, 0.05)
PX = 20.0


def overlap(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return np.clip(inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter), 0, 1)


def iou_of(gt, pred):
    iou = np.ones(len(gt)) * -1
    m = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        iou[m] = overlap(gt[m], pred[m])
    return iou


def auc_of(gt, pred):
    iou = iou_of(gt, pred)
    return float(np.mean([(iou > t).sum() / float(len(iou)) for t in THR]))


def load_gt(ds):
    meta = json.load(open(os.path.join(ROOT, ds, ds + '.json')))
    seqs = sorted(meta)
    return seqs, {s: np.array(meta[s]['gt_rect'], dtype=float) for s in seqs}


def pred(ds, run, s):
    return np.loadtxt(os.path.join('results', ds, run, s + '.txt'), delimiter=',', ndmin=2)


def boot(d, n=2000):
    d = np.asarray(d, float)
    rng = np.random.RandomState(0)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def run_stats(ds, run, seqs, gt):
    """per-sequence AUC and, on found frames (t >= 1, usable GT, centre error <= 20 px), IoU / width ratio / height ratio / area ratio.
    found share = found frames / ALL frames of the set (frame 0, frames without a usable GT and lost frames are in the denominator)."""
    auc, I, W, H, A = [], [], [], [], []
    n_all = 0
    for s in seqs:
        g, p = gt[s], pred(ds, run, s)
        assert len(g) == len(p), (ds, run, s)
        auc.append(auc_of(g, p))
        n_all += len(g)
        usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
        err = np.hypot(p[:, 0] + p[:, 2] / 2. - g[:, 0] - g[:, 2] / 2., p[:, 1] + p[:, 3] / 2. - g[:, 1] - g[:, 3] / 2.)
        found = usable & (err <= PX)
        found[0] = False
        I.append(iou_of(g, p)[found])
        W.append(p[found, 2] / g[found, 2])
        H.append(p[found, 3] / g[found, 3])
        A.append(p[found, 2] * p[found, 3] / (g[found, 2] * g[found, 3]))
    I, W, H, A = [np.concatenate(x) for x in (I, W, H, A)]
    q = lambda x: (float(np.median(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95)))
    return {'auc': np.array(auc), 'n_found': len(I), 'found_share': len(I) / float(n_all), 'iou': float(I.mean()),
            'w': q(W), 'h': q(H), 'a_gt_1.5': float((A > 1.5).mean()), 'a_lt_0.67': float((A < 0.67).mean())}
