"""Per-sequence illumination statistics for NAT2021 / NAT2021L, optionally joined with tracking results.

For every (strided) frame: mean grayscale (0-255) and mean HSV V channel (0-255).
Per sequence: mean / std over frames, plus the same measured only inside the GT box.
If tracker results exist, the per-sequence success AUC is added and correlated with brightness.
"""
import os
import sys
import csv
import json
import argparse
from multiprocessing import Pool

import cv2
import numpy as np


def frame_stats(args):
    path, box = args
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = img.max(axis=2)
    out = [gray.mean(), v.mean()]
    if not np.all(np.isfinite(box)):
        out.append(np.nan)
        return out
    x, y, w, h = [int(round(b)) for b in box]
    x, y = max(x, 0), max(y, 0)
    roi = gray[y:y + max(h, 1), x:x + max(w, 1)]
    out.append(roi.mean() if roi.size else np.nan)
    return out


def seq_stats(video, root, stride, pool):
    idx = list(range(0, len(video['img_names']), stride))
    jobs = [(os.path.join(root, video['img_names'][i]), video['gt_rect'][i]) for i in idx]
    rows = [r for r in pool.map(frame_stats, jobs, chunksize=8) if r is not None]
    a = np.array(rows, dtype=float)
    return {
        'n_frames': len(video['img_names']),
        'gray_mean': np.nanmean(a[:, 0]), 'gray_std': np.nanstd(a[:, 0]),
        'v_mean': np.nanmean(a[:, 1]),
        'roi_gray_mean': np.nanmean(a[:, 2]),
    }


def iou(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def success_auc(video, res_file):
    if not os.path.exists(res_file):
        return np.nan
    pred = np.array([[float(v) for v in l.strip().split(',')] for l in open(res_file) if l.strip()])
    gt = np.array(video['gt_rect'], dtype=float)
    n = min(len(pred), len(gt))
    ious = iou(pred[:n], gt[:n])
    ious = ious[np.isfinite(ious)]
    if ious.size == 0:
        return np.nan
    return float(np.mean([(ious > t).mean() for t in np.linspace(0, 1, 21)]))


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NAT2021')
    parser.add_argument('--dataset_root', default='../test_dataset')
    parser.add_argument('--results', default='results')
    parser.add_argument('--tracker', default='first1.2')
    parser.add_argument('--stride', type=int, default=5, help='use every Nth frame')
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--out', default='illumination')
    args = parser.parse_args()

    root = os.path.join(args.dataset_root, args.dataset)
    meta = json.load(open(os.path.join(root, args.dataset + '.json')))
    os.makedirs(args.out, exist_ok=True)

    rows = []
    with Pool(args.procs) as pool:
        for i, (name, video) in enumerate(sorted(meta.items())):
            s = seq_stats(video, root, args.stride, pool)
            s['seq'] = name
            s['success_auc'] = success_auc(
                video, os.path.join(args.results, args.dataset, args.tracker, name + '.txt'))
            rows.append(s)
            print('({:3d}/{}) {} gray={:.1f} v={:.1f} auc={:.3f}'.format(
                i + 1, len(meta), name, s['gray_mean'], s['v_mean'], s['success_auc']), flush=True)

    cols = ['seq', 'n_frames', 'gray_mean', 'gray_std', 'v_mean', 'roi_gray_mean', 'success_auc']
    with open(os.path.join(args.out, args.dataset + '_illumination.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    g = np.array([r['gray_mean'] for r in rows])
    w_ = np.array([r['n_frames'] for r in rows], dtype=float)
    print('\n=== {} ({} sequences, every {}th frame) ==='.format(args.dataset, len(rows), args.stride))
    print('mean gray (per sequence): mean {:.1f}  median {:.1f}  min {:.1f}  max {:.1f}'.format(
        g.mean(), np.median(g), g.min(), g.max()))
    print('mean gray (frame weighted): {:.1f}'.format(np.average(g, weights=w_)))
    print('mean HSV-V: {:.1f}'.format(np.mean([r['v_mean'] for r in rows])))
    print('mean gray inside GT box: {:.1f}'.format(np.nanmean([r['roi_gray_mean'] for r in rows])))
    print('sequences with mean gray < 50: {}/{}'.format(int((g < 50).sum()), len(g)))
    auc = np.array([r['success_auc'] for r in rows])
    ok = ~np.isnan(auc)
    if ok.sum() > 2:
        rho = np.corrcoef(rank(g[ok]), rank(auc[ok]))[0, 1]
        print('Spearman(brightness, success AUC) = {:.3f} over {} sequences'.format(rho, ok.sum()))
        order = np.argsort(g)
        q = max(len(order) // 4, 1)
        print('darkest quarter  : mean gray {:.1f}, mean AUC {:.3f}'.format(
            g[order[:q]].mean(), np.nanmean(auc[order[:q]])))
        print('brightest quarter: mean gray {:.1f}, mean AUC {:.3f}'.format(
            g[order[-q:]].mean(), np.nanmean(auc[order[-q:]])))
