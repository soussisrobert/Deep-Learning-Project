"""One line chart per sequence: mean frame brightness (gray, 0-255) against frame number.

Writes plots/<dataset>/<seq>.png and <dataset>_per_frame.csv (seq, frame, mean_gray) in --out.
All charts share one y-axis range so sequences can be compared by eye.
"""
import os
import csv
import json
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINE = '#2563c9'      # single series: one hue
INK = '#1f2933'       # primary text
MUTED = '#6b7785'     # secondary text / axes
GRID = '#e3e7ec'      # recessive grid
REF = '#9aa5b1'       # dataset-average reference


def mean_gray(path):
    img = cv2.imread(path)
    if img is None:
        return np.nan
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def plot_seq(name, frames, vals, ds_mean, ymax, out_png, dataset):
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=130)
    ax.plot(frames, vals, color=LINE, linewidth=1.8, solid_capstyle='round')
    ax.axhline(ds_mean, color=REF, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(frames[-1], ds_mean, ' {} avg {:.0f}'.format(dataset, ds_mean), color=MUTED,
            fontsize=7.5, va='bottom', ha='right')
    ax.set_ylim(0, ymax)
    ax.set_xlim(frames[0], frames[-1])
    ax.set_xlabel('Frame', color=MUTED, fontsize=9)
    ax.set_ylabel('Mean gray (0-255)', color=MUTED, fontsize=9)
    ax.set_title('{}: mean illumination per frame (this sequence avg {:.1f})'.format(
        name, float(np.nanmean(vals))), color=INK, fontsize=10.5, loc='left')
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(out_png, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NAT2021')
    parser.add_argument('--dataset_root', default='../test_dataset')
    parser.add_argument('--stride', type=int, default=5, help='use every Nth frame')
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--out', default='illumination')
    args = parser.parse_args()

    root = os.path.join(args.dataset_root, args.dataset)
    meta = json.load(open(os.path.join(root, args.dataset + '.json')))
    plot_dir = os.path.join(args.out, 'plots', args.dataset)
    os.makedirs(plot_dir, exist_ok=True)

    data = {}
    with Pool(args.procs) as pool:
        for i, (name, video) in enumerate(sorted(meta.items())):
            idx = list(range(0, len(video['img_names']), args.stride))
            paths = [os.path.join(root, video['img_names'][j]) for j in idx]
            vals = np.array(pool.map(mean_gray, paths, chunksize=8))
            data[name] = (np.array(idx) + 1, vals)       # 1-based frame numbers
            print('({:3d}/{}) {} frames={} mean={:.1f}'.format(
                i + 1, len(meta), name, len(idx), np.nanmean(vals)), flush=True)

    all_vals = np.concatenate([v for _, v in data.values()])
    ds_mean = float(np.nanmean(all_vals))
    ymax = float(np.ceil(np.nanmax(all_vals) / 10.0) * 10)
    print('dataset mean gray {:.1f}, shared y-axis 0-{:.0f}'.format(ds_mean, ymax))

    with open(os.path.join(args.out, args.dataset + '_per_frame.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq', 'frame', 'mean_gray'])
        for name, (fr, v) in data.items():
            for a, b in zip(fr, v):
                w.writerow([name, int(a), round(float(b), 2)])

    for name, (fr, v) in data.items():
        plot_seq(name, fr, v, ds_mean, ymax, os.path.join(plot_dir, name + '.png'), args.dataset)
    print('wrote', len(data), 'charts to', plot_dir)
