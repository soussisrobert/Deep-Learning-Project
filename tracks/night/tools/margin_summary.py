"""E13 summary for two margin-logged runs on the same dataset (logs written by tools/test_margin.py).

For each run: percentiles (5 / 25 / 50) of the margin (p1 - p2x) / p1 over all frames after the first, where p2x is the best
score of a cell whose decoded box centre is at least 0.3 x target size away from the top-1 (as in E13), and the number of
JUMP frames: frames where the top-1 box centre moves more than one target size (mean of w and h of the previous top-1 box) from
the previous frame's centre, which is almost always a switch to another peak (a heuristic, defined here for the comparison).
usage: python -I margin_summary.py <margin_dir> <run_a> <run_b>   (margin_dir = tools/margin/<dataset>)
"""
import os
import sys
import csv
import numpy as np

root, runs = sys.argv[1], sys.argv[2:]
for run in runs:
    margins, jumps, frames, seqs_jump = [], 0, 0, 0
    for f in sorted(os.listdir(os.path.join(root, run))):
        rows = list(csv.DictReader(open(os.path.join(root, run, f))))
        prev, sj = None, 0
        for r in rows:
            if int(r['frame']) == 0:
                prev = (float(r['cx1']), float(r['cy1']), float(r['w1']), float(r['h1']))
                continue
            p1, p2x = float(r['p1']), float(r['p2x'])
            if p1 > 0:
                margins.append((p1 - p2x) / p1)
            cx, cy = float(r['cx1']), float(r['cy1'])
            if prev is not None:
                d = np.hypot(cx - prev[0], cy - prev[1])
                if d > 0.5 * (prev[2] + prev[3]):
                    jumps += 1
                    sj = 1
            prev = (cx, cy, float(r['w1']), float(r['h1']))
            frames += 1
        seqs_jump += sj
    m = np.array(margins)
    print('{}: frames {}, margin percentiles 5/25/50 = {:.3f} / {:.3f} / {:.3f}, jump frames {} ({:.2f}% of frames, in {} sequences)'.format(
        run, frames, *np.percentile(m, [5, 25, 50]), jumps, 100. * jumps / frames, seqs_jump))
