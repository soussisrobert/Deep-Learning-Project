"""E21 step A (CPU): the 27-point grid on UAVDark135 val. usage (from tools/):
   python -I illumination/E21_select.py check     the default point must reproduce 0.4934 (+-0.002), else exit 1
   python -I illumination/E21_select.py select    full table, one-at-a-time sensitivity, selection by the pre-fixed rule
Grid: LR 0.15 / 0.30 / 0.45, PENALTY_K 0.04 / 0.08 / 0.16, WINDOW_INFLUENCE 0.32 / 0.42 / 0.52 (defaults in the middle for LR and WINDOW_INFLUENCE; PENALTY_K default 0.08).
Selection rule (fixed before any run): the point with the highest mean success AUC on val; if other points are within 0.002 of it, the one closest to the defaults
(smallest number of grid steps away from the default point, summed over the three parameters), then the higher AUC. A best-of-27 choice on 16 sequences: the val gain is optimistic.
Found frames: t >= 1, usable GT, centre error <= 20 px; found share = found frames / all frames of the 16 sequences.
"""
import os
import sys
import csv
import json
import itertools
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'illumination'))
from E21_common import load_gt, run_stats

DS = 'NATUD135VAL'
LRS, PKS, WIS = ['0.15', '0.30', '0.45'], ['0.04', '0.08', '0.16'], ['0.32', '0.42', '0.52']
DEFAULT = ('0.30', '0.08', '0.42')
TARGET = 0.4934


def folder(p):
    return 'first1.2_E21_lr{}_pk{}_wi{}'.format(*p)



def dist(p):
    return abs(LRS.index(p[0]) - 1) + abs(PKS.index(p[1]) - 1) + abs(WIS.index(p[2]) - 1)


seqs, gt = load_gt(DS)
mode = sys.argv[1]
if mode == 'check':
    st = run_stats(DS, folder(DEFAULT), seqs, gt)
    a = st['auc'].mean()
    print('default point {} on {}: mean success AUC {:.4f} (expected {:.4f}, tolerance 0.002)'.format(DEFAULT, DS, a, TARGET))
    if abs(a - TARGET) > 0.002:
        print('STOP: the default grid point does not reproduce the baseline')
        sys.exit(1)
    sys.exit(0)

pts = list(itertools.product(LRS, PKS, WIS))
res = {}
for p in pts:
    assert os.path.isdir(os.path.join('results', DS, folder(p))), folder(p)
    res[p] = run_stats(DS, folder(p), seqs, gt)
a_def = res[DEFAULT]['auc'].mean()
print('default point AUC {:.4f} (expected {:.4f})'.format(a_def, TARGET))
if abs(a_def - TARGET) > 0.002:
    print('STOP: default does not reproduce')
    sys.exit(1)

print('\nfull grid on {} ({} sequences); columns: LR, PENALTY_K, WINDOW_INFLUENCE | success AUC | found-frame mean IoU | found share | width ratio median [p5, p95] | height ratio median [p5, p95]'.format(DS, len(seqs)))
rows = []
for p in sorted(pts, key=lambda q: -res[q]['auc'].mean()):
    r = res[p]
    print('   {} {} {} | {:.4f} | {:.3f} | {:.3f} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} [{:.3f}, {:.3f}]{}'.format(
        *p, r['auc'].mean(), r['iou'], r['found_share'], *r['w'], *r['h'], '   <- default' if p == DEFAULT else ''))
for p in pts:
    r = res[p]
    rows.append([p[0], p[1], p[2], r['auc'].mean(), r['auc'].mean() - a_def, r['iou'], r['found_share'], r['n_found']] + list(r['w']) + list(r['h']) + [r['a_gt_1.5'], r['a_lt_0.67']])
with open('illumination/E21_grid_val.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['LR', 'PENALTY_K', 'WINDOW_INFLUENCE', 'success_auc', 'diff_vs_default', 'found_mean_iou', 'found_share', 'found_frames',
                'width_ratio_median', 'width_ratio_p5', 'width_ratio_p95', 'height_ratio_median', 'height_ratio_p5', 'height_ratio_p95', 'share_area_gt_1.5', 'share_area_lt_0.67'])
    w.writerows(rows)
print('wrote illumination/E21_grid_val.csv')

print('\none-at-a-time sensitivity (the other two parameters at their defaults) and marginal means (average of the 9 points with that value)')
sens = []
for i, (name, vals) in enumerate((('LR', LRS), ('PENALTY_K', PKS), ('WINDOW_INFLUENCE', WIS))):
    one = []
    marg = []
    for v in vals:
        q = list(DEFAULT); q[i] = v
        one.append(res[tuple(q)]['auc'].mean())
        marg.append(np.mean([res[p]['auc'].mean() for p in pts if p[i] == v]))
    print('   {:17s} one-at-a-time {} range {:.4f} | marginal {} range {:.4f}'.format(
        name, ' / '.join('{}: {:.4f}'.format(v, a) for v, a in zip(vals, one)), max(one) - min(one),
        ' / '.join('{}: {:.4f}'.format(v, a) for v, a in zip(vals, marg)), max(marg) - min(marg)))
    sens.append([name] + one + [max(one) - min(one)] + marg + [max(marg) - min(marg)])
with open('illumination/E21_sensitivity_val.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['parameter', 'one_low', 'one_default', 'one_high', 'one_range', 'marg_low', 'marg_default', 'marg_high', 'marg_range'])
    w.writerows(sens)

best = max(res[p]['auc'].mean() for p in pts)
cands = [p for p in pts if res[p]['auc'].mean() >= best - 0.002]
sel = sorted(cands, key=lambda p: (dist(p), -res[p]['auc'].mean(), p))[0]
print('\nselection: best val AUC {:.4f}; points within 0.002 of it: {}'.format(best, len(cands)))
for p in sorted(cands, key=lambda p: (dist(p), -res[p]['auc'].mean())):
    print('      {} AUC {:.4f} steps from default {}'.format(p, res[p]['auc'].mean(), dist(p)))
print('SELECTED: LR {} PENALTY_K {} WINDOW_INFLUENCE {}  val AUC {:.4f} (default {:.4f}, gain {:+.4f}); best of 27 chosen on 16 sequences, so this gain is optimistic'.format(
    *sel, res[sel]['auc'].mean(), a_def, res[sel]['auc'].mean() - a_def))
open('illumination/E21_selected.txt', 'w').write('{} {} {}\n'.format(*sel))
json.dump({'lr': sel[0], 'penalty_k': sel[1], 'window_influence': sel[2], 'val_auc': res[sel]['auc'].mean(), 'default_val_auc': a_def,
           'candidates_within_0.002': [list(c) for c in cands]}, open('illumination/E21_selected.json', 'w'))
