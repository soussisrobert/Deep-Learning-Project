"""E21 steps B and C (CPU, after the selected setting was run on NAT2021, NAT2021L, NATUD135TEST).
Selected setting: illumination/E21_selected.txt. Current settings: first1.2_m_anchor (NAT2021, NAT2021L), first1.2_ag (NATUD135TEST).
B: success AUC, paired bootstrap selected - current (2000 resamples, RandomState(0), 95%), sequences better / worse / same (+-0.005 per sequence), fps from timing/.../times.csv
   (precision and normalised precision come from the official eval_nat.py table printed by the job).
C: size diagnostics on found frames (t >= 1, usable GT, centre error <= 20 px) for the selected and the current setting.
usage (from tools/): python -I illumination/E21_report.py
"""
import os
import sys
import csv
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'illumination'))
from E21_common import load_gt, run_stats, boot

lr, pk, wi = open('illumination/E21_selected.txt').read().split()
SEL = 'first1.2_E21_lr{}_pk{}_wi{}'.format(lr, pk, wi)
SETS = [('NAT2021', 'first1.2_m_anchor'), ('NAT2021L', 'first1.2_m_anchor'), ('NATUD135TEST', 'first1.2_ag')]
print('selected: LR {} PENALTY_K {} WINDOW_INFLUENCE {} (folder {})'.format(lr, pk, wi, SEL))
brow, crow = [], []
for ds, cur in SETS:
    seqs, gt = load_gt(ds)
    s_sel, s_cur = run_stats(ds, SEL, seqs, gt), run_stats(ds, cur, seqs, gt)
    d = s_sel['auc'] - s_cur['auc']
    m, lo, hi = boot(d)
    better, worse = int((d > 0.005).sum()), int((d < -0.005).sum())
    same = len(d) - better - worse
    fps = {}
    for k, run in (('sel', SEL), ('cur', cur)):
        p = os.path.join('timing', ds, run, 'times.csv')
        if os.path.exists(p):
            t = list(csv.DictReader(open(p)))
            fps[k] = sum(float(r['frames']) for r in t) / sum(float(r['seconds']) for r in t)
        else:
            fps[k] = float('nan')
    verdict = 'differs' if (lo > 0 or hi < 0) else 'within noise'
    print('\n=== {} ({} sequences)'.format(ds, len(seqs)))
    print('   success AUC selected {:.4f}  current ({}) {:.4f}   fps selected {:.1f}'.format(s_sel['auc'].mean(), cur, s_cur['auc'].mean(), fps['sel']))
    print('   selected - current {:+.4f} [{:+.4f}, {:+.4f}]  better {} worse {} same {} (+-0.005)  {}'.format(m, lo, hi, better, worse, same, verdict))
    brow.append([ds, cur, SEL, len(seqs), s_cur['auc'].mean(), s_sel['auc'].mean(), m, lo, hi, better, worse, same, verdict, fps['cur'], fps['sel']])
    print('   size diagnostics on found frames (selected | current):')
    for name, st in (('selected', s_sel), ('current', s_cur)):
        print('      {:9s} found frames {:6d} (share of all frames {:.3f})  mean IoU {:.3f}  width ratio median {:.3f} [{:.3f}, {:.3f}]  height ratio median {:.3f} [{:.3f}, {:.3f}]  area > 1.5x {:.1f}%  area < 0.67x {:.1f}%'.format(
            name, st['n_found'], st['found_share'], st['iou'], *st['w'], *st['h'], 100 * st['a_gt_1.5'], 100 * st['a_lt_0.67']))
        crow.append([ds, name, st['n_found'], st['found_share'], st['iou']] + list(st['w']) + list(st['h']) + [st['a_gt_1.5'], st['a_lt_0.67']])
for name, header, body in (
        ('E21_selected_vs_current.csv', ['dataset', 'current_folder', 'selected_folder', 'n_sequences', 'auc_current', 'auc_selected', 'mean_diff', 'lo', 'hi', 'better', 'worse', 'same', 'verdict', 'fps_current', 'fps_selected'], brow),
        ('E21_size_diagnostics.csv', ['dataset', 'setting', 'found_frames', 'found_share', 'found_mean_iou', 'width_median', 'width_p5', 'width_p95', 'height_median', 'height_p5', 'height_p95', 'share_area_gt_1.5', 'share_area_lt_0.67'], crow)):
    with open(os.path.join('illumination', name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    print('wrote illumination/' + name)
