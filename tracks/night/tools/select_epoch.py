"""Per-epoch curve and checkpoint choice from eval_nat.py --csv output (per-sequence success AUC, one column per tracker).

usage: python -I select_epoch.py <csv> <run> <curve_substrings comma separated> <select_substring>
A column belongs to a curve if its name ends with <substring>_e<k>. Prints the whole curve (every epoch, not only the best) for each
curve and the reference columns (first.pth). The epoch with the highest MEAN per-sequence AUC of the <select_substring> curve is written to
<HIFT_ROOT>/experiments/night_ft/runs/<run>/selected.txt (ties -> later epoch); HIFT_ROOT = environment variable, default: the repo root (the parent of tools/).
"""
import os
import sys
import csv
import re
import numpy as np

path, run, curves, select = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
rows = list(csv.reader(open(path)))
head, body = rows[0], rows[1:]
auc = {h: np.array([float(r[i]) for r in body]) for i, h in enumerate(head) if i > 0}
print('{} sequences'.format(len(body)))
used = set()
curve_pts = {}
for p in curves:
    pts = []
    for h in head[1:]:
        m = re.search(re.escape(p) + r'_e(\d+)$', h)
        if m:
            pts.append((int(m.group(1)), auc[h].mean(), h))
            used.add(h)
    curve_pts[p] = sorted(pts)
    print('--- curve {}: mean per-sequence success AUC per epoch'.format(p))
    for e, a, h in curve_pts[p]:
        print('   epoch {:2d}  {:.4f}'.format(e, a))
print('--- reference columns (first.pth, not trained)')
for h in head[1:]:
    if h not in used:
        print('   {}: {:.4f}'.format(h, auc[h].mean()))
cand = curve_pts[select]
best = max(cand, key=lambda t: (round(t[1], 6), t[0]))
print('SELECTED epoch {} (mean success AUC {:.4f}) by the highest mean per-sequence AUC of curve {}'.format(best[0], best[1], select))
root = os.environ.get('HIFT_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
open(os.path.join(root, 'experiments/night_ft/runs/{}/selected.txt'.format(run)), 'w').write(str(best[0]) + '\n')
