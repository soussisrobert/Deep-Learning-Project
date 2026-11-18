"""E19: effect of the test-time settings on first.pth, with intervals. CPU only, existing per-sequence success AUCs, no tracking.
Source: tools/illumination/final_A_<dataset>.csv, columns 1-3 = first.pth with no enhancement / gamma 0.6 / anchor 0.2 + gamma 0.6
(the same three columns are repeated in the final_B/C/D files; E16 asserted they are identical).
Step 1: the means must match the logged ones within 0.002, otherwise the script prints the difference and exits with status 1 before any bootstrap.
Step 2: paired bootstrap over sequences (2000 resamples, RandomState(0), 95% percentile interval) of gamma - none, anchor+gamma - none,
anchor+gamma - gamma; better / worse / same = per-sequence difference > +0.005 / < -0.005 / otherwise.
usage (from tools/): python -I illumination/E19_settings_bootstrap.py
"""
import os
import sys
import csv
import numpy as np

OUT = 'illumination'
LOGGED = {'NAT2021': (0.372, 0.417, 0.453), 'NAT2021L': (0.299, 0.316, 0.377),
          'NATUD135TEST': (0.352, 0.434, 0.443), 'NATUAV123HELD': (0.612, 0.613, 0.616)}
NAMES = ['none', 'gamma0.6', 'anchor+gamma0.6']
PAIRS = [(1, 0), (2, 0), (2, 1)]
NB = 2000
TOL = 0.005


def read(ds):
    rows = list(csv.reader(open(os.path.join(OUT, 'final_A_{}.csv'.format(ds)))))
    return [r[0] for r in rows[1:]], np.array([[float(x) for x in r[1:4]] for r in rows[1:]])


data = {}
bad = False
print('step 1: means against the logged values (tolerance 0.002)')
for ds, logged in LOGGED.items():
    seqs, v = read(ds)
    data[ds] = (seqs, v)
    m = v.mean(axis=0)
    diff = m - np.array(logged)
    ok = bool((np.abs(diff) <= 0.002).all())
    bad |= not ok
    print('   {:14s} n={:3d} measured {} logged {} diff {}  {}'.format(
        ds, len(seqs), ' / '.join('{:.4f}'.format(x) for x in m), ' / '.join('{:.3f}'.format(x) for x in logged),
        ' / '.join('{:+.4f}'.format(x) for x in diff), 'ok' if ok else 'MISMATCH'))
if bad:
    print('STOP: at least one mean differs from the logged value by more than 0.002')
    sys.exit(1)

rows = []
print('\nstep 2: paired bootstrap over sequences, {} resamples, RandomState(0), 95% interval; better/worse/same at +-{}'.format(NB, TOL))
for ds in LOGGED:
    seqs, v = data[ds]
    print('\n=== {} ({} sequences)'.format(ds, len(seqs)))
    for a, b in PAIRS:
        d = v[:, a] - v[:, b]
        rng = np.random.RandomState(0)
        bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(NB)])
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        better, worse = int((d > TOL).sum()), int((d < -TOL).sum())
        same = len(d) - better - worse
        verdict = 'differs' if (lo > 0 or hi < 0) else 'within noise'
        name = '{} - {}'.format(NAMES[a], NAMES[b])
        print('   {:32s} {:+.4f} [{:+.4f}, {:+.4f}]  better {:3d} worse {:3d} same {:3d}  {}'.format(name, d.mean(), lo, hi, better, worse, same, verdict))
        rows.append([ds, name, len(d), v[:, a].mean(), v[:, b].mean(), d.mean(), lo, hi, better, worse, same, verdict])
path = os.path.join(OUT, 'E19_settings_bootstrap.csv')
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset', 'comparison', 'n_sequences', 'mean_a', 'mean_b', 'mean_diff', 'lo', 'hi', 'better', 'worse', 'same', 'verdict'])
    w.writerows(rows)
print('\nwrote', path)
