"""E16: paired bootstraps between the selected checkpoints of A (epoch 8), B (8), C (10), D (1) and first.pth, within the same test-time setting.
Reads the per-sequence success AUCs already written by job_night_final.sh (tools/illumination/final_<run>_<dataset>.csv: columns 2-4 = first.pth
none / gamma 0.6 / anchor+gamma, columns 5-7 = the run in the same order). No tracking. Bootstrap as in eval_nat.py: resample sequences, 2000
draws, RandomState(0), 95% percentile interval. better / worse = sequences where the first named run has the higher / lower AUC.
usage (from tools/): python -I ablation_boot.py
"""
import os
import csv
import numpy as np

OUT = 'illumination'
DATASETS = ['NAT2021', 'NAT2021L', 'NATUD135TEST', 'NATUAV123HELD']
SETTINGS = ['none', 'gamma0.6', 'anchor+gamma0.6']
RUNS = ['A', 'B', 'C', 'D']
COMPARE = [(1, 'B', 'C'), (1, 'A', 'C'), (1, 'A', 'B'), (2, 'A', 'D'), (2, 'B', 'D'), (2, 'C', 'D'),
           (3, 'A', 'first'), (3, 'B', 'first'), (3, 'C', 'first'), (3, 'D', 'first')]


def read(run, ds):
    rows = list(csv.reader(open(os.path.join(OUT, 'final_{}_{}.csv'.format(run, ds)))))
    seqs = [r[0] for r in rows[1:]]
    vals = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return seqs, vals


def boot(a, b, n=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(seed)
    m = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), len(d)


out = []
for ds in DATASETS:
    auc = {}
    seqs0 = None
    for run in RUNS:
        seqs, v = read(run, ds)
        if seqs0 is None:
            seqs0, first = seqs, v[:, :3]
        assert seqs == seqs0, (ds, run)
        assert np.allclose(v[:, :3], first), 'first.pth columns differ between runs: ' + ds + ' ' + run
        auc[run] = v[:, 3:6]
    auc['first'] = first
    print('\n=== {} ({} sequences); mean success AUC none / gamma 0.6 / anchor+gamma'.format(ds, len(seqs0)))
    for k in ['first'] + RUNS:
        print('   {:5s} {}'.format(k, ' / '.join('{:.4f}'.format(x) for x in auc[k].mean(axis=0))))
    for si, st in enumerate(SETTINGS):
        print('  -- setting {}'.format(st))
        for grp, a, b in COMPARE:
            m, lo, hi, better, worse, eq, n = boot(auc[a][:, si], auc[b][:, si])
            verdict = 'differs' if (lo > 0 or hi < 0) else 'within noise'
            print('     {}-{:5s} {:+.4f} [{:+.4f}, {:+.4f}]  better {:3d} worse {:3d} equal {:2d}  {}'.format(a, b, m, lo, hi, better, worse, eq, verdict))
            out.append([ds, st, grp, '{} - {}'.format(a, b), n, m, lo, hi, better, worse, eq, verdict])

with open(os.path.join(OUT, 'E16_ablation_bootstrap.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset', 'setting', 'group', 'comparison', 'n_sequences', 'mean_diff', 'lo', 'hi', 'better', 'worse', 'equal', 'verdict'])
    w.writerows(out)
print('\nwrote', os.path.join(OUT, 'E16_ablation_bootstrap.csv'), len(out), 'rows;', sum(1 for r in out if r[-1] == 'differs'), 'intervals exclude 0')

print('\n=== NAT2021, anchor+gamma, epochs 8 / 9 / 10 of B and C (per-sequence results of results/NAT2021/checkpoint_e<K>1.2_<RUN>_ag)')
for run in ('B', 'C'):
    for k in (8, 9, 10):
        p = os.path.join('results', 'NAT2021', 'checkpoint_e{}1.2_{}_ag'.format(k, run))
        print('   {} epoch {:2d}: {}'.format(run, k, 'exists' if os.path.isdir(p) else 'MISSING (no NAT2021 tracking run was made for this checkpoint)'))
