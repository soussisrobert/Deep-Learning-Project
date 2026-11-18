"""E12: (1) selection check for gamma, (2) per-attribute and per-darkness breakdown of the gains.

Uses only existing per-sequence AUC files, no tracking runs.
(1) Sequences are sorted by name; index 0, 2, 4, ... = "even", 1, 3, 5, ... = "odd". The gamma with the best mean AUC
    on one half is chosen, then reported on the other half (both directions).
(2) Gains (anchor + gamma 0.6 minus baseline, gamma 0.6 minus baseline, anchor minus gamma) by attribute flag
    (anno/att/<seq>.txt, columns ATT01..ATT12; the column names are NOT verified, see experiments_log.md) and by
    quantile of mean gray level per sequence.
Bootstrap: paired, 2000 resamples over sequences, 95% percentile interval (same as eval_nat.py --compare).
"""
import os
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--e4', default='illumination/E4_per_sequence_auc.csv')
parser.add_argument('--e10', default='illumination/E10_per_sequence_auc.csv')
parser.add_argument('--illum', default='illumination/NAT2021_illumination.csv')
parser.add_argument('--att_dir', default='../test_dataset/NAT2021/anno/att')
parser.add_argument('--out_csv', default='illumination/E12_attribute_breakdown.csv')
parser.add_argument('--n_boot', type=int, default=2000)
args = parser.parse_args()

rng = np.random.RandomState(0)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def boot(d, n_boot=args.n_boot):
    """mean and 95% interval of the mean of d (1-D array), resampling sequences"""
    d = np.asarray(d, dtype=float)
    n = len(d)
    idx = rng.randint(0, n, size=(n_boot, n))
    m = d[idx].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def verdict(lo, hi):
    return 'differs' if (lo > 0 or hi < 0) else 'within noise'


def boot_diff_groups(d, mask, n_boot=args.n_boot):
    """mean(d[mask]) - mean(d[~mask]), resampling sequences (group sizes vary per resample)"""
    d = np.asarray(d, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = len(d)
    out = []
    for _ in range(n_boot):
        i = rng.randint(0, n, size=n)
        a, b = d[i][mask[i]], d[i][~mask[i]]
        if len(a) and len(b):
            out.append(a.mean() - b.mean())
    out = np.array(out)
    return d[mask].mean() - d[~mask].mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


# ---------------------------------------------------------------- (1) gamma selection
e4 = read_csv(args.e4)
e4.sort(key=lambda r: r['seq'])
seqs = [r['seq'] for r in e4]
base = np.array([float(r['first1.2']) for r in e4])
gam = {
    '0.4': np.array([float(r['first1.2_gamma_both_180']) for r in e4]),
    '0.6': np.array([float(r['first1.2_gamma0.6_180']) for r in e4]),
    '0.8': np.array([float(r['first1.2_gamma0.8_180']) for r in e4]),
}
n = len(seqs)
even = np.arange(n) % 2 == 0
odd = ~even
print('=== (1) gamma selection check, {} sequences sorted by name, even = indices 0,2,4,... ({}), odd = {}'.format(
    n, even.sum(), odd.sum()))
print('all sequences: baseline {:.4f}'.format(base.mean()) + ''.join(
    ', gamma {} {:.4f}'.format(g, v.mean()) for g, v in gam.items()))
rows_out = []
for name, sel, rep in (('select on even, report on odd', even, odd), ('select on odd, report on even', odd, even)):
    print('\n-- ' + name)
    print('{:>8} | {:>12} | {:>12}'.format('gamma', 'mean (select)', 'mean (report)') + ' | baseline (report) {:.4f}'.format(base[rep].mean()))
    for g, v in gam.items():
        print('{:>8} | {:>12.4f} | {:>12.4f}'.format(g, v[sel].mean(), v[rep].mean()))
    best = max(gam, key=lambda g: gam[g][sel].mean())
    m, lo, hi = boot(gam[best][rep] - base[rep])
    print('chosen on the selection half: gamma {}'.format(best))
    print('reported half: gamma {} minus baseline = {:+.4f} [{:+.4f}, {:+.4f}] ({}), better on {}/{}'.format(
        best, m, lo, hi, verdict(lo, hi), int((gam[best][rep] > base[rep]).sum()), int(rep.sum())))
    for g in gam:
        if g != best:
            m2, lo2, hi2 = boot(gam[best][rep] - gam[g][rep])
            print('   gamma {} minus gamma {} on the reported half = {:+.4f} [{:+.4f}, {:+.4f}] ({})'.format(
                best, g, m2, lo2, hi2, verdict(lo2, hi2)))
    rows_out.append(['selection', name, best, '{:.4f}'.format(m), '{:.4f}'.format(lo), '{:.4f}'.format(hi), int(rep.sum())])

# ---------------------------------------------------------------- (2) breakdown
e10 = {r['seq']: r for r in read_csv(args.e10)}
illum = {r['seq']: r for r in read_csv(args.illum)}
seqs10 = sorted(e10)
B = np.array([float(e10[s]['first1.2']) for s in seqs10])
G = np.array([float(e10[s]['first1.2_gamma0.6_180']) for s in seqs10])
A = np.array([float(e10[s]['first1.2_p15_a0.2_180']) for s in seqs10])
gray = np.array([float(illum[s]['gray_mean']) for s in seqs10])
att = np.zeros((len(seqs10), 12), dtype=int)
for i, s in enumerate(seqs10):
    with open(os.path.join(args.att_dir, s + '.txt')) as f:
        att[i] = [int(x) for x in f.read().strip().split(',')]

print('\n=== (2) per-attribute breakdown, {} sequences (E10 per-sequence AUCs); column names unverified'.format(len(seqs10)))
print('gain columns: gamma = gamma 0.6 minus baseline; anchor = anchor 0.2 + gamma 0.6 minus baseline; '
      'in-vs-out = gain inside the attribute minus gain outside it')
hdr = '{:>6} {:>4} {:>7} | {:>7} {:>7} | {:>8} {:>18} | {:>8} {:>18} | {:>8} {:>18}'.format(
    'attr', 'n', 'base', 'gamma', 'anchor', 'gain g', '95% (gamma)', 'gain a', '95% (anchor)', 'in-vs-out', '95% (anchor)')
print(hdr)


def row(label, mask):
    nn = int(mask.sum())
    mg, lg, hg = boot((G - B)[mask])
    ma, la, ha = boot((A - B)[mask])
    if 0 < nn < len(mask):
        mi, li, hi_ = boot_diff_groups(A - B, mask)
    else:
        mi = li = hi_ = float('nan')
    print('{:>6} {:>4} {:>7.3f} | {:>7.3f} {:>7.3f} | {:>+8.3f} [{:>+6.3f},{:>+6.3f}] {:>1} | {:>+8.3f} [{:>+6.3f},{:>+6.3f}] {:>1} | {:>+8.3f} [{:>+6.3f},{:>+6.3f}] {:>1}'.format(
        label, nn, B[mask].mean(), G[mask].mean(), A[mask].mean(),
        mg, lg, hg, '*' if lg > 0 or hg < 0 else '',
        ma, la, ha, '*' if la > 0 or ha < 0 else '',
        mi, li, hi_, '*' if (li > 0 or hi_ < 0) else ''))
    rows_out.append(['breakdown', label, nn, '{:.4f}'.format(B[mask].mean()), '{:.4f}'.format(G[mask].mean()),
                     '{:.4f}'.format(A[mask].mean()), '{:.4f}'.format(mg), '{:.4f}'.format(lg), '{:.4f}'.format(hg),
                     '{:.4f}'.format(ma), '{:.4f}'.format(la), '{:.4f}'.format(ha),
                     '{:.4f}'.format(mi), '{:.4f}'.format(li), '{:.4f}'.format(hi_)])


row('ALL', np.ones(len(seqs10), dtype=bool))
for k in range(12):
    row('ATT{:02d}'.format(k + 1), att[:, k] == 1)
print('(* = 95% interval excludes 0. 12 attributes x 3 intervals are looked at, so a few stars can appear by chance.)')

print('\n--- by darkness (mean gray level per sequence, quantile groups; no attribute names needed)')
print(hdr.replace('attr', 'group'))
qs = np.percentile(gray, [100 / 3.0, 200 / 3.0])
groups = [('dark', gray <= qs[0]), ('mid', (gray > qs[0]) & (gray <= qs[1])), ('bright', gray > qs[1])]
for label, mask in groups:
    row(label, mask)
print('tertile cut points of gray_mean: {:.1f}, {:.1f}'.format(qs[0], qs[1]))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return np.corrcoef(rx, ry)[0, 1]


for name, d in (('gamma gain', G - B), ('anchor gain', A - B)):
    r = spearman(gray, d)
    idx = rng.randint(0, len(d), size=(args.n_boot, len(d)))
    rb = np.array([spearman(gray[i], d[i]) for i in idx])
    print('Spearman(gray_mean, {}) = {:+.3f} [{:+.3f}, {:+.3f}]'.format(name, r, np.percentile(rb, 2.5), np.percentile(rb, 97.5)))

with open(args.out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['kind', 'label', 'n_or_gamma', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12'])
    for r in rows_out:
        w.writerow(r)
print('\nwrote ' + args.out_csv)
