"""Which sequences did enhancement help? Joins per-sequence success AUC (E2) with mean brightness.

Groups by change in success AUC versus the baseline:
  improved   delta >  +tol
  unchanged  |delta| <= tol
  worse      delta <  -tol
"""
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID = '#1f2933', '#6b7785', '#e3e7ec'
COL = {'improved': '#2563c9', 'unchanged': '#9aa5b1', 'worse': '#c2410c'}


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auc_csv', default='illumination/E2_per_sequence_auc.csv')
    parser.add_argument('--illum_csv', default='illumination/NAT2021_illumination.csv')
    parser.add_argument('--baseline', default='first1.2')
    parser.add_argument('--variants', nargs='+',
                        default=['first1.2_gamma_both_180', 'first1.2_clahe_both_180', 'first1.2_gc_both_180'])
    parser.add_argument('--tol', type=float, default=0.01, help='AUC change treated as "no real change"')
    parser.add_argument('--out', default='illumination')
    args = parser.parse_args()

    auc = {}
    for r in csv.DictReader(open(args.auc_csv)):
        auc[r['seq']] = {k: float(v) for k, v in r.items() if k != 'seq'}
    gray = {r['seq']: float(r['gray_mean']) for r in csv.DictReader(open(args.illum_csv))}
    seqs = sorted(s for s in auc if s in gray)
    g = np.array([gray[s] for s in seqs])
    base = np.array([auc[s][args.baseline] for s in seqs])
    print('{} sequences, tolerance +/-{}'.format(len(seqs), args.tol))

    for v in args.variants:
        d = np.array([auc[s][v] for s in seqs]) - base
        grp = np.where(d > args.tol, 'improved', np.where(d < -args.tol, 'worse', 'unchanged'))
        print('\n=== {} vs {} ==='.format(v, args.baseline))
        print('sum of deltas: gains {:+.2f}, losses {:+.2f}, net {:+.2f}  (mean {:+.4f})'.format(
            d[d > 0].sum(), d[d < 0].sum(), d.sum(), d.mean()))
        print('improved (delta > 0): {}   not improved (delta <= 0): {}   exactly 0: {}'.format(
            int((d > 0).sum()), int((d <= 0).sum()), int((d == 0).sum())))
        for name in ('improved', 'unchanged', 'worse'):
            m = grp == name
            if m.sum() == 0:
                print('{:10s} n=0'.format(name))
                continue
            print('{:10s} n={:3d}  mean delta {:+.3f}  median {:+.3f}  min {:+.3f}  max {:+.3f}  | mean gray {:.1f} (median {:.1f})  baseline AUC {:.3f}'.format(
                name, int(m.sum()), d[m].mean(), np.median(d[m]), d[m].min(), d[m].max(),
                g[m].mean(), np.median(g[m]), base[m].mean()))
        npos = d > 0
        nonimp = ~npos
        print('not improved split: {} slightly down or same (|delta|<=tol incl. exact ties), {} clearly worse (delta < -tol)'.format(
            int(((d <= 0) & (d >= -args.tol)).sum()), int((d < -args.tol).sum())))
        rho = np.corrcoef(rank(g), rank(d))[0, 1]
        print('Spearman(mean gray, delta AUC) = {:.3f}'.format(rho))
        q = np.argsort(g)
        k = len(q) // 4
        print('darkest quarter mean delta {:+.3f} | brightest quarter mean delta {:+.3f}'.format(
            d[q[:k]].mean(), d[q[-k:]].mean()))

        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2), dpi=130,
                                         gridspec_kw={'width_ratios': [1, 1.25, 1]})
        bins = np.linspace(min(d.min(), -0.3), max(d.max(), 0.3), 31)
        for name in ('worse', 'unchanged', 'improved'):
            a1.hist(d[grp == name], bins=bins, color=COL[name], edgecolor='white', linewidth=0.6,
                    label='{} ({})'.format(name, int((grp == name).sum())))
        a1.axvline(0, color=INK, linewidth=1)
        a1.set_xlabel('Change in success AUC (enhanced - baseline)', color=MUTED, fontsize=9)
        a1.set_ylabel('Sequences', color=MUTED, fontsize=9)
        a1.set_title('How much each sequence changed', color=INK, fontsize=10.5, loc='left')
        a1.legend(frameon=False, fontsize=8)
        style(a1)

        for name in ('worse', 'unchanged', 'improved'):
            m = grp == name
            a2.scatter(g[m], d[m], s=22, color=COL[name], edgecolor='white', linewidth=0.6, alpha=0.95)
        a2.axhline(0, color=INK, linewidth=1)
        a2.axhspan(-args.tol, args.tol, color=GRID, alpha=0.6, zorder=0)
        a2.set_xlabel('Sequence mean gray (0-255)', color=MUTED, fontsize=9)
        a2.set_ylabel('Change in success AUC', color=MUTED, fontsize=9)
        a2.set_title('Change vs. brightness (Spearman {:.2f})'.format(rho), color=INK, fontsize=10.5, loc='left')
        style(a2)

        data = [g[grp == n] for n in ('improved', 'unchanged', 'worse') if (grp == n).sum()]
        names = [n for n in ('improved', 'unchanged', 'worse') if (grp == n).sum()]
        bp = a3.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker='o', markersize=3, markeredgecolor=MUTED))
        a3.set_xticks(range(1, len(names) + 1))
        a3.set_xticklabels(['{}\n(n={})'.format(n, len(x)) for n, x in zip(names, data)])
        for patch, n in zip(bp['boxes'], names):
            patch.set_facecolor(COL[n])
            patch.set_alpha(0.85)
            patch.set_edgecolor('white')
        a3.set_ylabel('Sequence mean gray (0-255)', color=MUTED, fontsize=9)
        a3.set_title('Brightness of each group', color=INK, fontsize=10.5, loc='left')
        style(a3)

        fig.suptitle('{}: effect on {} sequences (tolerance +/-{})'.format(v, len(seqs), args.tol),
                     color=INK, fontsize=11.5, x=0.01, ha='left')
        fig.patch.set_facecolor('white')
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = os.path.join(args.out, 'improvement_{}.png'.format(v))
        fig.savefig(out, facecolor='white')
        plt.close(fig)
        print('saved', out)

        with open(os.path.join(args.out, 'improvement_{}.csv'.format(v)), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seq', 'mean_gray', 'baseline_auc', 'enhanced_auc', 'delta', 'group'])
            for i, s in enumerate(seqs):
                w.writerow([s, round(g[i], 2), round(base[i], 4), round(auc[s][v], 4), round(d[i], 4), grp[i]])
