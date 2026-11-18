"""Fragility test: how much does each sequence's success AUC move under different input perturbations?

Input: per-sequence AUC csv from eval_nat.py (rows: sequences, columns: runs) and two lists of
sequences: the 'improved' half and the 'worse' half (both defined by the gamma 0.4 run).
For every run, change in AUC versus the baseline is computed per sequence and summarised per half.
"""
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID = '#1f2933', '#6b7785', '#e3e7ec'
C_IMP, C_WORSE = '#2563c9', '#c2410c'

# (result folder name, label); only runs present in the csv are used.
RUNS = [
    ('first1.2_f_repeat_baseline', 'repeat of\nbaseline'),
    ('first1.2_f_gamma0.95', 'gamma\n0.95'),
    ('first1.2_f_gamma0.9', 'gamma\n0.9'),
    ('first1.2_f_gamma0.8', 'gamma\n0.8'),
    ('first1.2_f_gamma0.6', 'gamma\n0.6'),
    ('first1.2_f_gamma0.5', 'gamma\n0.5'),
    ('first1.2_gamma_both_180', 'gamma\n0.4 *'),
    ('first1.2_f_clahe1.0', 'CLAHE\nclip 1'),
    ('first1.2_f_clahe2.0', 'CLAHE\nclip 2'),
    ('first1.2_clahe_both_180', 'CLAHE\nclip 3'),
    ('first1.2_f_clahe4.0', 'CLAHE\nclip 4'),
]


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auc_csv', default='illumination/E3c_per_sequence_auc.csv')
    parser.add_argument('--improved_list', default='illumination/E3c_improved_list.txt')
    parser.add_argument('--worse_list', default='illumination/E3c_worse_list.txt')
    parser.add_argument('--baseline', default='first1.2')
    parser.add_argument('--out', default='illumination')
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.auc_csv)))
    seqs = [r['seq'] for r in rows]
    auc = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'seq'}
    imp = np.array([s in set(read_list(args.improved_list)) for s in seqs])
    wor = np.array([s in set(read_list(args.worse_list)) for s in seqs])
    base = auc[args.baseline]
    runs = [(n, l) for n, l in RUNS if n in auc]
    sel = auc['first1.2_gamma_both_180']
    print('{} sequences: {} from the improved half, {} from the worse half'.format(len(seqs), imp.sum(), wor.sum()))
    print('* = the run that defined the halves (gamma 0.4), so its halves are separated by construction\n')

    hdr = '{:28s} {:>9s} {:>9s} {:>9s} {:>8s} {:>8s} {:>8s} {:>9s}'.format(
        'run', 'meanD imp', 'meanD wor', 'mean|D|', 'std D', '|D|>.05', '|D|>.2', 'corr w/0.4')
    print(hdr)
    print('-' * len(hdr))
    stats = {}
    for n, l in runs:
        d = auc[n] - base
        corr = np.corrcoef(d, sel - base)[0, 1] if d.std() > 0 else float('nan')
        stats[n] = d
        print('{:28s} {:+9.3f} {:+9.3f} {:9.3f} {:8.3f} {:7.0%} {:7.0%} {:9.2f}'.format(
            n, d[imp].mean(), d[wor].mean(), np.abs(d).mean(), d.std(),
            (np.abs(d) > 0.05).mean(), (np.abs(d) > 0.2).mean(), corr))

    print('\nmeanD imp / wor: mean change in AUC vs baseline in the improved / worse half.')
    print('If the halves keep their sign under other settings, the effect belongs to the sequence;')
    print('if both go to about 0 (or flip), the halves were mostly luck of the 0.4 run.')
    print('mean|D|, |D|>.05, |D|>.2: how much sequences move regardless of direction (fragility).')
    print('corr w/0.4: correlation of the per-sequence change with the change under gamma 0.4.')

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8), dpi=130, gridspec_kw={'width_ratios': [2.3, 1]})
    rng = np.random.RandomState(0)
    for i, (n, l) in enumerate(runs):
        d = stats[n]
        for m, c, off in ((imp, C_IMP, -0.17), (wor, C_WORSE, 0.17)):
            x = i + off + rng.uniform(-0.09, 0.09, m.sum())
            a1.scatter(x, d[m], s=14, color=c, edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
            a1.plot([i + off - 0.13, i + off + 0.13], [d[m].mean()] * 2, color=INK, linewidth=2, zorder=4)
    a1.axhline(0, color=INK, linewidth=1)
    a1.set_xticks(range(len(runs)))
    a1.set_xticklabels([l for _, l in runs], fontsize=8)
    a1.set_ylabel('Change in success AUC vs. baseline', color=MUTED, fontsize=9)
    a1.set_title('Per-sequence change under each setting (bar = half mean)', color=INK, fontsize=10.5, loc='left')
    a1.scatter([], [], s=18, color=C_IMP, label='half that improved at gamma 0.4 (n={})'.format(imp.sum()))
    a1.scatter([], [], s=18, color=C_WORSE, label='half that got worse at gamma 0.4 (n={})'.format(wor.sum()))
    a1.legend(frameon=False, fontsize=8, loc='upper right')

    mad = [np.abs(stats[n]).mean() for n, _ in runs]
    a2.bar(range(len(runs)), mad, color='#2563c9', width=0.6)
    a2.set_xticks(range(len(runs)))
    a2.set_xticklabels([l for _, l in runs], fontsize=7)
    a2.set_ylabel('Mean |change in AUC|', color=MUTED, fontsize=9)
    a2.set_title('How much sequences move, either way', color=INK, fontsize=10.5, loc='left')
    for a in (a1, a2):
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, length=0)
        a.grid(axis='y', color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
    fig.suptitle('Fragility test on {} sampled sequences (* = run that defined the halves)'.format(len(seqs)),
                 color=INK, fontsize=11.5, x=0.01, ha='left')
    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(args.out, 'fragility.png')
    fig.savefig(out, facecolor='white')
    print('saved', out)
