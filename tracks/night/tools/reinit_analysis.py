"""Does resetting the tracker to the ground truth every N frames remove the sensitivity to tiny input changes?

For each condition (no reset / reset every 200 / reset every 50) there are three runs on the same 40 sequences:
baseline, gamma 0.95 (almost no change) and gamma 0.6 (the best setting). The tracker is deterministic, so any
difference between the baseline run and a gamma run of the same condition comes from the input change.
Reported per condition: mean success AUC, mean |dAUC| between baseline and the gamma run, share of sequences
with |dAUC| > 0.05, and how many sequences' boxes split by more than 50 px (sum |dx| + |dy| of the top-left corner)
at some point and at the end.
"""
import os
import csv
import argparse

import numpy as np

CONDITIONS = [
    ('no reset', 'first1.2', 'first1.2_f_gamma0.95', 'first1.2_f_gamma0.6'),
    ('reset every 200', 'first1.2_r200_base', 'first1.2_r200_g0.95', 'first1.2_r200_g0.6'),
    ('reset every 50', 'first1.2_r50_base', 'first1.2_r50_g0.95', 'first1.2_r50_g0.6'),
]


def load(path):
    return np.array([[float(v) for v in l.strip().split(',')] for l in open(path) if l.strip()])


def split_stats(base_dir, run_dir, seqs):
    ever = end = 0
    for s in seqs:
        a, b = load(os.path.join(base_dir, s + '.txt')), load(os.path.join(run_dir, s + '.txt'))
        n = min(len(a), len(b))
        d = np.abs(a[:n, 0] - b[:n, 0]) + np.abs(a[:n, 1] - b[:n, 1])
        ever += int((d > 50).any())
        end += int(d[-1] > 50)
    return ever, end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auc_csv', default='illumination/E6_per_sequence_auc.csv')
    parser.add_argument('--list', default='illumination/E3c_all_list.txt')
    parser.add_argument('--results', default='results/NAT2021')
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.auc_csv)))
    seqs = [r['seq'] for r in rows]
    auc = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'seq'}
    n = len(seqs)

    hdr = '{:18s} {:8s} {:>9s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s}'.format(
        'condition', 'run', 'mean AUC', 'mean|dAUC|', '|dAUC|>.05', '|dAUC|>.2', 'split>50px', 'end>50px')
    print(hdr)
    print('-' * len(hdr))
    for name, base, g95, g6 in CONDITIONS:
        if base not in auc:
            print('{:18s} (missing runs)'.format(name))
            continue
        print('{:18s} {:8s} {:9.3f}'.format(name, 'baseline', auc[base].mean()))
        for label, run in (('gamma .95', g95), ('gamma .6', g6)):
            d = auc[run] - auc[base]
            ever, end = split_stats(os.path.join(args.results, base), os.path.join(args.results, run), seqs)
            print('{:18s} {:8s} {:9.3f} {:10.3f} {:10.0%} {:9.0%} {:7d}/{} {:7d}/{}'.format(
                '', label, auc[run].mean(), np.abs(d).mean(), (np.abs(d) > 0.05).mean(),
                (np.abs(d) > 0.2).mean(), ever, n, end, n))
        print()
    print('Reading it: if resetting to the ground truth makes |dAUC| and the splits shrink a lot, the feedback loop')
    print('(each frame\'s box decides the next crop) is a main source of the sensitivity. AUC values go up because the')
    print('ground truth is used: this is a diagnostic, not a usable tracker.')
