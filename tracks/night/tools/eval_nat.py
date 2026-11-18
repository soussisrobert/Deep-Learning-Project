import os
import sys
import csv
import argparse
sys.path.append("./")

import numpy as np

from toolkit.datasets import NATDataset
from toolkit.evaluation import OPEBenchmark


def paired_bootstrap(a, b, n=2000, seed=0):
    """mean(a - b) with a 95% bootstrap interval, resampling sequences."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), int((d > 0).sum()), len(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAT2021 evaluation')
    parser.add_argument('--dataset', default='NAT2021', type=str)
    parser.add_argument('--dataset_root', default='../test_dataset', type=str)
    parser.add_argument('--tracker_path', default='results', type=str)
    parser.add_argument('--tracker_prefix', default='first', type=str)
    parser.add_argument('--trackers', nargs='+', default=None,
                        help='exact result folder names, overrides --tracker_prefix')
    parser.add_argument('--show_video_level', action='store_true')
    parser.add_argument('--every', default=1, type=int,
                        help='evaluate only every Nth video (sorted by name), same rule as test.py')
    parser.add_argument('--video_list', default='', type=str,
                        help='text file with one sequence name per line; evaluate only those')
    parser.add_argument('--compare', action='store_true',
                        help='paired bootstrap of per-sequence success AUC between every pair of trackers')
    parser.add_argument('--csv', default='', type=str,
                        help='write per-sequence success AUC (rows: sequences, columns: trackers) here')
    args = parser.parse_args()

    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    if args.trackers:
        trackers = list(args.trackers)
    else:
        trackers = sorted(x for x in os.listdir(tracker_dir)
                          if x.startswith(args.tracker_prefix))
    assert len(trackers) > 0, 'no results in ' + tracker_dir

    dataset = NATDataset(args.dataset, os.path.join(args.dataset_root, args.dataset))
    if args.every > 1:
        keep = sorted(dataset.videos.keys())[::args.every]
        dataset.videos = {k: dataset.videos[k] for k in keep}
        print('evaluating {} of the sequences (every {}th)'.format(len(keep), args.every))
    if args.video_list:
        names = set(l.strip() for l in open(args.video_list) if l.strip())
        dataset.videos = {k: v for k, v in dataset.videos.items() if k in names}
        print('evaluating {} sequences from {}'.format(len(dataset.videos), args.video_list))
    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    for t in trackers:
        success_ret.update(benchmark.eval_success(t))
    precision_ret = {}
    for t in trackers:
        precision_ret.update(benchmark.eval_precision(t))
    norm_precision_ret = {}
    for t in trackers:
        norm_precision_ret.update(benchmark.eval_norm_precision(t))
    benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                          show_video_level=args.show_video_level)

    seqs = sorted(success_ret[trackers[0]].keys())
    auc = {t: np.array([np.mean(success_ret[t][s]) for s in seqs]) for t in trackers}

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seq'] + trackers)
            for i, s in enumerate(seqs):
                w.writerow([s] + ['{:.4f}'.format(auc[t][i]) for t in trackers])
        print('wrote', args.csv)

    if args.compare and len(trackers) > 1:
        print('\npaired comparison of per-sequence success AUC over {} sequences'.format(len(seqs)))
        print('mean difference (A - B), 95% bootstrap interval, sequences where A > B')
        for i, a in enumerate(trackers):
            for b in trackers[i + 1:]:
                m, lo, hi, wins, n = paired_bootstrap(auc[a], auc[b])
                verdict = 'differs' if (lo > 0 or hi < 0) else 'within noise'
                print('{:>28s} - {:<28s} {:+.3f}  [{:+.3f}, {:+.3f}]  {}/{}  {}'.format(
                    a, b, m, lo, hi, wins, n, verdict))
