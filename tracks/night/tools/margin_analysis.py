"""E13 analysis: do two runs split at frames where the score map has a near tie?

Inputs (from job_margin.sh): results/<dataset>/<name>/<seq>.txt (boxes) and margin/<dataset>/<name>/<seq>.csv (log).
For two runs A and B and one sequence:
  split frame = first frame t >= 1 where IoU(box_A[t], box_B[t]) < thr (default 0.5)
  margin      = (p1 - p2d) / p1, with p1 the top-1 value of the final score map and p2d the best value among cells at least
                2 grid cells away from it (a different peak); margin_a uses p2a (second highest anywhere).
Tests (all over sequences, bootstrap over the split events):
  T1  percentile of the split frame's margin among the same sequence's earlier frames (before the split), using
      min(margin_A, margin_B). Under "the split is unrelated to the margin" the mean percentile is 0.5; near ties
      would give a value well below 0.5.
  T2  mean margin at the split frame vs mean margin over all earlier frames (A, B separately).
  T3  does the other run pick the first run's second peak? Distance in image coordinates between B's top-1 centre and
      A's top-1 / top-2 centre, and the same with A and B swapped. A split caused by a near tie should show B's top-1 close
      to A's top-2d (and A's top-1 close to B's top-2d).
  T4  margin in the 5 frames before the split vs earlier frames (is a near tie visible before the runs part?).
"""
import os
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NAT2021')
parser.add_argument('--runs', nargs='+', default=['first1.2_m_base', 'first1.2_m_g0.6', 'first1.2_m_anchor'])
parser.add_argument('--thr', nargs='+', type=float, default=[0.5, 0.3, 0.7])
parser.add_argument('--min_prefix', type=int, default=20, help='split events with fewer earlier frames are skipped in T1')
parser.add_argument('--n_boot', type=int, default=2000)
parser.add_argument('--key', default='mx', choices=['mx', 'm'],
                    help="mx: second peak = best cell whose decoded box centre is >= 0.3 x target size from the top-1 "
                         "(image space, 1.0 if none); m: best cell >= 2 grid cells away (often the same box)")
parser.add_argument('--out_csv', default='illumination/E13_split_events.csv')
args = parser.parse_args()
KEY = args.key
SUF = '2x' if KEY == 'mx' else '2d'
rng = np.random.RandomState(0)


def load_boxes(run, seq):
    return np.loadtxt(os.path.join('results', args.dataset, run, seq + '.txt'), delimiter=',')


def load_margin(run, seq):
    with open(os.path.join('margin', args.dataset, run, seq + '.csv')) as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    cols['m'] = (cols['p1'] - cols['p2d']) / cols['p1']
    cols['mx'] = (cols['p1'] - cols['p2x']) / cols['p1']
    cols['ma'] = (cols['p1'] - cols['p2a']) / cols['p1']
    return cols


def iou(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def boot_mean(x):
    x = np.asarray(x, dtype=float)
    idx = rng.randint(0, len(x), size=(args.n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


seqs = sorted(f[:-4] for f in os.listdir(os.path.join('results', args.dataset, args.runs[0])) if f.endswith('.txt'))
print('{} sequences, runs: {}'.format(len(seqs), args.runs))

# overall margin distribution per run (all tracked frames)
allm = {}
for r in args.runs:
    allm[r] = np.concatenate([load_margin(r, s)[KEY] for s in seqs])
    q = np.percentile(allm[r], [5, 10, 25, 50, 75, 90])
    print('{}: margin (p1 - p2d)/p1 over {} frames, percentiles 5/10/25/50/75/90 = {}'.format(
        r, len(allm[r]), ' '.join('{:.3f}'.format(v) for v in q)))

pairs = [(a, b) for i, a in enumerate(args.runs) for b in args.runs[i + 1:]]
events_out = []
for a, b in pairs:
    for thr in args.thr:
        ev = []
        n_never = 0
        for s in seqs:
            ba, bb = load_boxes(a, s), load_boxes(b, s)
            n = min(len(ba), len(bb))
            i = iou(ba[:n], bb[:n])
            below = np.where(i[1:] < thr)[0]
            if len(below) == 0:
                n_never += 1
                continue
            t = int(below[0]) + 1
            ma, mb = load_margin(a, s), load_margin(b, s)
            k = t - 1  # margin rows start at frame 1
            pre = np.arange(0, k)
            ev.append(dict(seq=s, t=t, n_pre=len(pre),
                           mA=ma[KEY][k], mB=mb[KEY][k],
                           preA=ma[KEY][pre].mean() if len(pre) else np.nan,
                           preB=mb[KEY][pre].mean() if len(pre) else np.nan,
                           pct=(np.mean(np.minimum(ma[KEY][pre], mb[KEY][pre]) <= min(ma[KEY][k], mb[KEY][k]))
                                if len(pre) >= args.min_prefix else np.nan),
                           last5A=ma[KEY][max(0, k - 5):k].mean() if k > 0 else np.nan,
                           last5B=mb[KEY][max(0, k - 5):k].mean() if k > 0 else np.nan,
                           # T3: centres in image coordinates
                           dB1_A1=np.hypot(mb['cx1'][k] - ma['cx1'][k], mb['cy1'][k] - ma['cy1'][k]),
                           dB1_A2=np.hypot(mb['cx1'][k] - ma['cx' + SUF][k], mb['cy1'][k] - ma['cy' + SUF][k]),
                           dA1_B2=np.hypot(ma['cx1'][k] - mb['cx' + SUF][k], ma['cy1'][k] - mb['cy' + SUF][k]),
                           tgt=float(np.sqrt(ba[t, 2] * ba[t, 3]))))
        print('\n=== {} vs {}, split = first frame with IoU < {}: {} of {} sequences split, {} never'.format(
            a, b, thr, len(ev), len(seqs), n_never))
        if len(ev) < 5:
            continue
        ts = np.array([e['t'] for e in ev])
        print('split frame: median {:.0f}, 25-75% {:.0f}-{:.0f}'.format(np.median(ts), *np.percentile(ts, [25, 75])))
        pct = np.array([e['pct'] for e in ev if not np.isnan(e['pct'])])
        if len(pct) >= 5:
            m, lo, hi = boot_mean(pct)
            print('T1 percentile of min(mA, mB) at the split among earlier frames (0.5 = unrelated): {:.3f} [{:.3f}, {:.3f}] over {} events with >= {} earlier frames'.format(
                m, lo, hi, len(pct), args.min_prefix))
        for nm, key_s, key_p in (('A', 'mA', 'preA'), ('B', 'mB', 'preB')):
            d = np.array([e[key_s] - e[key_p] for e in ev if not np.isnan(e[key_p])])
            m, lo, hi = boot_mean(d)
            print('T2 run {}: margin at split minus mean margin of earlier frames = {:+.4f} [{:+.4f}, {:+.4f}] (n {})'.format(
                nm, m, lo, hi, len(d)))
        for nm, key_5, key_p in (('A', 'last5A', 'preA'), ('B', 'last5B', 'preB')):
            d = np.array([e[key_5] - e[key_p] for e in ev if not np.isnan(e[key_p]) and not np.isnan(e[key_5])])
            m, lo, hi = boot_mean(d)
            print('T4 run {}: mean margin of the 5 frames before the split minus mean of all earlier frames = {:+.4f} [{:+.4f}, {:+.4f}]'.format(
                nm, m, lo, hi))
        tg = np.array([e['tgt'] for e in ev])
        d11 = np.array([e['dB1_A1'] for e in ev])
        d12 = np.array([e['dB1_A2'] for e in ev])
        d21 = np.array([e['dA1_B2'] for e in ev])
        print('T3 distance (px) between the chosen centres of the two runs at the split: median {:.1f}; target size median {:.1f}'.format(
            np.median(d11), np.median(tg)))
        v12, v21 = ~np.isnan(d12), ~np.isnan(d21)
        print('T3 (second peak = {}; events where that peak exists: {} / {} for A, {} / {} for B)'.format(
            SUF, int(v12.sum()), len(ev), int(v21.sum()), len(ev)))
        if v12.sum() and v21.sum():
            print('T3 B top-1 is nearer to A second peak than to A top-1 in {:.1%} of those events; A top-1 nearer to B second peak than to B top-1 in {:.1%}'.format(
                np.mean(d12[v12] < d11[v12]), np.mean(d21[v21] < d11[v21])))
            print('T3 median distance B top-1 to A second peak: {:.1f} px; A top-1 to B second peak: {:.1f} px'.format(
                np.median(d12[v12]), np.median(d21[v21])))
        for e in ev:
            events_out.append([a, b, thr, e['seq'], e['t'], '{:.4f}'.format(e['mA']), '{:.4f}'.format(e['mB']),
                               '{:.4f}'.format(e['preA']), '{:.4f}'.format(e['preB']), '{:.3f}'.format(e['pct']),
                               '{:.1f}'.format(e['dB1_A1']), '{:.1f}'.format(e['dB1_A2']), '{:.1f}'.format(e['dA1_B2'])])

with open(args.out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run_a', 'run_b', 'thr', 'seq', 'split_frame', 'margin_a', 'margin_b', 'mean_margin_before_a',
                'mean_margin_before_b', 'percentile_among_earlier', 'dist_b1_a1', 'dist_b1_a2d', 'dist_a1_b2d'])
    w.writerows(events_out)
print('\nwrote ' + args.out_csv)
