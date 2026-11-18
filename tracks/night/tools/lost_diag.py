"""E15: diagnostic for re-detection. Reads existing result boxes, E13 score logs and GT only; the tracker is not touched.

Tracker: first.pth + anchor 0.2 + gamma 0.6 (first_ag), the boxes and margin logs of the margin run named first1.2_m_anchor.
Datasets: NAT2021 (180), NATUD135VAL (16, UAVDark135 val split), NAT2021L (23).
1. GT handling of the toolkit (read from the code, plus a count of the absent-GT frames).
2. Lost frames = frames t >= 1 with a usable GT whose box centre (x + w/2, y + h/2) is more than 20 px from the GT centre. Search window of frame t:
   centred on the box of frame t - 1 (the tracker state), side round(s_x) with s_z = sqrt((w + p) (h + p)), p = CONTEXT_AMOUNT * (w + h),
   s_x = s_z * INSTANCE_SIZE / EXEMPLAR_SIZE (287 / 127), using that box; the tracker's own rule "box area > half the image: s_z = the template s_z of
   frame 0" is applied too. The GT centre is inside if both coordinates are within side / 2 of the window centre.
   Lost stretch = maximal run of consecutive lost frames (a frame without a usable GT or an on-target frame ends it).
   A stretch is "unrecovered" if no later frame of the sequence is on target (usable GT, error <= 20 px).
3. Separation of lost from on-target frames by (a) the peak score p1, (b) the margin (p1 - p2x) / p1, (c) p1 divided by the median of p1 over the
   earlier logged frames; frames with at least 10 earlier logged frames and a usable GT; lower score = more likely lost; ROC AUC, and the share of
   lost frames flagged at 5% and 10% of the on-target frames flagged (cut on 2000 quantile bins of the pooled scores); bootstrap over sequences.
4. Oracles on the success AUC (toolkit definition: mean over IoU thresholds 0, 0.05 .. 1 of the share of ALL frames with IoU > threshold; frames
   without a usable GT count as failures). Rough oracle: in every lost stretch, from the first frame whose search window contains the GT centre
   to the end of the stretch the box is replaced; if no frame of the stretch has the GT centre in its window nothing changes. Full oracle: the
   whole stretch is replaced. Two replacement boxes: "centre" = GT centre with the box size the tracker had on that frame; "gt" = the GT box.
   Static replacement: the windows of later frames are not recomputed.
usage (from tools/): python -I lost_diag.py
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('..'))
from pysot.core.config import cfg
cfg.merge_from_file('../experiments/config.yaml')

OUT = 'illumination'
ROOT = '../test_dataset'
RUN = 'first1.2_m_anchor'
SETS = [('NAT2021', 'first1.2_p15_a0.2_180'), ('NATUD135VAL', 'first1.2_ag'), ('NAT2021L', 'first1.2_p15_a0.2_L')]
PX = 20.0
NB = 2000
MIN_PREFIX = 10
rng = np.random.RandomState(0)
THR = np.arange(0, 1.05, 0.05)
LEN_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 10 ** 9)]


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', path)


def overlap(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return np.clip(inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter), 0, 1)


def auc_of(gt, pred):
    """toolkit success_overlap + mean over thresholds, n_frame = all frames; frames with gt w or h not > 0 (NaN included) never count"""
    iou = np.ones(len(gt)) * -1
    m = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        iou[m] = overlap(gt[m], pred[m])
    return float(np.mean([(iou > t).sum() / float(len(gt)) for t in THR]))


def boot_mean(x):
    x = np.asarray(x, float)
    m = x[rng.randint(0, len(x), size=(NB, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load_set(ds, ref_name):
    meta = json.load(open(os.path.join(ROOT, ds, ds + '.json')))
    seqs = sorted(meta)
    data = {}
    for s in seqs:
        g = np.array(meta[s]['gt_rect'], dtype=float)
        pred = np.loadtxt(os.path.join('results', ds, RUN, s + '.txt'), delimiter=',', ndmin=2)
        n = min(len(g), len(pred))
        assert len(g) == len(pred), (ds, s, len(g), len(pred))
        mrows = list(csv.DictReader(open(os.path.join('margin', ds, RUN, s + '.csv'))))
        mg = {k: np.array([float(r[k]) for r in mrows]) for k in mrows[0]}
        first = meta[s]['img_names'][0]
        first = first if os.path.isabs(first) else os.path.join(ROOT, ds, first)
        import cv2
        img = cv2.imread(first)
        assert img is not None, first
        data[s] = {'gt': g, 'pred': pred, 'margin': mg, 'img_hw': img.shape[:2]}
    return seqs, data


def windows(pred, img_hw):
    """side of the search window of frames 1..N-1 and its centre (from the box of the previous frame)"""
    w, h = pred[:-1, 2], pred[:-1, 3]
    pad = cfg.TRACK.CONTEXT_AMOUNT * (w + h)
    s_z = np.sqrt((w + pad) * (h + pad))
    w0, h0 = pred[0, 2], pred[0, 3]
    pad0 = cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0)
    s_z0 = round(np.sqrt((w0 + pad0) * (h0 + pad0)))
    big = (w * h) > 0.5 * img_hw[0] * img_hw[1]
    s_z = np.where(big, s_z0, s_z)
    side = np.round(s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE))
    return side, pred[:-1, 0] + w / 2., pred[:-1, 1] + h / 2.


def analyse(ds, seqs, data):
    rows_seq, stretch_rows = [], []
    for s in seqs:
        d = data[s]
        g, pred = d['gt'], d['pred']
        n = len(g)
        usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
        gcx, gcy = g[:, 0] + g[:, 2] / 2., g[:, 1] + g[:, 3] / 2.
        pcx, pcy = pred[:, 0] + pred[:, 2] / 2., pred[:, 1] + pred[:, 3] / 2.
        err = np.hypot(pcx - gcx, pcy - gcy)
        side, wcx, wcy = windows(pred, d['img_hw'])
        inside = np.zeros(n, bool)
        inside[1:] = (np.abs(gcx[1:] - wcx) <= side / 2.) & (np.abs(gcy[1:] - wcy) <= side / 2.)
        off = usable & (err > PX)
        off[0] = False
        on = usable & (err <= PX)
        d.update(usable=usable, off=off, on=on, inside=inside, err=err, gcx=gcx, gcy=gcy)
        rows_seq.append([ds, s, n, int((~usable).sum()), int(usable[1:].sum()), int(off.sum()), int((off & inside).sum()), int((off & ~inside).sum()),
                         (off & inside).sum() / off.sum() if off.sum() else ''])
        # stretches
        t = 1
        st = []
        while t < n:
            if off[t]:
                e = t
                while e + 1 < n and off[e + 1]:
                    e += 1
                recovered = bool(on[e + 1:].any())
                ins = np.where(inside[t:e + 1])[0]
                st.append((t, e, recovered, int(ins[0]) if len(ins) else -1))
                stretch_rows.append([ds, s, t, e - t + 1, int(recovered), int(inside[t]), int(ins[0]) if len(ins) else -1])
                t = e + 1
            else:
                t += 1
        d['stretches'] = st
    return rows_seq, stretch_rows


def oracle(ds, seqs, data):
    base, r_c, r_g, f_c, f_g, n_rec, n_st = [], [], [], [], [], 0, 0
    chk = []
    for s in seqs:
        d = data[s]
        g, pred = d['gt'], d['pred']
        base.append(auc_of(g, pred))
        variants = {k: pred.copy() for k in ('rc', 'rg', 'fc', 'fg')}
        for (a, e, rec, first_in) in d['stretches']:
            n_st += 1
            for kind, start in (('f', a), ('r', a + first_in if first_in >= 0 else None)):
                if start is None:
                    continue
                if kind == 'r':
                    n_rec += 1
                sl = slice(start, e + 1)
                cen = pred[sl].copy()
                cen[:, 0] = d['gcx'][sl] - cen[:, 2] / 2.
                cen[:, 1] = d['gcy'][sl] - cen[:, 3] / 2.
                variants[kind + 'c'][sl] = cen
                variants[kind + 'g'][sl] = g[sl]
        r_c.append(auc_of(g, variants['rc']))
        r_g.append(auc_of(g, variants['rg']))
        f_c.append(auc_of(g, variants['fc']))
        f_g.append(auc_of(g, variants['fg']))
    base = np.array(base)
    rows = []
    print('   oracle success AUC (mean over {} sequences), static replacement:'.format(len(seqs)))
    for name, v in (('baseline (recomputed from results/{})'.format(RUN), base), ('rough oracle, GT centre + tracker size', np.array(r_c)),
                    ('rough oracle, GT box', np.array(r_g)), ('full oracle, GT centre + tracker size', np.array(f_c)), ('full oracle, GT box', np.array(f_g))):
        m, lo, hi = boot_mean(v - base) if v is not base else (0, 0, 0)
        print('      {:48s} {:.4f}   gain {:+.4f} [{:+.4f}, {:+.4f}]'.format(name, v.mean(), m, lo, hi))
        rows.append([ds, name, v.mean(), m, lo, hi, n_st, n_rec])
    print('      stretches {}, of which the rough oracle can recover (GT centre inside the window on some frame): {}'.format(n_st, n_rec))
    return rows, base


def stretch_summary(ds, seqs, data, stretch_rows, rows_seq):
    off_n = sum(r[5] for r in rows_seq)
    ins = sum(r[6] for r in rows_seq)
    outs = sum(r[7] for r in rows_seq)
    print('   lost frames {}: GT centre inside the search window {} ({:.1f}%), outside {} ({:.1f}%)'.format(off_n, ins, 100. * ins / max(off_n, 1), outs, 100. * outs / max(off_n, 1)))
    per = np.array([r[8] for r in rows_seq if r[8] != ''], float)
    print('   per sequence (sequences with lost frames: {} of {}): mean share inside {:.3f}, median {:.3f}'.format(len(per), len(seqs), per.mean() if len(per) else float('nan'), np.median(per) if len(per) else float('nan')))
    L = np.array([r[3] for r in stretch_rows])
    rec = np.array([r[4] for r in stretch_rows])
    first_in = np.array([r[5] for r in stretch_rows])
    anyin = np.array([r[6] >= 0 for r in stretch_rows])
    print('   lost stretches {}: length median {:.0f}, mean {:.1f}, 90th percentile {:.0f}, max {}; unrecovered at sequence end {} ({:.1f}% of stretches, {:.1f}% of lost frames)'.format(
        len(L), np.median(L), L.mean(), np.percentile(L, 90), L.max(), int((rec == 0).sum()), 100. * (rec == 0).mean(), 100. * L[rec == 0].sum() / L.sum()))
    print('   stretches whose first frame has the GT centre inside the window: {} ({:.1f}%); with the GT centre inside the window on some frame: {} ({:.1f}%)'.format(
        int(first_in.sum()), 100. * first_in.mean(), int(anyin.sum()), 100. * anyin.mean()))
    out = []
    for lo, hi in LEN_BINS:
        m = (L >= lo) & (L <= hi)
        label = '{}'.format(lo) if lo == hi else ('{}+'.format(lo) if hi > 10 ** 8 else '{}-{}'.format(lo, hi))
        out.append([ds, label, int(m.sum()), int(L[m].sum()), int((rec[m] == 0).sum())])
        print('      length {:8s} stretches {:4d}  lost frames {:6d}  unrecovered {:4d}'.format(label, int(m.sum()), int(L[m].sum()), int((rec[m] == 0).sum())))
    return out, (ins, outs)


# ------------------------------------------------------------------ separation
def score_arrays(d):
    m = d['margin']
    fr = m['frame'].astype(int)
    p1 = m['p1']
    margin = (p1 - m['p2x']) / np.where(p1 > 0, p1, np.nan)
    ratio = np.full(len(p1), np.nan)
    for i in range(MIN_PREFIX, len(p1)):
        med = np.median(p1[:i])
        ratio[i] = p1[i] / med if med > 0 else np.nan
    keep = np.arange(len(p1)) >= MIN_PREFIX
    keep &= d['usable'][fr] & np.isfinite(margin) & np.isfinite(ratio)
    lost = d['off'][fr]
    onf = d['on'][fr]
    sel = keep & (lost | onf)
    return {'peak': p1[sel], 'margin': margin[sel], 'ratio': ratio[sel]}, lost[sel]


def binned(score_by_seq, lost_by_seq, nbins=2000):
    allv = np.concatenate(list(score_by_seq.values()))
    edges = np.unique(np.quantile(allv, np.linspace(0, 1, nbins + 1))[1:-1])
    nb = len(edges) + 1
    seqs = list(score_by_seq)
    pos = np.zeros((len(seqs), nb))
    neg = np.zeros((len(seqs), nb))
    for i, s in enumerate(seqs):
        b = np.searchsorted(edges, score_by_seq[s], side='right')
        np.add.at(pos[i], b, lost_by_seq[s].astype(float))
        np.add.at(neg[i], b, (~lost_by_seq[s]).astype(float))
    return pos, neg


def roc_stats(pos, neg):
    """detector flags a frame as lost if its score <= cut; AUC = P(score_lost < score_on) + 0.5 ties"""
    P, N = pos.sum(axis=0), neg.sum(axis=0)
    if P.sum() == 0 or N.sum() == 0:
        return (float('nan'), float('nan'), float('nan')), None
    cp, cn = np.cumsum(P) / P.sum(), np.cumsum(N) / N.sum()
    tpr = np.concatenate([[0], cp])
    fpr = np.concatenate([[0], cn])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.))
    out = [auc]
    for f in (0.05, 0.10):
        k = np.searchsorted(cn, f, side='right') - 1
        out.append(float(cp[k]) if k >= 0 else 0.0)
    return tuple(out), (fpr, tpr)


def separation(ds, seqs, data):
    sc = {'peak': {}, 'margin': {}, 'ratio': {}}
    lost = {}
    for s in seqs:
        arr, l = score_arrays(data[s])
        if len(l) == 0:
            continue
        for k in sc:
            sc[k][s] = arr[k]
        lost[s] = l
    names = list(lost)
    nl = int(sum(l.sum() for l in lost.values()))
    nn = int(sum((~l).sum() for l in lost.values()))
    print('   separation on frames with >= {} earlier logged frames and a usable GT: lost {} (on target {}); sequences with such frames {}'.format(MIN_PREFIX, nl, nn, len(names)))
    rows, curves = [], {}
    bi = rng.randint(0, len(names), size=(NB, len(names)))
    for k in ('peak', 'margin', 'ratio'):
        pos, neg = binned(sc[k], lost)
        (auc, t5, t10), curve = roc_stats(pos, neg)
        curves[k] = curve
        boots = []
        for idx in bi:
            r, _ = roc_stats(pos[idx], neg[idx])
            if r[0] == r[0]:
                boots.append(r)
        boots = np.array(boots)
        ci = [np.percentile(boots[:, j], [2.5, 97.5]) for j in range(3)]
        print('      {:7s} ROC AUC {:.3f} [{:.3f}, {:.3f}]  TPR at 5% FPR {:.3f} [{:.3f}, {:.3f}]  at 10% FPR {:.3f} [{:.3f}, {:.3f}]'.format(
            k, auc, ci[0][0], ci[0][1], t5, ci[1][0], ci[1][1], t10, ci[2][0], ci[2][1]))
        rows.append([ds, k, nl, nn, auc, ci[0][0], ci[0][1], t5, ci[1][0], ci[1][1], t10, ci[2][0], ci[2][1]])
    return rows, curves


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print('=== item 1: GT handling of the toolkit')
    print('   toolkit/utils/statistics.py success_overlap: mask = (gt w > 0) & (gt h > 0), IoU = -1 where the mask is false (NaN > 0 is false), and the')
    print('   success at every threshold divides by n_frame = ALL frames of the sequence (toolkit/evaluation/ope_benchmark.py; NATVideo has no "absent"')
    print('   attribute, so no frame is dropped). Frames with an absent GT are therefore SCORED, as failures, in every tracker.')
    seq_rows, stretch_rows_all, len_rows, oracle_rows, sep_rows, share_rows = [], [], [], [], [], []
    curves_all = {}
    for ds, ref in SETS:
        seqs, data = load_set(ds, ref)
        ref_auc = []
        for s in seqs:
            r = np.loadtxt(os.path.join('results', ds, ref, s + '.txt'), delimiter=',', ndmin=2)
            ref_auc.append(auc_of(data[s]['gt'], r))
        print('\n##### {} ({} sequences, tracker boxes from results/{}/{})'.format(ds, len(seqs), ds, RUN))
        n_abs = sum(int((~(np.isfinite(data[s]['gt']).all(axis=1) & (data[s]['gt'][:, 2] > 0) & (data[s]['gt'][:, 3] > 0))).sum()) for s in seqs)
        n_all = sum(len(data[s]['gt']) for s in seqs)
        n_abs_seq = sum(1 for s in seqs if (~(np.isfinite(data[s]['gt']).all(axis=1) & (data[s]['gt'][:, 2] > 0) & (data[s]['gt'][:, 3] > 0))).any())
        print('   frames without a usable GT: {} of {} ({:.2f}%), in {} sequences; mean success AUC of results/{}/{} recomputed here {:.4f}'.format(
            n_abs, n_all, 100. * n_abs / n_all, n_abs_seq, ds, ref, np.mean(ref_auc)))
        rows_seq, stretch_rows = analyse(ds, seqs, data)
        seq_rows += rows_seq
        stretch_rows_all += stretch_rows
        lr, (ins, outs) = stretch_summary(ds, seqs, data, stretch_rows, rows_seq)
        len_rows += lr
        share_rows.append([ds, ins, outs])
        orows, base = oracle(ds, seqs, data)
        print('   check: recomputed baseline AUC from results/{}/{} (margin run) {:.4f} against results/{}/{} {:.4f}'.format(ds, RUN, base.mean(), ds, ref, np.mean(ref_auc)))
        oracle_rows += orows
        srows, curves = separation(ds, seqs, data)
        sep_rows += srows
        curves_all[ds] = curves
    write_csv('E15_per_seq_window.csv', ['dataset', 'seq', 'n_frames', 'n_no_gt', 'n_usable_after_0', 'n_lost', 'lost_inside', 'lost_outside', 'share_inside'], seq_rows)
    write_csv('E15_stretches.csv', ['dataset', 'seq', 'start', 'length', 'recovered_later', 'gt_in_window_first_frame', 'first_frame_gt_in_window_offset'], stretch_rows_all)
    write_csv('E15_stretch_length_hist.csv', ['dataset', 'length', 'n_stretches', 'lost_frames', 'unrecovered'], len_rows)
    write_csv('E15_oracle.csv', ['dataset', 'variant', 'mean_auc', 'gain_vs_baseline', 'lo', 'hi', 'n_stretches', 'n_rough_recoverable'], oracle_rows)
    write_csv('E15_separation.csv', ['dataset', 'score', 'n_lost', 'n_on', 'roc_auc', 'lo', 'hi', 'tpr_at_5pct_fpr', 'lo', 'hi', 'tpr_at_10pct_fpr', 'lo', 'hi'], sep_rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    col = {'NAT2021': '#0072B2', 'NATUD135VAL': '#E69F00', 'NAT2021L': '#555555'}
    ax = axes[0, 0]
    for i, (ds, ins, outs) in enumerate(share_rows):
        tot = max(ins + outs, 1)
        ax.bar(i, ins / tot, color=col[ds], label='inside' if i == 0 else None)
        ax.bar(i, outs / tot, bottom=ins / tot, color='#dddddd', edgecolor=col[ds], label='outside' if i == 0 else None)
        ax.text(i, 0.5, '{:.0f}% inside\n({} lost frames)'.format(100. * ins / tot, ins + outs), ha='center', fontsize=9)
    ax.set_xticks(range(len(share_rows)))
    ax.set_xticklabels([r[0] for r in share_rows])
    ax.set_ylabel('share of lost frames')
    ax.set_title('GT centre inside the search window (coloured) or outside (grey)')
    ax = axes[0, 1]
    for ds, _ in SETS:
        rr = [r for r in len_rows if r[0] == ds]
        tot = max(sum(r[3] for r in rr), 1)
        ax.plot(range(len(rr)), [r[3] / tot for r in rr], marker='o', color=col[ds], label=ds)
    ax.set_xticks(range(len(LEN_BINS)))
    ax.set_xticklabels([r[1] for r in len_rows[:len(LEN_BINS)]], rotation=45)
    ax.set_xlabel('length of the lost stretch [frames]')
    ax.set_ylabel('share of the lost frames')
    ax.set_title('where the lost frames sit (by stretch length)')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    style = {'peak': '-', 'margin': '--', 'ratio': ':'}
    for k in ('peak', 'margin', 'ratio'):
        fpr, tpr = curves_all['NAT2021'][k]
        ax.plot(fpr, tpr, style[k], color=col['NAT2021'], label='NAT2021 ' + k)
    ax.plot([0, 1], [0, 1], color='#bbbbbb', lw=1)
    ax.set_xlabel('false positive rate (on-target frames flagged)')
    ax.set_ylabel('true positive rate (lost frames flagged)')
    ax.set_title('ROC of the scores on NAT2021 (lower score = lost)')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    w = 0.25
    for j, (ds, _) in enumerate(SETS):
        for i, k in enumerate(('peak', 'margin', 'ratio')):
            r = [x for x in sep_rows if x[0] == ds and x[1] == k][0]
            ax.bar(i + (j - 1) * w, r[7], w, color=col[ds], label=ds if i == 0 else None, yerr=[[r[7] - r[8]], [r[9] - r[7]]], capsize=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['peak', 'margin', 'ratio to running median'])
    ax.set_ylabel('lost frames flagged at 5% false positives')
    ax.set_title('true positive rate at 5% FPR (95% bootstrap over sequences)')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'E15_lost_diag.png'), dpi=130)
    print('wrote', os.path.join(OUT, 'E15_lost_diag.png'))
    print('\ndone')
