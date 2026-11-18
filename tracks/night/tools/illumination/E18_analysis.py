"""E18: oracle template bank, analysis (CPU). usage (from tools/):
   python -I illumination/E18_analysis.py check NAT2021 NAT2021L      R0 against the stored gamma-only result (exit 1 if the mean success AUC differs by more than 0.002)
   python -I illumination/E18_analysis.py report                      R0 / R1 / stored runs, comparisons and diagnostics on NAT2021 and NAT2021L
Success AUC per sequence = toolkit definition (mean over IoU thresholds 0, 0.05 .. 1 of the share of ALL frames with IoU > threshold; frames without a usable GT count as failures).
Lost frame = frame t >= 1 with a usable GT whose box centre (x + w / 2, y + h / 2) is more than 20 px from the GT centre (as E15).
"""
import os
import sys
import csv
import json
import numpy as np

OUT = 'illumination'
ROOT = '../test_dataset'
NB = 2000
PX = 20.0
THR = np.arange(0, 1.05, 0.05)
K = 5
STORED = {'NAT2021': 'first1.2_gamma0.6_180', 'NAT2021L': 'first1.2_gamma0.6_L'}
BEST = 'first1.2_m_anchor'
R0, R1 = 'first1.2_bank_K1', 'first1.2_bank_K5'


def overlap(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return np.clip(inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter), 0, 1)


def iou_of(gt, pred):
    iou = np.ones(len(gt)) * -1
    m = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        iou[m] = overlap(gt[m], pred[m])
    return iou


def auc_from_iou(iou):
    return float(np.mean([(iou > t).sum() / float(len(iou)) for t in THR]))


def load(ds):
    meta = json.load(open(os.path.join(ROOT, ds, ds + '.json')))
    seqs = sorted(meta)
    gt = {s: np.array(meta[s]['gt_rect'], dtype=float) for s in seqs}
    return seqs, gt


def pred_of(ds, run, s):
    return np.loadtxt(os.path.join('results', ds, run, s + '.txt'), delimiter=',', ndmin=2)


def boot(d, rng):
    d = np.asarray(d, float)
    m = d[rng.randint(0, len(d), size=(NB, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def check(dss):
    bad = False
    for ds in dss:
        seqs, gt = load(ds)
        a = np.array([auc_from_iou(iou_of(gt[s], pred_of(ds, STORED[ds], s))) for s in seqs])
        b = np.array([auc_from_iou(iou_of(gt[s], pred_of(ds, R0, s))) for s in seqs])
        d = b - a
        print('R0 check {}: mean success AUC stored {:.4f}  R0 {:.4f}  difference {:+.5f}; sequences differing by > 1e-9: {} of {}; max |diff| {:.5f}'.format(
            ds, a.mean(), b.mean(), d.mean(), int((np.abs(d) > 1e-9).sum()), len(seqs), np.abs(d).max()))
        for i in np.argsort(-np.abs(d))[:int((np.abs(d) > 1e-9).sum())]:
            print('     {:10s} stored {:.4f}  R0 {:.4f}  diff {:+.4f}'.format(seqs[i], a[i], b[i], d[i]))
        if abs(d.mean()) > 0.002:
            print('R0 FAILS the 0.002 criterion on', ds)
            bad = True
    if bad:
        sys.exit(1)
    print('R0 passes on', ' '.join(dss))


def wr(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', p)


def read_bank(ds, run, s):
    rows = list(csv.DictReader(open(os.path.join('bank', ds, run, s + '.csv'))))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def report():
    rng = np.random.RandomState(0)
    summary_rows, cmp_rows, auc_rows, top_rows, add_rows, fill_rows, stretch_rows, rec_rows = [], [], [], [], [], [], [], []
    for ds in ('NAT2021', 'NAT2021L'):
        seqs, gt = load(ds)
        runs = {'stored_gamma': STORED[ds], 'R0': R0, 'R1': R1, 'best_anchor_gamma': BEST}
        iou = {k: {s: iou_of(gt[s], pred_of(ds, v, s)) for s in seqs} for k, v in runs.items()}
        auc = {k: np.array([auc_from_iou(iou[k][s]) for s in seqs]) for k in runs}
        print('\n##### {} ({} sequences)'.format(ds, len(seqs)))
        for k in runs:
            print('   {:18s} mean success AUC {:.4f}'.format(runs[k], auc[k].mean()))
        # fps
        for k in ('R0', 'R1'):
            t = [r for r in csv.DictReader(open(os.path.join('bank', ds, runs[k], 'times.csv')))]
            fr = sum(float(r['frames']) for r in t)
            sec = sum(float(r['seconds']) for r in t)
            summary_rows.append([ds, k, runs[k], auc[k].mean(), fr / sec])
            print('   {} fps (all frames / all seconds, incl. the GT crop and similarity work) {:.1f}'.format(k, fr / sec))
        for i, s in enumerate(seqs):
            auc_rows.append([ds, s] + ['{:.4f}'.format(auc[k][i]) for k in runs])
        # comparisons
        for a, b in (('R1', 'R0'), ('R1', 'best_anchor_gamma'), ('R0', 'stored_gamma'), ('R1', 'stored_gamma')):
            d = auc[a] - auc[b]
            m, lo, hi = boot(d, rng)
            verdict = 'differs' if (lo > 0 or hi < 0) else 'within noise'
            print('   {:3s} - {:18s} {:+.4f} [{:+.4f}, {:+.4f}]  better {} worse {} equal {}  {}'.format(a, b, m, lo, hi, int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), verdict))
            cmp_rows.append([ds, '{} - {}'.format(a, b), len(seqs), m, lo, hi, int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), verdict])
        # top changes
        d = auc['R1'] - auc['R0']
        order = np.argsort(d)
        for tag, idxs in (('gain', order[::-1][:10]), ('loss', order[:10])):
            print('   10 largest {}s R1 - R0:'.format(tag), ', '.join('{} {:+.3f}'.format(seqs[i], d[i]) for i in idxs))
            for rank, i in enumerate(idxs, 1):
                top_rows.append([ds, tag, rank, seqs[i], auc['R0'][i], auc['R1'][i], d[i]])
        # bank
        bk = {s: read_bank(ds, R1, s) for s in seqs}
        adds = {}
        for r in csv.DictReader(open(os.path.join('bank', ds, R1, 'additions.csv'))):
            adds.setdefault(r['seq'], []).append((int(r['frame']), float(r['max_sim'])))
        nadd = np.array([len(adds.get(s, [])) for s in seqs])
        size = 1 + nadd
        print('   bank size per sequence (final): mean {:.2f} median {:.0f}; reaching K={}: {} of {} ({:.1f}%); additions total {} (mean {:.2f} per sequence); sequences with no addition {}'.format(
            size.mean(), np.median(size), K, int((size >= K).sum()), len(seqs), 100. * (size >= K).mean(), int(nadd.sum()), nadd.mean(), int((nadd == 0).sum())))
        for i, s in enumerate(seqs):
            fr = [a[0] for a in adds.get(s, [])]
            fill_rows.append([ds, s, len(bk[s]['frame']) + 1, int(size[i])] + fr + [''] * (K - 1 - len(fr)))
            for j, (f, sim) in enumerate(adds.get(s, []), 1):
                add_rows.append([ds, s, j, f, sim])
        for j in range(1, K):
            fj = np.array([adds[s][j - 1][0] for s in seqs if len(adds.get(s, [])) >= j], float)
            if len(fj):
                print('   addition #{}: {} sequences; frame index median {:.0f}, 25th {:.0f}, 75th {:.0f}, min {:.0f}, max {:.0f}'.format(j, len(fj), np.median(fj), np.percentile(fj, 25), np.percentile(fj, 75), fj.min(), fj.max()))
        # winner
        tot = nw = tot2 = nw2 = 0
        for s in seqs:
            w = bk[s]['winner']
            tot += len(w)
            nw += int((w != 0).sum())
            m = bk[s]['bank_size'] > 1
            tot2 += int(m.sum())
            nw2 += int((w[m] != 0).sum())
        print('   winner is not the frame-1 template: {} of {} tracked frames ({:.1f}%); among frames with bank size > 1: {} of {} ({:.1f}%)'.format(nw, tot, 100. * nw / tot, nw2, tot2, 100. * nw2 / max(tot2, 1)))
        rec_rows.append([ds, 'winner_not_frame1_all_frames', nw, tot, nw / tot])
        rec_rows.append([ds, 'winner_not_frame1_bank_gt1', nw2, tot2, nw2 / max(tot2, 1)])
        # lost-frame diagnostic (user addition 1): R0 lost frames, R1 per-template errors at the same frame index
        for label, restrict in (('all frames', False), ('frames with bank size > 1', True)):
            n = win_ok = any_not_win = none_ok = 0
            for s in seqs:
                g = gt[s]
                usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
                p0 = pred_of(ds, R0, s)
                e0 = np.hypot(p0[:, 0] + p0[:, 2] / 2. - g[:, 0] - g[:, 2] / 2., p0[:, 1] + p0[:, 3] / 2. - g[:, 1] - g[:, 3] / 2.)
                b = bk[s]
                fr = b['frame'].astype(int)
                lost = usable[fr] & (e0[fr] > PX)
                if restrict:
                    lost &= b['bank_size'] > 1
                errs = np.stack([b['err%d' % j] for j in range(K)], axis=1)
                w = b['winner'].astype(int)
                ew = errs[np.arange(len(w)), w]
                with np.errstate(invalid='ignore'):
                    anyok = np.nanmin(np.where(np.isnan(errs), np.inf, errs), axis=1) <= PX
                    winok = ew <= PX
                n += int(lost.sum())
                win_ok += int((lost & winok).sum())
                any_not_win += int((lost & ~winok & anyok).sum())
                none_ok += int((lost & ~anyok).sum())
            print('   R0-lost frames ({}) {}: winner within 20 px {} ({:.1f}%); some bank template within 20 px but the winner not {} ({:.1f}%); no template within 20 px {} ({:.1f}%)'.format(
                label, n, win_ok, 100. * win_ok / max(n, 1), any_not_win, 100. * any_not_win / max(n, 1), none_ok, 100. * none_ok / max(n, 1)))
            rec_rows.append([ds, 'R0_lost[{}]: n / winner<=20 / other_template<=20_winner_not / none<=20'.format(label), n, win_ok, any_not_win, none_ok])
        # IoU change inside / outside the E15 stretches (stretch table from first_ag results of that dataset)
        cats = {'inside, recovered': [], 'inside, unrecovered': [], 'outside all stretches': []}
        rows15 = [r for r in csv.DictReader(open(os.path.join(OUT, 'E15_stretches.csv')))]
        keys = list(rows15[0].keys())
        kds, kseq, kstart, klen, krec = keys[0], keys[1], keys[2], keys[3], keys[4]
        by_seq = {}
        for r in rows15:
            if r[kds] == ds:
                by_seq.setdefault(r[kseq], []).append((int(r[kstart]), int(r[klen]), int(r[krec])))
        n_st = sum(len(v) for v in by_seq.values())
        sums = {c: np.zeros((len(seqs), 2)) for c in cats}
        for i, s in enumerate(seqs):
            n = len(gt[s])
            usable = np.isfinite(gt[s]).all(axis=1) & (gt[s][:, 2] > 0) & (gt[s][:, 3] > 0)
            cat = np.full(n, 2)
            for (a, L, rec) in by_seq.get(s, []):
                cat[a:a + L] = 0 if rec else 1
            cat[0] = -1
            dI = iou['R1'][s] - iou['R0'][s]
            for c, name in enumerate(cats):
                m = usable & (cat == c)
                sums[name][i] = (dI[m].sum(), m.sum())
        print('   IoU change R1 - R0 per frame (frames with a usable GT; stretches = E15 lost stretches of this dataset, {} stretches):'.format(n_st))
        for name in cats:
            S = sums[name]
            tot = S[:, 0].sum() / max(S[:, 1].sum(), 1)
            bs = []
            for _ in range(NB):
                idx = rng.randint(0, len(seqs), len(seqs))
                bs.append(S[idx, 0].sum() / max(S[idx, 1].sum(), 1))
            lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            print('      {:24s} frames {:7d}  mean change {:+.4f} [{:+.4f}, {:+.4f}] (bootstrap over sequences)'.format(name, int(S[:, 1].sum()), tot, lo, hi))
            stretch_rows.append([ds, name, int(S[:, 1].sum()), tot, lo, hi])
    wr('E18_summary.csv', ['dataset', 'run', 'folder', 'success_auc', 'fps'], summary_rows)
    wr('E18_compare.csv', ['dataset', 'comparison', 'n_seq', 'mean_diff', 'lo', 'hi', 'better', 'worse', 'equal', 'verdict'], cmp_rows)
    wr('E18_per_seq_auc.csv', ['dataset', 'seq', 'stored_gamma', 'R0', 'R1', 'best_anchor_gamma'], auc_rows)
    wr('E18_top_changes.csv', ['dataset', 'kind', 'rank', 'seq', 'auc_R0', 'auc_R1', 'diff'], top_rows)
    wr('E18_additions.csv', ['dataset', 'seq', 'k', 'frame', 'max_sim'], add_rows)
    wr('E18_bank_fill.csv', ['dataset', 'seq', 'n_frames', 'final_bank_size', 'add1', 'add2', 'add3', 'add4'], fill_rows)
    wr('E18_iou_change_by_stretch.csv', ['dataset', 'category', 'frames', 'mean_iou_change', 'lo', 'hi'], stretch_rows)
    wr('E18_winner_and_lost.csv', ['dataset', 'item', 'a', 'b', 'c', 'd'], [r + [''] * (6 - len(r)) for r in rec_rows])


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        check(sys.argv[2:])
    else:
        report()
