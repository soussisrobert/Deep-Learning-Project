"""E14: per-sequence analysis of B_ag and A_ag against first_ag (first.pth + anchor 0.2 + gamma 0.6). CPU only, no tracking.

Reads existing outputs only:
  illumination/final_{A,B}_<dataset>.csv   per-sequence success AUC (toolkit, absent-target frames count as failures)
  results/NAT2021/<name>/<seq>.txt         tracker boxes (x, y, w, h), row i = frame i
  margin/NAT2021/<name>/<seq>.csv          E13 log (frames 1..N-1): p1, p2x, top-1 centre cx1, cy1 and size w1, h1
  illumination/NAT2021_illumination.csv    per-sequence mean gray, mean gray inside the GT box
  ../test_dataset/<dataset>/<dataset>.json GT boxes, attribute names (ATT01..ATT12 = att00..att11 here), image paths
Frames whose GT is absent (non-finite or w, h <= 0) are left out of items 2 to 4; item 1 uses the toolkit AUC as it is.
Centre error = distance between the box centre (x + w/2, y + h/2) and the GT centre. "On target" = error < 20 px, or error < half
of sqrt(w * h) of the GT box. All intervals are 95% percentile bootstraps that resample whole sequences (2000 draws, seed 0).
usage (from tools/): python -I per_seq_analysis.py
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.stats import rankdata
except Exception:
    def rankdata(a):
        a = np.asarray(a)
        order = a.argsort()
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        return r

sys.path.insert(0, os.getcwd())
OUT = 'illumination'
ROOT = '../test_dataset'
NB = 2000
rng = np.random.RandomState(0)
PX = 20.0
COL = {'first': '#555555', 'A': '#E69F00', 'B': '#0072B2'}
LAB = {'first': 'first.pth anchor+gamma 0.6', 'A': 'A (epoch 8) anchor+gamma 0.6', 'B': 'B (epoch 8) anchor+gamma 0.6'}
RES_N = {'first': 'first1.2_p15_a0.2_180', 'A': 'checkpoint_e81.2_A_ag', 'B': 'checkpoint_e81.2_B_ag'}
MARGIN_RUN = {'first': 'first1.2_m_anchor', 'A': 'checkpoint_e81.2_m_A_ag', 'B': 'checkpoint_e81.2_m_B_ag'}


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', path)


def read_auc(path, col):
    return {r['seq']: float(r[col]) for r in csv.DictReader(open(path))}


def load_json(ds):
    return json.load(open(os.path.join(ROOT, ds, ds + '.json')))


def gt_array(meta):
    g = np.array(meta['gt_rect'], dtype=float)
    ok = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
    return g, ok


def centre(b):
    return b[:, 0] + b[:, 2] / 2., b[:, 1] + b[:, 3] / 2.


def iou(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def boot_idx(n):
    return rng.randint(0, n, size=(NB, n))


def boot_mean(x):
    x = np.asarray(x, dtype=float)
    m = x[boot_idx(len(x))].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def mark(lo, hi):
    return 'excludes 0' if (lo > 0 or hi < 0) else 'includes 0'


# ------------------------------------------------------------------------------------------------ item 1 and 5
def gray_features(ds, meta, step=10):
    """mean gray of every step-th frame, and mean gray inside the GT box on the frames where it exists (cv2 BGR2GRAY)."""
    import cv2
    out = {}
    for seq, m in meta.items():
        g, ok = gt_array(m)
        full, roi = [], []
        for i in range(0, len(m['img_names']), step):
            img = cv2.imread(m['img_names'][i], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            full.append(img.mean())
            if ok[i]:
                x, y, w, h = g[i]
                x0, y0 = max(0, int(x)), max(0, int(y))
                x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
                if x1 > x0 and y1 > y0:
                    roi.append(img[y0:y1, x0:x1].mean())
        out[seq] = (float(np.mean(full)) if full else float('nan'), float(np.mean(roi)) if roi else float('nan'))
    return out


def per_sequence(ds, tag, auc, feats, attrs=None):
    seqs = sorted(auc['first'])
    fa = np.array([auc['first'][s] for s in seqs])
    aa = np.array([auc['A'][s] for s in seqs])
    ba = np.array([auc['B'][s] for s in seqs])
    d = {'A': aa - fa, 'B': ba - fa}
    n = len(seqs)
    print('\n=== item {}: {} ({} sequences), delta = AUC(model_ag) - AUC(first_ag)'.format('1' if ds == 'NAT2021' else '5', ds, n))
    write_csv('E14_per_seq_{}.csv'.format(ds), ['seq', 'first_ag', 'A_ag', 'B_ag', 'dA', 'dB'] + list(feats),
              [[s, fa[i], aa[i], ba[i], d['A'][i], d['B'][i]] + [feats[k][i] for k in feats] for i, s in enumerate(seqs)])
    top_rows = []
    edges = np.arange(-1.0, 1.0001, 0.1)
    for k in ('B', 'A'):
        x = d[k]
        mean, lo, hi = boot_mean(x)
        better, worse, within = int((x > 0.01).sum()), int((x < -0.01).sum()), int((np.abs(x) <= 0.01).sum())
        print('{}: mean {:+.4f} [{:+.4f}, {:+.4f}] ({}), median {:+.4f}, std {:.4f}; better (> +0.01) {}, worse (< -0.01) {}, within +-0.01 {}'.format(
            k, mean, lo, hi, mark(lo, hi), np.median(x), x.std(), better, worse, within))
        h, _ = np.histogram(x, bins=edges)
        print('   histogram of delta, bins of 0.1 from -1.0: ' + ' '.join('{:.1f}:{}'.format(e, c) for e, c in zip(edges[:-1], h) if c))
        order = np.argsort(x)
        win, los = order[::-1][:10], order[:10]
        net = x.sum()
        print('   net delta (sum over sequences) {:+.3f}; top 10 winners sum {:+.3f} ({:.0f}% of the net), top 10 losers sum {:+.3f} ({:.0f}% of the net), the other {} sequences {:+.3f}'.format(
            net, x[win].sum(), 100 * x[win].sum() / net if net else float('nan'), x[los].sum(), 100 * x[los].sum() / net if net else float('nan'),
            n - 20, net - x[win].sum() - x[los].sum()))
        print('   winners: ' + ', '.join('{} {:+.3f} ({:.3f}->{:.3f})'.format(seqs[i], x[i], fa[i], fa[i] + x[i]) for i in win))
        print('   losers:  ' + ', '.join('{} {:+.3f} ({:.3f}->{:.3f})'.format(seqs[i], x[i], fa[i], fa[i] + x[i]) for i in los))
        for kind, idx in (('winner', win), ('loser', los)):
            for r, i in enumerate(idx):
                top_rows.append([k, kind, r + 1, seqs[i], fa[i], fa[i] + x[i], x[i]])
        top_rows.append([k, 'summary', 0, 'net_sum={:.4f} winners_sum={:.4f} losers_sum={:.4f} mean={:.4f}'.format(net, x[win].sum(), x[los].sum(), mean), '', '', ''])
    write_csv('E14_top_{}.csv'.format(ds), ['model', 'kind', 'rank', 'seq', 'first_ag', 'model_ag', 'delta'], top_rows)
    print('corr(dA, dB) pearson {:.3f} (shared structure of the two deltas)'.format(pearson(d['A'], d['B'])))

    fe = dict(feats)
    fe['first_ag_auc'] = fa
    corr_rows, quart_rows = [], []
    print('correlation of delta with sequence properties (pearson / spearman [95% bootstrap interval of spearman]):')
    for fname, v in fe.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        for k in ('B', 'A'):
            x = d[k][ok]
            vv = v[ok]
            sp = spearman(vv, x)
            bs = np.array([spearman(vv[i], x[i]) for i in boot_idx(len(x))[:500]])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            print('   {:13s} d{}: pearson {:+.3f}  spearman {:+.3f} [{:+.3f}, {:+.3f}] (n {})'.format(fname, k, pearson(vv, x), sp, lo, hi, ok.sum()))
            corr_rows.append([fname, k, int(ok.sum()), pearson(vv, x), sp, lo, hi])
        q = np.nanpercentile(v, [25, 50, 75])
        bins = np.searchsorted(q, v, side='right')
        bins[~ok] = -1
        for b in range(4):
            m = bins == b
            if not m.any():
                continue
            row = [fname, 'Q{}'.format(b + 1), int(m.sum()), float(np.nanmin(v[m])), float(np.nanmax(v[m]))]
            for k in ('B', 'A'):
                mean, lo, hi = boot_mean(d[k][m])
                row += [mean, lo, hi]
            quart_rows.append(row)
            print('   {:13s} Q{} [{:.3g} .. {:.3g}] n {:3d}: dB {:+.4f} [{:+.4f}, {:+.4f}]  dA {:+.4f} [{:+.4f}, {:+.4f}]'.format(
                fname, b + 1, row[3], row[4], row[2], row[5], row[6], row[7], row[8], row[9], row[10]))
    write_csv('E14_delta_corr_{}.csv'.format(ds), ['property', 'model', 'n', 'pearson', 'spearman', 'spearman_lo', 'spearman_hi'], corr_rows)
    write_csv('E14_delta_quartiles_{}.csv'.format(ds), ['property', 'quartile', 'n', 'min', 'max', 'dB', 'dB_lo', 'dB_hi', 'dA', 'dA_lo', 'dA_hi'], quart_rows)

    if attrs is not None:
        rows = []
        print('delta by attribute flag (att00..att11 = the 12 flags of anno/att in file order; names unconfirmed):')
        for a in range(12):
            fl = np.array([attrs[s][a] for s in seqs], dtype=bool)
            for state, m in (('flagged', fl), ('unflagged', ~fl)):
                row = ['att{:02d}'.format(a), state, int(m.sum())]
                for k in ('B', 'A'):
                    if m.sum() >= 2:
                        row += list(boot_mean(d[k][m]))
                    else:
                        row += [float('nan')] * 3
                rows.append(row)
            f, u = rows[-2], rows[-1]
            print('   att{:02d} flagged n {:3d} dB {:+.4f} [{:+.4f}, {:+.4f}] dA {:+.4f} [{:+.4f}, {:+.4f}] | unflagged n {:3d} dB {:+.4f} [{:+.4f}, {:+.4f}] dA {:+.4f} [{:+.4f}, {:+.4f}]'.format(
                a, f[2], f[3], f[4], f[5], f[6], f[7], f[8], u[2], u[3], u[4], u[5], u[6], u[7], u[8]))
        write_csv('E14_delta_attr_{}.csv'.format(ds), ['attribute', 'state', 'n', 'dB', 'dB_lo', 'dB_hi', 'dA', 'dA_lo', 'dA_hi'], rows)
    return seqs, fa, d, fe


def figure1(parts):
    fig, axes = plt.subplots(len(parts), 3, figsize=(13, 4 * len(parts)), squeeze=False)
    for r, (ds, seqs, fa, d, fe) in enumerate(parts):
        ax = axes[r, 0]
        edges = np.arange(-0.8, 0.8001, 0.1)
        for k in ('A', 'B'):
            ax.hist(d[k], bins=edges, histtype='step', lw=2, color=COL[k], label='{} (mean {:+.3f})'.format(k, d[k].mean()))
        ax.axvline(0, color='#999999', lw=1)
        ax.set_xlabel('per-sequence delta success AUC vs first_ag')
        ax.set_ylabel('sequences')
        ax.set_title('{}: distribution of the delta (n {})'.format(ds, len(seqs)))
        ax.legend(frameon=False, fontsize=8)
        ax = axes[r, 1]
        for k in ('A', 'B'):
            ax.scatter(fa, d[k], s=14, color=COL[k], alpha=0.7, label=k)
        ax.axhline(0, color='#999999', lw=1)
        ax.set_xlabel('first_ag AUC of the sequence')
        ax.set_ylabel('delta')
        ax.set_title('delta against the baseline AUC')
        ax.legend(frameon=False, fontsize=8)
        ax = axes[r, 2]
        ax.scatter(fe['target_size'], d['B'], s=14, color=COL['B'], alpha=0.7, label='B')
        ax.scatter(fe['target_size'], d['A'], s=14, color=COL['A'], alpha=0.5, label='A')
        ax.axhline(0, color='#999999', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('median target size sqrt(w*h) [px]')
        ax.set_ylabel('delta')
        ax.set_title('delta against target size')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'E14_fig1_per_sequence_delta.png'), dpi=130)
    print('wrote', os.path.join(OUT, 'E14_fig1_per_sequence_delta.png'))


# ------------------------------------------------------------------------------------------------ items 2 to 4
def load_margin(run, seq):
    rows = list(csv.DictReader(open(os.path.join('margin', 'NAT2021', run, seq + '.csv'))))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def frame_data(meta, seqs):
    """per model, per sequence: aligned arrays for the logged frames with a usable GT."""
    data = {k: {} for k in MARGIN_RUN}
    off = []
    for s in seqs:
        g, ok = gt_array(meta[s])
        gcx, gcy = centre(g)
        gsz = np.sqrt(np.maximum(g[:, 2], 0) * np.maximum(g[:, 3], 0))
        for k, run in MARGIN_RUN.items():
            m = load_margin(run, s)
            fr = m['frame'].astype(int)
            pred = np.loadtxt(os.path.join('results', 'NAT2021', run, s + '.txt'), delimiter=',', ndmin=2)
            pcx, pcy = centre(pred)
            e_log = np.hypot(m['cx1'] - gcx[fr], m['cy1'] - gcy[fr])
            off.append(np.hypot(m['cx1'] - pcx[fr], m['cy1'] - pcy[fr]))
            data[k][s] = {'frame': fr, 'p1': m['p1'], 'p2x': m['p2x'], 'cx': m['cx1'], 'cy': m['cy1'], 'w': m['w1'], 'h': m['h1'],
                          'valid': ok[fr] & (m['p1'] > 0), 'err_log': e_log, 'gsz': gsz[fr], 'pred': pred, 'gt': g, 'gok': ok,
                          'err_box': np.hypot(pcx - gcx, pcy - gcy)}
    off = np.concatenate(off)
    print('sanity: distance between the logged top-1 centre and the centre of the box in results/: median {:.2f} px, 90th percentile {:.2f} px, max {:.1f} px'.format(
        np.median(off), np.percentile(off, 90), off.max()))
    return data


def item2(data, seqs):
    print('\n=== item 2: margin (p1 - p2x) / p1 against correctness (logged top-1 centre; frames with a GT)')
    rows = []
    res = {}
    for scheme in ('own', 'common'):
        for k in ('first', 'A', 'B'):
            m_all = np.concatenate([(data[k][s]['p1'] - data[k][s]['p2x'])[data[k][s]['valid']] / data[k][s]['p1'][data[k][s]['valid']] for s in seqs])
            ref = m_all if scheme == 'own' else np.concatenate([(data['first'][s]['p1'] - data['first'][s]['p2x'])[data['first'][s]['valid']] /
                                                              data['first'][s]['p1'][data['first'][s]['valid']] for s in seqs])
            edges = np.percentile(ref, np.arange(0, 101, 10))
            tot = np.zeros((len(seqs), 10))
            h20 = np.zeros((len(seqs), 10))
            hh = np.zeros((len(seqs), 10))
            for i, s in enumerate(seqs):
                dd = data[k][s]
                v = dd['valid']
                mg = (dd['p1'][v] - dd['p2x'][v]) / dd['p1'][v]
                b = np.clip(np.searchsorted(edges[1:-1], mg, side='right'), 0, 9)
                e = dd['err_log'][v]
                hs = e < 0.5 * dd['gsz'][v]
                np.add.at(tot[i], b, 1)
                np.add.at(h20[i], b, e < PX)
                np.add.at(hh[i], b, hs)
            res[(scheme, k)] = (tot, h20, hh, edges)
            bi = boot_idx(len(seqs))
            for b in range(10):
                n = tot[:, b].sum()
                s20 = h20[:, b].sum() / max(n, 1)
                sh = hh[:, b].sum() / max(n, 1)
                bs20 = h20[bi][:, :, b].sum(axis=1) / np.maximum(tot[bi][:, :, b].sum(axis=1), 1)
                bsh = hh[bi][:, :, b].sum(axis=1) / np.maximum(tot[bi][:, :, b].sum(axis=1), 1)
                rows.append([scheme, k, b + 1, edges[b], edges[b + 1], int(n), s20, *np.percentile(bs20, [2.5, 97.5]), sh, *np.percentile(bsh, [2.5, 97.5])])
    write_csv('E14_margin_vs_correct.csv', ['edges', 'model', 'decile', 'margin_lo', 'margin_hi', 'n_frames', 'share_err20', 'lo', 'hi', 'share_half_box', 'lo', 'hi'], rows)
    for scheme in ('own', 'common'):
        print('{} decile edges; share of frames with centre error < 20 px by decile (1 = lowest margin):'.format(scheme))
        for k in ('first', 'A', 'B'):
            print('   {:5s} '.format(k) + ' '.join('{:.3f}'.format(r[6]) for r in rows if r[0] == scheme and r[1] == k))
    print('confidently wrong at high margin, common edges (deciles of first_ag), share of frames with error >= 20 px, bootstrap over sequences:')
    out = []
    bi = boot_idx(len(seqs))
    for bins, label in (([9], 'top decile'), ([8, 9], 'top two deciles')):
        wr = {}
        for k in ('first', 'A', 'B'):
            tot, h20, _, _ = res[('common', k)]
            t = tot[:, bins].sum(axis=1)
            w = t - h20[:, bins].sum(axis=1)
            wr[k] = (t, w)
        for k in ('A', 'B'):
            def rate(idx, kk):
                return wr[kk][1][idx].sum() / np.maximum(wr[kk][0][idx].sum(), 1)
            full = np.arange(len(seqs))
            diff = rate(full, k) - rate(full, 'first')
            bs = np.array([rate(i, k) - rate(i, 'first') for i in bi])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print('   {} {}: first wrong {:.4f}, {} wrong {:.4f}, difference {:+.4f} [{:+.4f}, {:+.4f}] ({}); frames per model {} / {}'.format(
                label, k, rate(full, 'first'), k, rate(full, k), diff, lo, hi, mark(lo, hi), int(wr['first'][0].sum()), int(wr[k][0].sum())))
            out.append([label, k, rate(full, 'first'), rate(full, k), diff, lo, hi, int(wr['first'][0].sum()), int(wr[k][0].sum())])
    write_csv('E14_margin_high_wrong.csv', ['bins', 'model', 'wrong_first', 'wrong_model', 'diff', 'lo', 'hi', 'n_first', 'n_model'], out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, scheme in zip(axes, ('own', 'common')):
        for k in ('first', 'A', 'B'):
            r = [x for x in rows if x[0] == scheme and x[1] == k]
            ax.errorbar(np.arange(1, 11) + {'first': -0.12, 'A': 0, 'B': 0.12}[k], [x[6] for x in r],
                        yerr=[[x[6] - x[7] for x in r], [x[8] - x[6] for x in r]], color=COL[k], marker='o', ms=4, lw=1.5, capsize=2, label=LAB[k])
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('margin decile (1 = lowest), {}'.format('own deciles of each model' if scheme == 'own' else 'deciles of first_ag'))
        ax.set_ylabel('share of frames with centre error < 20 px')
        ax.set_ylim(0, 1)
        ax.set_title('margin vs correctness, NAT2021')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'E14_fig2_margin_vs_correct.png'), dpi=130)
    print('wrote', os.path.join(OUT, 'E14_fig2_margin_vs_correct.png'))


def item3(data, seqs):
    print('\n=== item 3: localisation against box quality (boxes in results/, frames 1.. with a GT, centre error < 20 px = found)')
    stats = {}
    ratios = {}
    for k in ('first', 'A', 'B'):
        arr = np.zeros((len(seqs), 5))   # valid, found, sum iou, sum ratio, found with iou < 0.5
        rr = []
        for i, s in enumerate(seqs):
            dd = data[k][s]
            g, ok, pred = dd['gt'], dd['gok'], dd['pred']
            n = min(len(g), len(pred))
            v = ok[:n].copy()
            v[0] = False
            v &= np.isfinite(pred[:n]).all(axis=1) & (pred[:n, 2] > 0) & (pred[:n, 3] > 0)
            e = dd['err_box'][:n]
            f = v & (e < PX)
            q = iou(pred[:n][f], g[:n][f])
            ra = (pred[:n, 2] * pred[:n, 3])[f] / (g[:n, 2] * g[:n, 3])[f]
            arr[i] = [v.sum(), f.sum(), q.sum(), ra.sum(), (q < 0.5).sum()]
            rr.append(ra)
        stats[k] = arr
        ratios[k] = np.concatenate(rr)
    metrics = [('share_found', lambda a: a[:, 1].sum() / a[:, 0].sum()),
               ('mean_iou_found', lambda a: a[:, 2].sum() / a[:, 1].sum()),
               ('mean_area_ratio_found', lambda a: a[:, 3].sum() / a[:, 1].sum()),
               ('share_iou_lt_0.5_of_found', lambda a: a[:, 4].sum() / a[:, 1].sum()),
               ('share_iou_lt_0.5_of_valid', lambda a: a[:, 4].sum() / a[:, 0].sum())]
    bi = boot_idx(len(seqs))
    rows = []
    for name, fn in metrics:
        base_bs = np.array([fn(stats['first'][i]) for i in bi])
        for k in ('first', 'A', 'B'):
            val = fn(stats[k])
            bs = np.array([fn(stats[k][i]) for i in bi])
            row = [name, k, int(stats[k][:, 0].sum()), val, *np.percentile(bs, [2.5, 97.5])]
            if k != 'first':
                dlt = bs - base_bs
                row += [val - fn(stats['first']), *np.percentile(dlt, [2.5, 97.5])]
            else:
                row += ['', '', '']
            rows.append(row)
            print('   {:27s} {:5s} {:.4f} [{:.4f}, {:.4f}]'.format(name, k, val, row[4], row[5]) +
                  ('   minus first {:+.4f} [{:+.4f}, {:+.4f}] ({})'.format(row[6], row[7], row[8], mark(row[7], row[8])) if k != 'first' else ''))
    for k in ('first', 'A', 'B'):
        print('   median predicted/GT area ratio on found frames, {}: {:.3f} (n {})'.format(k, np.median(ratios[k]), len(ratios[k])))
        rows.append(['median_area_ratio_found', k, len(ratios[k]), float(np.median(ratios[k])), '', '', '', '', ''])
    write_csv('E14_box_quality.csv', ['metric', 'model', 'n_valid_frames', 'value', 'lo', 'hi', 'diff_vs_first', 'diff_lo', 'diff_hi'], rows)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, title) in zip(axes[:3], (('share_found', 'share of frames found (error < 20 px)'),
                                              ('mean_iou_found', 'mean IoU on found frames'),
                                              ('share_iou_lt_0.5_of_found', 'found frames with IoU < 0.5'))):
        for j, k in enumerate(('first', 'A', 'B')):
            r = [x for x in rows if x[0] == name and x[1] == k][0]
            ax.errorbar([j], [r[3]], yerr=[[r[3] - r[4]], [r[5] - r[3]]], color=COL[k], marker='o', ms=7, capsize=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['first', 'A', 'B'])
        ax.set_title(title, fontsize=10)
    for k in ('first', 'A', 'B'):
        axes[3].hist(np.log2(ratios[k]), bins=np.arange(-3, 3.01, 0.1), histtype='step', lw=1.8, color=COL[k], density=True, label=k)
    axes[3].set_xlabel('log2(predicted area / GT area), found frames')
    axes[3].set_ylabel('density')
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'E14_fig3_box_quality.png'), dpi=130)
    print('wrote', os.path.join(OUT, 'E14_fig3_box_quality.png'))


def item4(data, seqs):
    print('\n=== item 4: jump frames (E13 definition: the logged top-1 centre moves more than 0.5 * (w + h) of the previous top-1 box from the previous frame)')
    print('    harmful = centre error before < 20 px and after >= 20 px; recovering = before >= 20 and after < 20; neutral = the rest with a GT on both frames; no_gt = GT absent on either frame')
    cnt = {}
    for k in ('first', 'A', 'B'):
        a = np.zeros((len(seqs), 4))
        for i, s in enumerate(seqs):
            dd = data[k][s]
            g, ok = dd['gt'], dd['gok']
            fr = dd['frame']
            for j in range(1, len(fr)):
                if fr[j] != fr[j - 1] + 1:
                    continue
                dist = np.hypot(dd['cx'][j] - dd['cx'][j - 1], dd['cy'][j] - dd['cy'][j - 1])
                if dist <= 0.5 * (dd['w'][j - 1] + dd['h'][j - 1]):
                    continue
                if not (ok[fr[j]] and ok[fr[j - 1]]):
                    a[i, 3] += 1
                    continue
                b, c = dd['err_log'][j - 1], dd['err_log'][j]
                if b < PX and c >= PX:
                    a[i, 0] += 1
                elif b >= PX and c < PX:
                    a[i, 1] += 1
                else:
                    a[i, 2] += 1
        cnt[k] = a
    rows = []
    for k in ('first', 'A', 'B'):
        t = cnt[k].sum(axis=0)
        print('   {:5s} jump frames {:4d}: harmful {}, recovering {}, neutral {}, no_gt {}; sequences with a harmful jump {}'.format(
            k, int(t.sum()), int(t[0]), int(t[1]), int(t[2]), int(t[3]), int((cnt[k][:, 0] > 0).sum())))
        rows.append([k, int(t.sum()), *[int(x) for x in t], int((cnt[k][:, 0] > 0).sum()), '', '', '', ''])
    bi = boot_idx(len(seqs))
    for k in ('B', 'A'):
        for j, name in ((0, 'harmful'), (1, 'recovering'), (2, 'neutral')):
            dseq = cnt[k][:, j] - cnt['first'][:, j]
            tot = dseq.sum()
            bs = dseq[bi].sum(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print('   {} minus first, {} jumps: total {:+.0f} [{:+.0f}, {:+.0f}] ({})'.format(k, name, tot, lo, hi, mark(lo, hi)))
            rows.append([k + '_minus_first', name, '', '', '', '', '', tot, lo, hi, mark(lo, hi)])
    write_csv('E14_jumps_split.csv', ['model', 'kind_or_total', 'harmful', 'recovering', 'neutral', 'no_gt', 'seqs_with_harmful', 'diff', 'lo', 'hi', 'note'], rows)


# ------------------------------------------------------------------------------------------------ main
def build_features(ds, meta, seqs, ill=None):
    g = {}
    for s in seqs:
        gt, ok = gt_array(meta[s])
        size = np.sqrt(gt[ok, 2] * gt[ok, 3])
        g[s] = (float(np.median(size)) if ok.any() else float('nan'), len(gt))
    if ill is not None:
        gray = {s: (ill[s]['gray_mean'], ill[s]['roi_gray_mean']) for s in seqs}
    else:
        gray = gray_features(ds, {s: meta[s] for s in seqs})
    return {'mean_gray': np.array([gray[s][0] for s in seqs], float), 'roi_gray': np.array([gray[s][1] for s in seqs], float),
            'target_size': np.array([g[s][0] for s in seqs]), 'length': np.array([g[s][1] for s in seqs], float)}


def read_attrs(seqs):
    out = {}
    for s in seqs:
        t = open(os.path.join(ROOT, 'NAT2021', 'anno', 'att', s + '.txt')).read().replace(',', ' ').split()
        out[s] = [int(v) for v in t]
        assert len(out[s]) == 12, s
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    parts = []
    # NAT2021
    auc = {'first': read_auc(os.path.join(OUT, 'final_B_NAT2021.csv'), 'first1.2_p15_a0.2_180'),
           'A': read_auc(os.path.join(OUT, 'final_A_NAT2021.csv'), 'checkpoint_e81.2_A_ag'),
           'B': read_auc(os.path.join(OUT, 'final_B_NAT2021.csv'), 'checkpoint_e81.2_B_ag')}
    seqs = sorted(auc['first'])
    assert len(seqs) == 180 and set(seqs) == set(auc['A']) == set(auc['B'])
    meta = load_json('NAT2021')
    ill = {r['seq']: {k: float(v) for k, v in r.items() if k != 'seq'} for r in csv.DictReader(open(os.path.join(OUT, 'NAT2021_illumination.csv')))}
    feats = build_features('NAT2021', meta, seqs, ill)
    attrs = read_attrs(seqs)
    s1, fa, d, fe = per_sequence('NAT2021', 'NAT2021', auc, feats, attrs)
    parts.append(('NAT2021', s1, fa, d, fe))
    # UAV123 held-out day
    auc = {'first': read_auc(os.path.join(OUT, 'final_B_NATUAV123HELD.csv'), 'first1.2_ag'),
           'A': read_auc(os.path.join(OUT, 'final_A_NATUAV123HELD.csv'), 'checkpoint_e81.2_A_ag'),
           'B': read_auc(os.path.join(OUT, 'final_B_NATUAV123HELD.csv'), 'checkpoint_e81.2_B_ag')}
    seqs_h = sorted(auc['first'])
    assert len(seqs_h) == 30
    meta_h = load_json('NATUAV123HELD')
    feats_h = build_features('NATUAV123HELD', meta_h, seqs_h)
    s2, fa2, d2, fe2 = per_sequence('NATUAV123HELD', 'NATUAV123HELD', auc, feats_h)
    parts.append(('NATUAV123HELD', s2, fa2, d2, fe2))
    figure1(parts)
    # items 2 to 4 on NAT2021
    data = frame_data(meta, seqs)
    item2(data, seqs)
    item3(data, seqs)
    item4(data, seqs)
    print('\ndone')
