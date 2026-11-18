"""E17 step 2 (CPU): does the template similarity to frame 1 coincide with lost frames? Reads illumination/E17_template_sim.csv, the boxes of first_ag
(results/NAT2021/first1.2_m_anchor, as E15) and GT. Tracker not touched.
Lost frame: t >= 1, usable GT, centre error > 20 px (as E15); found: error <= 20 px.  Samples: every 10th frame (>= 10) with a usable GT.
a. ROC AUC of the similarity (low = lost) over sampled frames, pooled; bootstrap over sequences (2000, 95%). Extra: same with the similarity minus the sequence median.
b. Lost stretches (E15 definition, maximal runs of lost frames). A stretch start at t counts as 'preceded' if some sampled frame f in [t - 30, t] has sim < the
   median of the sequence's sampled sims (frame 0 excluded from the median). Baseline: for each sequence the same number of start points drawn uniformly
   from frames t >= 1 with a usable GT whose previous frame is not lost; 2000 draws; the share is recomputed with the same rule. Also the crossing version:
   a sampled frame f in [t - 30, t] with sim < median and the previous sampled sim >= median.
c. Share of sampled lost frames whose sim is in the top tercile of all sampled (found + lost) frames (pooled cut), and with the cut taken within the sequence.
d. Spearman (min sim over sampled frames >= 10, sequences with >= 3 samples) vs the sequence success AUC of first_ag; bootstrap over sequences.
usage (from tools/): python -I template_sim_analysis.py
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def rankdata(a):
    """average ranks, 1-based (no scipy in the HiFT env)"""
    a = np.asarray(a, float)
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    return ((ends - cnt + 1 + ends) / 2.0)[inv]


def spearmanr(x, y):
    return (np.corrcoef(rankdata(x), rankdata(y))[0, 1],)

OUT = 'illumination'
ROOT = '../test_dataset'
DS = 'NAT2021'
RUN = 'first1.2_m_anchor'
PX = 20.0
NB = 2000
NPERM = 2000
WIN = 30
THR = np.arange(0, 1.05, 0.05)
rng = np.random.RandomState(0)


def overlap(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return np.clip(inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter), 0, 1)


def auc_of(gt, pred):
    iou = np.ones(len(gt)) * -1
    m = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        iou[m] = overlap(gt[m], pred[m])
    return float(np.mean([(iou > t).sum() / float(len(gt)) for t in THR]))


def roc_auc(x, lost):
    """P(x_lost < x_found) + 0.5 ties"""
    n1, n0 = int(lost.sum()), int((~lost).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(x)
    # low x = lost: AUC = P(x_found > x_lost)
    return float((r[~lost].sum() - n0 * (n0 + 1) / 2.) / (n0 * n1))


def pct(x, lo=2.5, hi=97.5):
    return np.percentile(x, lo), np.percentile(x, hi)


def main():
    meta = json.load(open(os.path.join(ROOT, DS, DS + '.json')))
    seqs = sorted(meta)
    sims = {}
    for r in csv.DictReader(open(os.path.join(OUT, 'E17_template_sim.csv'))):
        sims.setdefault(r['seq'], {})[int(r['frame'])] = float(r['sim'])
    D = {}
    for s in seqs:
        g = np.array(meta[s]['gt_rect'], dtype=float)
        pred = np.loadtxt(os.path.join('results', DS, RUN, s + '.txt'), delimiter=',', ndmin=2)
        assert len(g) == len(pred), s
        n = len(g)
        usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
        err = np.hypot(pred[:, 0] + pred[:, 2] / 2. - g[:, 0] - g[:, 2] / 2., pred[:, 1] + pred[:, 3] / 2. - g[:, 1] - g[:, 3] / 2.)
        off = usable & (err > PX)
        off[0] = False
        on = usable & (err <= PX)
        sf = sorted(f for f in sims.get(s, {}) if f >= 10 and f < n and usable[f])
        sv = np.array([sims[s][f] for f in sf])
        allf = sorted(sims.get(s, {}))
        med = float(np.median(sv)) if len(sv) else float('nan')
        # stretches
        st, t = [], 1
        while t < n:
            if off[t]:
                e = t
                while e + 1 < n and off[e + 1]:
                    e += 1
                st.append((t, e))
                t = e + 1
            else:
                t += 1
        D[s] = dict(n=n, off=off, on=on, usable=usable, sf=np.array(sf, int), sv=sv, med=med, st=st, auc=auc_of(g, pred),
                    sim0=sims.get(s, {}))
    # ---- a
    S = [s for s in seqs if len(D[s]['sf'])]
    lost_by, sim_by, simc_by = {}, {}, {}
    for s in S:
        d = D[s]
        lost = d['off'][d['sf']]
        found = d['on'][d['sf']]
        keep = lost | found
        lost_by[s] = lost[keep]
        sim_by[s] = d['sv'][keep]
        simc_by[s] = d['sv'][keep] - d['med']
    def pooled(sd, ld, idx):
        return np.concatenate([sd[S[i]] for i in idx]), np.concatenate([ld[S[i]] for i in idx])
    res_a = []
    for name, sd in (('similarity to frame 1', sim_by), ('similarity minus sequence median', simc_by)):
        x, l = pooled(sd, lost_by, range(len(S)))
        a0 = roc_auc(x, l)
        bs = []
        for _ in range(NB):
            idx = rng.randint(0, len(S), len(S))
            xb, lb = pooled(sd, lost_by, idx)
            bs.append(roc_auc(xb, lb))
        lo, hi = pct(np.array(bs))
        print('a. {:36s} ROC AUC {:.3f} [{:.3f}, {:.3f}]   sampled lost {} found {} in {} sequences'.format(name, a0, lo, hi, int(l.sum()), int((~l).sum()), len(S)))
        res_a.append([name, a0, lo, hi, int(l.sum()), int((~l).sum()), len(S)])
    x, l = pooled(sim_by, lost_by, range(len(S)))
    print('   mean sim lost {:.3f} found {:.3f}; median lost {:.3f} found {:.3f}'.format(x[l].mean(), x[~l].mean(), np.median(x[l]), np.median(x[~l])))
    # ---- b
    def preceded(d, starts, crossing):
        c = 0
        for t in starts:
            ok = False
            for k, f in enumerate(d['sf']):
                if t - WIN <= f <= t and d['sv'][k] < d['med']:
                    if not crossing:
                        ok = True
                    else:
                        prev = d['sv'][k - 1] if k > 0 else d['sim0'].get(0, 1.0)
                        ok = ok or prev >= d['med']
                if ok:
                    break
            c += int(ok)
        return c
    n_st = sum(len(D[s]['st']) for s in S)
    res_b = []
    for crossing in (False, True):
        hit = sum(preceded(D[s], [a for a, e in D[s]['st']], crossing) for s in S)
        perm = []
        for _ in range(NPERM):
            h = 0
            for s in S:
                d = D[s]
                k = len(d['st'])
                if k == 0:
                    continue
                prev_off = np.concatenate([[False], d['off'][:-1]])
                cand = np.where(d['usable'] & ~prev_off & (np.arange(d['n']) >= 1))[0]
                starts = rng.choice(cand, size=k, replace=len(cand) < k)
                h += preceded(d, starts, crossing)
            perm.append(h / float(n_st))
        perm = np.array(perm)
        p = (np.sum(perm >= hit / float(n_st)) + 1.) / (NPERM + 1.)
        lo, hi = pct(perm)
        name = 'crossing below median' if crossing else 'some sampled frame below median'
        print('b. {:34s} stretch starts {}: preceded {} ({:.3f}); random starts mean {:.3f} [{:.3f}, {:.3f}] ; P(random >= observed) {:.3f}'.format(
            name, n_st, hit, hit / float(n_st), perm.mean(), lo, hi, p))
        res_b.append([name, n_st, hit, hit / float(n_st), perm.mean(), lo, hi, p])
    # ---- c
    xa = np.concatenate([sim_by[s] for s in S])
    cut = np.quantile(xa, 2. / 3.)
    xl = np.concatenate([sim_by[s][lost_by[s]] for s in S])
    share_pool = float((xl >= cut).mean())
    hi_in, tot = 0, 0
    for s in S:
        if len(sim_by[s]) < 3:
            continue
        c = np.quantile(sim_by[s], 2. / 3.)
        v = sim_by[s][lost_by[s]]
        hi_in += int((v >= c).sum())
        tot += len(v)
    print('c. sampled lost frames {}; sim in the top tercile (pooled cut {:.3f}): {} ({:.3f}); within-sequence top tercile (sequences with >= 3 samples): {} of {} ({:.3f}); expected 0.333 if no relation'.format(
        len(xl), cut, int((xl >= cut).sum()), share_pool, hi_in, tot, hi_in / max(tot, 1)))
    res_c = [len(xl), cut, int((xl >= cut).sum()), share_pool, hi_in, tot, hi_in / max(tot, 1)]
    # ---- d
    Sd = [s for s in S if len(D[s]['sv']) >= 3]
    mn = np.array([D[s]['sv'].min() for s in Sd])
    au = np.array([D[s]['auc'] for s in Sd])
    rho = spearmanr(mn, au)[0]
    bs = []
    for _ in range(NB):
        idx = rng.randint(0, len(Sd), len(Sd))
        bs.append(spearmanr(mn[idx], au[idx])[0])
    lo, hi = pct(np.array(bs))
    print('d. Spearman(min sim, sequence AUC) {:.3f} [{:.3f}, {:.3f}] over {} sequences (>= 3 samples)'.format(rho, lo, hi, len(Sd)))
    res_d = [rho, lo, hi, len(Sd)]
    # ---- outputs
    def wr(name, header, rows):
        with open(os.path.join(OUT, name), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print('wrote', os.path.join(OUT, name))
    wr('E17_roc_auc.csv', ['measure', 'auc', 'lo', 'hi', 'n_lost', 'n_found', 'n_seq'], res_a)
    wr('E17_stretch_precedence.csv', ['rule', 'n_stretches', 'preceded', 'share', 'random_mean', 'random_lo', 'random_hi', 'p_random_ge'], res_b)
    wr('E17_high_sim_lost.csv', ['n_lost_sampled', 'pooled_top_tercile_cut', 'n_high_pooled', 'share_pooled', 'n_high_within', 'n_within_total', 'share_within'], [res_c])
    wr('E17_per_seq.csv', ['seq', 'n_samples', 'min_sim', 'median_sim', 'auc_first_ag', 'n_stretches'],
       [[s, len(D[s]['sv']), D[s]['sv'].min() if len(D[s]['sv']) else '', D[s]['med'], D[s]['auc'], len(D[s]['st'])] for s in seqs])
    wr('E17_spearman.csv', ['rho', 'lo', 'hi', 'n_seq'], [res_d])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist([x[~l], x[l]], bins=30, density=True, label=['found', 'lost'], color=['#2a7', '#c42'])
    ax[0].set_xlabel('cosine similarity to frame-1 template'); ax[0].legend(); ax[0].set_title('a. sampled frames')
    ax[1].bar(['observed', 'random starts'], [res_b[0][3], res_b[0][4]], yerr=[[0, res_b[0][4] - res_b[0][5]], [0, res_b[0][6] - res_b[0][4]]], color=['#c42', '#999'])
    ax[1].set_title('b. stretch starts with sim < median in prior 30 frames'); ax[1].set_ylim(0, 1)
    ax[2].scatter(mn, au, s=10)
    ax[2].set_xlabel('minimum sampled similarity'); ax[2].set_ylabel('sequence AUC (first_ag)'); ax[2].set_title('d. Spearman {:.2f}'.format(rho))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'E17_template_sim.png'), dpi=130)
    print('wrote E17_template_sim.png')


if __name__ == '__main__':
    main()
