"""E22 A-C: loss taxonomy on NAT2021 (when and how the tracker loses the target). CPU only; saved boxes, E13 logs, GT, frames, attributes, E15 stretches.
Run from tools/:  python -I illumination/E22_loss_taxonomy.py
Run: first.pth + anchor 0.2 + gamma 0.6 = results/NAT2021/first1.2_m_anchor, E13 logs margin/NAT2021/first1.2_m_anchor/<seq>.csv.
Lost frame = usable GT and centre error (x + w / 2, y + h / 2) > 20 px; stretch = E15 (illumination/E15_stretches.csv, dataset NAT2021); start = first lost frame.
Controls: per stretch start one tracked frame (usable GT, centre error <= 20 px, previous frame also tracked, frame >= 10, a log row) of the SAME sequence, drawn uniformly
  from that sequence's control pool; the pool is the eligible frames capped at 150 per sequence (random subset, fixed seed) so that the image features stay affordable.
  2000 draws, no resampling, give the point estimate; the 95% intervals come from 2000 bootstrap replicates that resample sequences (with starts and a pool) and draw controls anew.
Logged peaks: top-1 (cx1, cy1), 2d (cx2d, cy2d) and 2x (cx2x, cy2x) are box centres in image coordinates; the error is the distance to the GT centre.
  A NaN box (2x: no qualifying cell, n_distinct = 0) counts as NOT within 20 px.
"""
import os
import sys
import csv
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath('..'))
from pysot.core.config import cfg
cfg.merge_from_file('../experiments/config.yaml')
cfg.CUDA = False
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.enhance import make_enhancer
from pysot.utils.bbox import get_axis_aligned_bbox

DS = 'NAT2021'
ROOT = '../test_dataset/' + DS
RUN = 'first1.2_m_anchor'
OUT = 'illumination'
PX = 20.0
POOL_CAP = 150
NB = 2000
ND = 2000
rng_pool = np.random.RandomState(0)
rng_draw = np.random.RandomState(1)
rng_boot = np.random.RandomState(2)


class _M(object):
    def eval(self):
        return self


tracker = HiFTTracker(_M())
gamma_enh = make_enhancer('gamma', 0.6, 3.0, 'none')


def crop(img, pos, model_sz, orig_sz, avg, enh):
    t = tracker.get_subwindow(img, pos, model_sz, orig_sz, avg, enh)
    return np.ascontiguousarray(t.numpy()[0].transpose(1, 2, 0)).astype(np.uint8)


def rankdata(a):
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    return ((ends - cnt + 1 + ends) / 2.0)[inv]


def auc_start_gt_control(x, y):
    """P(start > control) + 0.5 ties; NaNs dropped"""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    r = rankdata(np.concatenate([x, y]))
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2.) / (len(x) * len(y)))


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', os.path.join(OUT, name))


def pct(x):
    return np.nanpercentile(x, 2.5, axis=0), np.nanpercentile(x, 97.5, axis=0)


# ----------------------------------------------------------------------------- load
meta = json.load(open(os.path.join(ROOT, DS + '.json')))
seqs = sorted(meta)
S = {}
for s in seqs:
    g = np.array(meta[s]['gt_rect'], dtype=float)
    p = np.loadtxt(os.path.join('results', DS, RUN, s + '.txt'), delimiter=',', ndmin=2)
    assert len(g) == len(p)
    n = len(g)
    usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
    gcx, gcy = g[:, 0] + g[:, 2] / 2., g[:, 1] + g[:, 3] / 2.
    err = np.hypot(p[:, 0] + p[:, 2] / 2. - gcx, p[:, 1] + p[:, 3] / 2. - gcy)
    err[~usable] = np.nan
    rows = list(csv.DictReader(open(os.path.join('margin', DS, RUN, s + '.csv'))))
    fr = np.array([int(r['frame']) for r in rows])
    assert (fr == np.arange(1, n)).all(), s
    pk = np.full((n, 4), np.nan)           # error of top-1, 2d, 2x peak; n_distinct
    for j, (cx, cy) in enumerate((('cx1', 'cy1'), ('cx2d', 'cy2d'), ('cx2x', 'cy2x'))):
        x = np.array([float(r[cx]) for r in rows]); y = np.array([float(r[cy]) for r in rows])
        pk[1:, j] = np.hypot(x - gcx[1:], y - gcy[1:])
    pk[1:, 3] = [float(r['n_distinct']) for r in rows]
    pk[~usable, :3] = np.nan
    lost = usable & (err > PX); lost[0] = False
    S[s] = dict(g=g, p=p, n=n, usable=usable, gcx=gcx, gcy=gcy, err=err, pk=pk, lost=lost, att=[int(v) for v in open(os.path.join(ROOT, 'anno/att', s + '.txt')).read().strip().split(',')])
    S[s]['img0'] = meta[s]['img_names'][0]
    S[s]['names'] = meta[s]['img_names']

stretches = {}
for r in csv.DictReader(open(os.path.join(OUT, 'E15_stretches.csv'))):
    if r['dataset'] == DS:
        stretches.setdefault(r['seq'], []).append((int(r['start']), int(r['length']), int(r['recovered_later'])))
n_st = sum(len(v) for v in stretches.values())
print('NAT2021: {} sequences, {} stretches in {} sequences, {} lost frames; lost frames recomputed here {}'.format(
    len(seqs), n_st, len(stretches), sum(L for v in stretches.values() for _, L, _ in v), sum(int(S[s]['lost'].sum()) for s in seqs)))
for s in stretches:  # consistency of the E15 table with the lost flags
    for a, L, _ in stretches[s]:
        assert S[s]['lost'][a:a + L].all() and (a == 1 or not S[s]['lost'][a - 1]), (s, a, L)
top1_bad = sum(int((S[s]['pk'][a, 0] <= PX)) for s in stretches for a, _, _ in stretches[s])
print('starts whose logged top-1 peak is within 20 px of the GT (clip / smoothing effects): {} of {}'.format(top1_bad, n_st))

# controls
for s in seqs:
    d = S[s]
    n = d['n']
    ok = d['usable'] & (d['err'] <= PX)
    elig = [t for t in range(10, n) if ok[t] and ok[t - 1]]
    d['n_elig'] = len(elig)
    if len(elig) > POOL_CAP:
        elig = sorted(rng_pool.choice(elig, POOL_CAP, replace=False))
    d['pool'] = np.array(elig, int)
    d['starts'] = np.array([a for a, _, _ in stretches.get(s, [])], int)
paired = [s for s in seqs if len(S[s]['starts']) and len(S[s]['pool'])]
nostart_ctrl = [s for s in seqs if len(S[s]['starts']) and not len(S[s]['pool'])]
print('sequences with a start and a control pool: {} ({} starts); sequences with starts but no eligible control frame: {} ({} starts)'.format(
    len(paired), sum(len(S[s]['starts']) for s in paired), len(nostart_ctrl), sum(len(S[s]['starts']) for s in nostart_ctrl)))

# ----------------------------------------------------------------------------- features
FEATS = ['speed5', 'dlogw10', 'dlogh10', 'abs_dlogarea10', 'size', 'border_dist', 'window_edge_dist']
for who in ('tgt', 'srch'):
    for g_ in ('raw', 'gamma'):
        FEATS += ['{}_{}_{}'.format(who, k, g_) for k in ('lum', 'contrast', 'lapstd')]


def window_of(d, f):
    """search window of frame f: from the tracker box of frame f - 1 (E15 rule)"""
    p = d['p']
    w, h = p[f - 1, 2], p[f - 1, 3]
    pad = cfg.TRACK.CONTEXT_AMOUNT * (w + h)
    sz = np.sqrt((w + pad) * (h + pad))
    w0, h0 = p[0, 2], p[0, 3]
    sz0 = round(np.sqrt((w0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0)) * (h0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0))))
    return sz, sz0


def stats_of(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return [gray.mean(), gray.std(), cv2.Laplacian(gray, cv2.CV_64F).std()]


def gt_features(d, f, H, W):
    out = [np.nan] * 7
    if not d['usable'][f]:
        return out
    size = np.sqrt(d['g'][f, 2] * d['g'][f, 3])
    disp = []
    for k in range(f - 4, f + 1):
        if k >= 1 and d['usable'][k] and d['usable'][k - 1]:
            disp.append(np.hypot(d['gcx'][k] - d['gcx'][k - 1], d['gcy'][k] - d['gcy'][k - 1]) / size)
    out[0] = float(np.mean(disp)) if len(disp) >= 3 else np.nan
    if f >= 10 and d['usable'][f - 10]:
        out[1] = float(np.log(d['g'][f, 2] / d['g'][f - 10, 2]))
        out[2] = float(np.log(d['g'][f, 3] / d['g'][f - 10, 3]))
        out[3] = abs(out[1] + out[2])
    out[4] = float(size)
    out[5] = float(min(d['gcx'][f], d['gcy'][f], W - d['gcx'][f], H - d['gcy'][f]))
    sz, sz0 = window_of(d, f)
    if d['p'][f - 1, 2] * d['p'][f - 1, 3] > 0.5 * H * W:
        sz = sz0
    side = np.round(sz * cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    wcx, wcy = d['p'][f - 1, 0] + d['p'][f - 1, 2] / 2., d['p'][f - 1, 1] + d['p'][f - 1, 3] / 2.
    out[6] = float(side / 2. - max(abs(d['gcx'][f] - wcx), abs(d['gcy'][f] - wcy)))
    return out


def seq_features(s, frames):
    """feature matrix (len(frames), len(FEATS)); images read once per frame"""
    d = S[s]
    img0 = cv2.imread(os.path.join(ROOT, d['img0']))
    H, W = img0.shape[:2]
    avg = np.mean(img0, axis=(0, 1))
    b0 = d['g'][0]
    cx0, cy0, w0, h0 = get_axis_aligned_bbox(b0)
    wz0 = w0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0); hz0 = h0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0)
    scaleaa = round(np.sqrt(wz0 * hz0))
    X = np.full((len(frames), len(FEATS)), np.nan)
    for i, f in enumerate(frames):
        X[i, :7] = gt_features(d, f, H, W)
        if not d['usable'][f]:
            continue
        img = cv2.imread(os.path.join(ROOT, d['names'][f]))
        cx, cy, w, h = get_axis_aligned_bbox(d['g'][f])
        wz = w + cfg.TRACK.CONTEXT_AMOUNT * (w + h); hz = h + cfg.TRACK.CONTEXT_AMOUNT * (w + h)
        sz = round(np.sqrt(wz * hz))
        if w * h > 0.5 * H * W:
            sz = scaleaa
        pos = np.array([cx, cy])
        j = 7
        for enh in (None, gamma_enh):  # order in FEATS: tgt raw, tgt gamma, srch raw, srch gamma
            X[i, j:j + 3] = stats_of(crop(img, pos, cfg.TRACK.EXEMPLAR_SIZE, sz, avg, enh))
            j += 3
        # search crop from the tracker state at f - 1
        pw, ph = d['p'][f - 1, 2], d['p'][f - 1, 3]
        spos = np.array([d['p'][f - 1, 0] + pw / 2., d['p'][f - 1, 1] + ph / 2.])
        wz = pw + cfg.TRACK.CONTEXT_AMOUNT * (pw + ph); hz = ph + cfg.TRACK.CONTEXT_AMOUNT * (pw + ph)
        sz_ = np.sqrt(wz * hz)
        if pw * ph > 0.5 * H * W:
            sz_ = scaleaa
        sx = sz_ * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        for enh in (None, gamma_enh):
            X[i, j:j + 3] = stats_of(crop(img, spos, cfg.TRACK.INSTANCE_SIZE, round(sx), avg, enh))
            j += 3
    # the FEATS order is tgt raw, tgt gamma, srch raw, srch gamma (FEATS built with who outer, g_ inner)
    return X


print('computing features ({} starts + control pools) ...'.format(n_st), flush=True)
for k, s in enumerate(paired + nostart_ctrl):
    d = S[s]
    d['Fs'] = seq_features(s, d['starts'])
    d['Fc'] = seq_features(s, d['pool']) if len(d['pool']) else np.zeros((0, len(FEATS)))
    if k % 20 == 0:
        print('   {} / {}'.format(k, len(paired) + len(nostart_ctrl)), flush=True)

# ----------------------------------------------------------------------------- paired statistics
NF = len(FEATS)


def draw(seq_list, rng):
    Xs, Xc, As, Ac = [], [], [], []
    for s in seq_list:
        d = S[s]
        idx = rng.randint(0, len(d['pool']), len(d['starts']))
        Xs.append(d['Fs']); Xc.append(d['Fc'][idx])
        As.append(d['pk'][d['starts']]); Ac.append(d['pk'][d['pool'][idx]])
    return np.concatenate(Xs), np.concatenate(Xc), np.concatenate(As), np.concatenate(Ac)


def metrics(Xs, Xc, As, Ac):
    md = np.nanmean(Xs, axis=0) - np.nanmean(Xc, axis=0)
    au = np.array([auc_start_gt_control(Xs[:, j], Xc[:, j]) for j in range(NF)])
    alt_s = np.nanmean(((As[:, 1] <= PX) | (As[:, 2] <= PX)).astype(float))
    alt_c = np.nanmean(((Ac[:, 1] <= PX) | (Ac[:, 2] <= PX)).astype(float))
    return np.concatenate([md, au, [alt_s, alt_c, np.nanmean(As[:, 3]), np.nanmean(Ac[:, 3])]])


pt = np.mean([metrics(*draw(paired, rng_draw)) for _ in range(ND)], axis=0)
bt = np.array([metrics(*draw([paired[i] for i in rng_boot.randint(0, len(paired), len(paired))], rng_draw)) for _ in range(NB)])
lo, hi = pct(bt)

# ----------------------------------------------------------------------------- A
print('\n===== A. Was the target a logged peak? =====')
starts_all = [(s, a) for s in stretches for a, _, _ in stretches[s]]
PKS = np.array([S[s]['pk'][a] for s, a in starts_all])
run_2d = PKS[:, 1] <= PX
run_2x = PKS[:, 2] <= PX
runner = run_2d | run_2x
print('stretch starts: {}'.format(len(PKS)))


def share_boot(flag_by_seq):
    """flag_by_seq: dict seq -> (hits, n); bootstrap over sequences"""
    ks = list(flag_by_seq)
    h = np.array([flag_by_seq[k][0] for k in ks], float); n = np.array([flag_by_seq[k][1] for k in ks], float)
    pt_ = h.sum() / n.sum()
    bs = []
    for _ in range(NB):
        i = rng_boot.randint(0, len(ks), len(ks))
        bs.append(h[i].sum() / max(n[i].sum(), 1))
    return pt_, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(h.sum()), int(n.sum())


def by_seq(mask_fn):
    out = {}
    for s in stretches:
        a = np.array([x for x, _, _ in stretches[s]])
        m = mask_fn(S[s]['pk'][a])
        out[s] = (int(m.sum()), len(a))
    return out


rowsA = []
for name, fn in (('runner-up: top-1 wrong, 2d or 2x within 20 px', lambda P: (P[:, 1] <= PX) | (P[:, 2] <= PX)),
                 ('  of which 2d within 20 px', lambda P: P[:, 1] <= PX),
                 ('  of which 2x within 20 px', lambda P: P[:, 2] <= PX),
                 ('not a logged peak: none of the three within 20 px', lambda P: ~((P[:, 0] <= PX) | (P[:, 1] <= PX) | (P[:, 2] <= PX)))):
    p_, l_, h_, hh, nn = share_boot(by_seq(fn))
    print('   starts, {:55s} {:.3f} [{:.3f}, {:.3f}]  ({} of {})'.format(name, p_, l_, h_, hh, nn))
    rowsA.append(['starts', name.strip(), p_, l_, h_, hh, nn])
j0 = NF * 2
names4 = ['share with 2d or 2x within 20 px: starts', 'same: controls', 'mean n_distinct: starts', 'mean n_distinct: controls']
for i, nm in enumerate(names4):
    print('   {:48s} {:.3f} [{:.3f}, {:.3f}]'.format(nm, pt[j0 + i], lo[j0 + i], hi[j0 + i]))
    rowsA.append(['starts vs controls', nm, pt[j0 + i], lo[j0 + i], hi[j0 + i], '', ''])
print('   (controls are tracked frames, so their top-1 peak is within 20 px by construction; "2d or 2x within 20 px" on a control means a second logged peak is also on the target.)')
print('   n_distinct at starts minus controls: {:+.2f}'.format(pt[j0 + 2] - pt[j0 + 3]))
print('   sequence-level control draws: {} sequences, {} starts paired with controls'.format(len(paired), sum(len(S[s]['starts']) for s in paired)))
write_csv('E22_A_start_categories.csv', ['group', 'category', 'share_or_mean', 'lo', 'hi', 'hits', 'n'], rowsA)

print('\n   share of lost frames where any logged peak (top-1, 2d or 2x) is within 20 px of the GT (the top-1 is wrong on a lost frame by definition):')
rowsL = []
for label, sel in (('1st lost frame of each stretch', lambda a, L: [a] if L >= 1 else []),
                   ('5th lost frame (stretches >= 5 frames)', lambda a, L: [a + 4] if L >= 5 else []),
                   ('20th lost frame (stretches >= 20 frames)', lambda a, L: [a + 19] if L >= 20 else []),
                   ('all lost frames of stretches >= 101 frames', lambda a, L: list(range(a, a + L)) if L >= 101 else [])):
    for sub, fn in (('any peak', lambda P: ((P[:, 0] <= PX) | (P[:, 1] <= PX) | (P[:, 2] <= PX))),
                    ('2d', lambda P: P[:, 1] <= PX), ('2x', lambda P: P[:, 2] <= PX)):
        out = {}
        for s in stretches:
            fl = [f for a, L, _ in stretches[s] for f in sel(a, L)]
            if fl:
                out[s] = (int(fn(S[s]['pk'][np.array(fl)]).sum()), len(fl))
        if not out:
            continue
        p_, l_, h_, hh, nn = share_boot(out)
        print('      {:44s} {:9s} {:.3f} [{:.3f}, {:.3f}]  ({} of {} frames in {} sequences)'.format(label, sub, p_, l_, h_, hh, nn, len(out)))
        rowsL.append([label, sub, p_, l_, h_, hh, nn, len(out)])
write_csv('E22_A_lost_frame_peaks.csv', ['frames', 'peak', 'share', 'lo', 'hi', 'hits', 'n_frames', 'n_sequences'], rowsL)

# ----------------------------------------------------------------------------- B
print('\n===== B. Features at the start of a loss versus controls (paired set: {} sequences) ====='.format(len(paired)))
print('   {:24s} {:>10s} {:>10s} {:>22s}   {:>7s} {:>16s}'.format('feature', 'mean start', 'mean ctrl', 'difference [95%]', 'AUC', '[95%]'))
Xs0, Xc0, _, _ = draw(paired, np.random.RandomState(3))
rowsB = []
for j, f in enumerate(FEATS):
    ms, mc = np.nanmean(Xs0[:, j]), np.nanmean(Xc0[:, j])
    print('   {:24s} {:10.3f} {:10.3f} {:+8.3f} [{:+8.3f}, {:+8.3f}]   {:.3f} [{:.3f}, {:.3f}]'.format(f, ms, mc, pt[j], lo[j], hi[j], pt[NF + j], lo[NF + j], hi[NF + j]))
    rowsB.append([f, ms, mc, pt[j], lo[j], hi[j], pt[NF + j], lo[NF + j], hi[NF + j], int(np.isfinite(Xs0[:, j]).sum()), int(np.isfinite(Xc0[:, j]).sum())])
write_csv('E22_B_features.csv', ['feature', 'mean_start', 'mean_control', 'mean_diff', 'diff_lo', 'diff_hi', 'auc', 'auc_lo', 'auc_hi', 'n_start_valid', 'n_control_valid'], rowsB)
print('   ({} comparisons, no multiplicity adjustment; AUC > 0.5 means larger values at starts. Feature columns tgt_* = template-sized crop at the GT box of the start frame, srch_* = 287 crop at the tracker box of the previous frame; raw = before gamma, gamma = after gamma 0.6.)'.format(NF * 2))

# size terciles over all frames with usable GT
allsz = np.concatenate([np.sqrt(S[s]['g'][S[s]['usable'], 2] * S[s]['g'][S[s]['usable'], 3]) for s in seqs])
q1, q2 = np.percentile(allsz, [100 / 3., 200 / 3.])
print('\n   loss share by GT target-size tercile (sqrt(w h) cut at {:.1f} and {:.1f} px; all frames with a usable GT, frames >= 1):'.format(q1, q2))
rowsT = []
cnt = {}
for s in seqs:
    d = S[s]
    sz = np.sqrt(np.where(d['usable'], d['g'][:, 2] * d['g'][:, 3], np.nan))
    ter = np.where(np.isnan(sz), -1, np.where(sz <= q1, 0, np.where(sz <= q2, 1, 2)))
    ter[0] = -1
    st = np.zeros(len(sz), bool); st[d['starts']] = True
    cnt[s] = np.array([[int((ter == t).sum()), int((d['lost'] & (ter == t)).sum()), int((st & (ter == t)).sum())] for t in range(3)])
for t, nm in enumerate(('small', 'medium', 'large')):
    a = np.array([cnt[s][t] for s in seqs], float)
    bs = []
    for _ in range(NB):
        i = rng_boot.randint(0, len(seqs), len(seqs))
        bs.append(a[i, 1].sum() / a[i, 0].sum())
    print('      {:7s} frames {:7d}  lost share {:.3f} [{:.3f}, {:.3f}]  stretch starts per 1000 frames {:.2f}'.format(
        nm, int(a[:, 0].sum()), a[:, 1].sum() / a[:, 0].sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), 1000 * a[:, 2].sum() / a[:, 0].sum()))
    rowsT.append([nm, int(a[:, 0].sum()), int(a[:, 1].sum()), a[:, 1].sum() / a[:, 0].sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), 1000 * a[:, 2].sum() / a[:, 0].sum()])
write_csv('E22_B_size_tercile.csv', ['tercile', 'frames', 'lost_frames', 'lost_share', 'lo', 'hi', 'starts_per_1000_frames'], rowsT)

# attributes
N = np.array([S[s]['n'] for s in seqs], float)
LOST = np.array([S[s]['lost'].sum() for s in seqs], float)
STARTS = np.array([len(S[s]['starts']) for s in seqs], float)
ATT = np.array([S[s]['att'] for s in seqs])
NA = ATT.shape[1]
print('\n   attribute flags (att00..att{:02d}; the order is unknown): per sequence 0/1, {} sequences'.format(NA - 1, len(seqs)))


def per_seq_stat(s):
    d = S[s]
    u = d['usable']
    sz = np.sqrt(d['g'][u, 2] * d['g'][u, 3])
    sp = [np.hypot(d['gcx'][k] - d['gcx'][k - 1], d['gcy'][k] - d['gcy'][k - 1]) / np.sqrt(d['g'][k, 2] * d['g'][k, 3]) for k in range(1, d['n']) if u[k] and u[k - 1]]
    return [1 - u.mean(), sz.mean(), np.std(np.log(sz)) if len(sz) > 1 else np.nan, np.mean(sp) if sp else np.nan]


STAT = np.array([per_seq_stat(s) for s in seqs])
STAT_NAMES = ['GT-absent share', 'mean GT size (px)', 'std of log GT size', 'mean GT speed (size units/frame)']
rowsAt, rowsId = [], []
print('   {:5s} {:>5s} {:>7s} | {:>14s} {:>14s} {:>20s} | {:>9s} {:>9s} {:>20s}'.format('flag', 'n_seq', 'frames', 'starts/1000 w', 'w/o', 'diff [95%]', 'lost sh w', 'w/o', 'diff [95%]'))
for a in range(NA):
    m = ATT[:, a] == 1
    if m.sum() == 0 or (~m).sum() == 0:
        continue

    def rates(msk):
        return 1000 * STARTS[msk].sum() / N[msk].sum(), LOST[msk].sum() / N[msk].sum()
    r1, r0 = rates(m), rates(~m)
    bs = []
    for _ in range(NB):
        i = rng_boot.randint(0, len(seqs), len(seqs))
        mm = m[i]
        if mm.sum() == 0 or (~mm).sum() == 0:
            continue
        x1 = (1000 * STARTS[i][mm].sum() / N[i][mm].sum(), LOST[i][mm].sum() / N[i][mm].sum())
        x0 = (1000 * STARTS[i][~mm].sum() / N[i][~mm].sum(), LOST[i][~mm].sum() / N[i][~mm].sum())
        bs.append((x1[0] - x0[0], x1[1] - x0[1]))
    bs = np.array(bs)
    l0, h0 = np.percentile(bs[:, 0], [2.5, 97.5]); l1, h1 = np.percentile(bs[:, 1], [2.5, 97.5])
    print('   att{:02d} {:5d} {:7d} | {:14.2f} {:14.2f} {:+7.2f} [{:+5.2f},{:+5.2f}] | {:9.3f} {:9.3f} {:+6.3f} [{:+.3f},{:+.3f}]'.format(
        a, int(m.sum()), int(N[m].sum()), r1[0], r0[0], r1[0] - r0[0], l0, h0, r1[1], r0[1], r1[1] - r0[1], l1, h1))
    rowsAt.append(['att%02d' % a, int(m.sum()), int(N[m].sum()), r1[0], r0[0], r1[0] - r0[0], l0, h0, r1[1], r0[1], r1[1] - r0[1], l1, h1])
    # identification: standardised difference of per-sequence statistics
    for k, nm in enumerate(STAT_NAMES):
        v = STAT[:, k]
        def dstd(msk, vv):
            x1, x0 = vv[msk], vv[~msk]
            x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
            sd = np.sqrt((x1.var() * len(x1) + x0.var() * len(x0)) / (len(x1) + len(x0)))
            return (x1.mean() - x0.mean()) / sd if sd > 0 else np.nan
        d0 = dstd(m, v)
        bd = []
        for _ in range(NB):
            i = rng_boot.randint(0, len(seqs), len(seqs))
            mm = m[i]
            if mm.sum() < 2 or (~mm).sum() < 2:
                continue
            bd.append(dstd(mm, v[i]))
        bl, bh = np.nanpercentile(bd, [2.5, 97.5])
        rowsId.append(['att%02d' % a, nm, np.nanmean(v[m]), np.nanmean(v[~m]), d0, bl, bh, int(abs(d0) >= 1.0 and (bl > 0 or bh < 0))])
write_csv('E22_B_attributes.csv', ['flag', 'n_seq_with', 'frames_with', 'starts_per_1000_with', 'starts_per_1000_without', 'diff', 'lo', 'hi', 'lost_share_with', 'lost_share_without', 'lost_diff', 'lo', 'hi'], rowsAt)
write_csv('E22_B_attribute_identification.csv', ['flag', 'statistic', 'mean_with', 'mean_without', 'standardised_diff', 'lo', 'hi', 'clear(|d|>=1 and interval excludes 0)'], rowsId)
print('\n   data-only identification: standardised difference d of per-sequence statistics (with flag minus without), "clear" = |d| >= 1 and the bootstrap interval excludes 0 (threshold fixed in advance):')
for a in range(NA):
    rr = [r for r in rowsId if r[0] == 'att%02d' % a]
    print('      att{:02d}: '.format(a) + '; '.join('{} d={:+.2f}{}'.format(r[1], r[4], ' CLEAR' if r[7] else '') for r in rr))
print('   share of sequences with each flag: ' + ' '.join('att%02d=%.2f' % (a, ATT[:, a].mean()) for a in range(NA)))

# ----------------------------------------------------------------------------- C
print('\n===== C. Concentration =====')
order = np.argsort(-LOST)
rowsC = []
for k in (5, 10, 20):
    top = order[:k]
    ordst = np.argsort(-STARTS)[:k]
    print('   top {:2d} sequences by lost frames: {:.1f}% of all lost frames; top {:2d} by stretch starts: {:.1f}% of all starts (the {} sequences have {:.1f}% of all frames)'.format(
        k, 100 * LOST[top].sum() / LOST.sum(), k, 100 * STARTS[ordst].sum() / STARTS.sum(), k, 100 * N[top].sum() / N.sum()))
    rowsC.append([k, LOST[top].sum() / LOST.sum(), STARTS[ordst].sum() / STARTS.sum(), N[top].sum() / N.sum()])
write_csv('E22_C_concentration.csv', ['top_k_sequences', 'share_of_lost_frames(top by lost)', 'share_of_starts(top by starts)', 'share_of_frames(top by lost)'], rowsC)
print('   top-20 sequences by lost frames:  seq, frames, lost frames, lost share, starts, GT size, absent share, size std, speed, flags')
rowsTop = []
for i in order[:20]:
    print('      {:7s} {:5d} {:6d} {:.2f} {:3d} {:7.1f} {:.2f} {:.2f} {:.3f}  {}'.format(
        seqs[i], int(N[i]), int(LOST[i]), LOST[i] / N[i], int(STARTS[i]), STAT[i, 1], STAT[i, 0], STAT[i, 2], STAT[i, 3], ''.join(str(v) for v in ATT[i])))
    rowsTop.append([seqs[i], int(N[i]), int(LOST[i]), LOST[i] / N[i], int(STARTS[i])] + list(STAT[i]) + list(ATT[i]))
write_csv('E22_C_top_sequences.csv', ['seq', 'frames', 'lost_frames', 'lost_share', 'starts', 'absent_share', 'mean_size', 'std_log_size', 'mean_speed'] + ['att%02d' % a for a in range(NA)], rowsTop)
top10 = np.zeros(len(seqs), bool); top10[order[:10]] = True
rr = np.random.RandomState(5)
print('   top-10 vs the other {} sequences (per-sequence statistics; permutation test of the difference of means, 2000 permutations):'.format(len(seqs) - 10))
cols = STAT_NAMES + ['frames (log10)'] + ['att%02d share' % a for a in range(NA)]
V = np.column_stack([STAT, np.log10(N), ATT])
for k, nm in enumerate(cols):
    v = V[:, k]
    x1, x0 = v[top10], v[~top10]
    dd = np.nanmean(x1) - np.nanmean(x0)
    perm = []
    for _ in range(2000):
        pm = rr.permutation(top10)
        perm.append(np.nanmean(v[pm]) - np.nanmean(v[~pm]))
    p = (np.sum(np.abs(perm) >= abs(dd)) + 1.) / 2001.
    print('      {:34s} top-10 mean {:9.3f}  others {:9.3f}  diff {:+8.3f}  permutation p {:.3f}'.format(nm, np.nanmean(x1), np.nanmean(x0), dd, p))
print('\ndone')
