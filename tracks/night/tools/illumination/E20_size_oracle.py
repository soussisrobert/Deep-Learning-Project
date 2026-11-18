"""E20: box-size oracle on every frame (CPU only, saved boxes + GT, no tracking).
Boxes: first.pth + anchor 0.2 + gamma 0.6. Folder per set: NAT2021, NAT2021L, NATUD135VAL -> results/<ds>/first1.2_m_anchor;
NATUD135TEST -> results/NATUD135TEST/first1.2_ag (no m_anchor folder exists there; use agreed with the user).
Step 0: per-sequence AUC and box comparison of first1.2_ag against first1.2_m_anchor where both exist (NATUD135VAL); for NAT2021 / NAT2021L there is no
        first1.2_ag folder, the corresponding anchor+gamma run first1.2_p15_a0.2_180 / first1.2_p15_a0.2_L is compared instead (labelled as such).
Step 1: scoring as toolkit/eval_nat.py (toolkit/utils/statistics.py success_overlap): mask = GT w > 0 and h > 0 (NaN fails the test); IoU = -1 where the mask is
        false; success at each of the 21 thresholds 0, 0.05 .. 1 = share of ALL frames with IoU > threshold (n_frame = all frames, so frames without a usable
        GT count as failures); AUC = mean over thresholds. Baseline must match 0.453 / 0.377 / 0.493 / 0.443 within 0.002, otherwise exit 1.
Step 2: variants (tracker box kept unless said otherwise; frames without a usable GT cannot be altered and stay failures):
   a size oracle: tracker centre + GT width and height, all frames;   b centre oracle: GT centre + tracker width and height, all frames;
   c size oracle on found frames only (centre error <= 20 px);       d size oracle on lost frames only (centre error > 20 px).
   centre = (x + w / 2, y + h / 2); error = distance of the tracker centre to the GT centre; found / lost need a usable GT; frame 0 (the init box) is a found frame.
Step 3: paired bootstrap over sequences against the baseline, 2000 resamples, RandomState(0), 95% percentile interval.
Step 4: on found frames t >= 1 (usable GT, error <= 20 px) for the baseline: mean IoU, median / 5th / 95th percentile of pred width / GT width and pred height /
        GT height, share of frames with pred area / GT area > 1.5 and < 0.67; per dataset and pooled over the four sets.
usage (from tools/): python -I illumination/E20_size_oracle.py
"""
import os
import sys
import csv
import json
import numpy as np

OUT = 'illumination'
ROOT = '../test_dataset'
SETS = [('NAT2021', 'first1.2_m_anchor', 0.453), ('NAT2021L', 'first1.2_m_anchor', 0.377),
        ('NATUD135VAL', 'first1.2_m_anchor', 0.493), ('NATUD135TEST', 'first1.2_ag', 0.443)]
THR = np.arange(0, 1.05, 0.05)
PX = 20.0
NB = 2000


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


def auc_of(gt, pred):
    iou = iou_of(gt, pred)
    return float(np.mean([(iou > t).sum() / float(len(iou)) for t in THR]))


def load_gt(ds):
    meta = json.load(open(os.path.join(ROOT, ds, ds + '.json')))
    seqs = sorted(meta)
    return seqs, {s: np.array(meta[s]['gt_rect'], dtype=float) for s in seqs}


def pred(ds, run, s):
    return np.loadtxt(os.path.join('results', ds, run, s + '.txt'), delimiter=',', ndmin=2)


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.RandomState(0)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(NB)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# ---------------------------------------------------------------- folder check (all four listed folders must exist)
missing = [os.path.join('results', ds, run) for ds, run, _ in SETS if not os.path.isdir(os.path.join('results', ds, run))]
if missing:
    print('MISSING result folders:', missing)
    sys.exit(1)

print('folders used:')
for ds, run, _ in SETS:
    print('   {:13s} results/{}/{}'.format(ds, ds, run))

# ---------------------------------------------------------------- step 0
print('\nstep 0: first1.2_ag vs first1.2_m_anchor, per-sequence success AUC and boxes')
chk = []
for ds, a, b, note in (('NAT2021', 'first1.2_p15_a0.2_180', 'first1.2_m_anchor', 'first1.2_ag does not exist on this set; p15_a0.2_180 (E10 anchor 0.2 + gamma 0.6) used instead'),
                       ('NAT2021L', 'first1.2_p15_a0.2_L', 'first1.2_m_anchor', 'first1.2_ag does not exist on this set; p15_a0.2_L (anchor 0.2 + gamma 0.6) used instead'),
                       ('NATUD135VAL', 'first1.2_ag', 'first1.2_m_anchor', 'both folders exist')):
    seqs, gt = load_gt(ds)
    if not (os.path.isdir(os.path.join('results', ds, a)) and os.path.isdir(os.path.join('results', ds, b))):
        print('   {}: folder missing, skipped'.format(ds))
        continue
    da = np.array([auc_of(gt[s], pred(ds, a, s)) for s in seqs])
    db = np.array([auc_of(gt[s], pred(ds, b, s)) for s in seqs])
    boxd = max(float(np.abs(pred(ds, a, s) - pred(ds, b, s)).max()) for s in seqs)
    nd = int((np.abs(da - db) > 1e-12).sum())
    print('   {:12s} {} vs {}: mean AUC {:.4f} vs {:.4f}; max |per-sequence AUC difference| {:.2e}; sequences with any difference {} of {}; max |box coordinate difference| {:.2e}  ({})'.format(
        ds, a, b, da.mean(), db.mean(), np.abs(da - db).max(), nd, len(seqs), boxd, note))
    chk.append([ds, a, b, len(seqs), da.mean(), db.mean(), float(np.abs(da - db).max()), nd, boxd, note])

# ---------------------------------------------------------------- step 1
data = {}
bad = False
print('\nstep 1: baseline reproduction (tolerance 0.002)')
for ds, run, target in SETS:
    seqs, gt = load_gt(ds)
    P = {s: pred(ds, run, s) for s in seqs}
    for s in seqs:
        assert len(P[s]) == len(gt[s]), (ds, s)
    base = np.array([auc_of(gt[s], P[s]) for s in seqs])
    nan_frames = sum(int((~((gt[s][:, 2] > 0) & (gt[s][:, 3] > 0))).sum()) for s in seqs)
    tot = sum(len(gt[s]) for s in seqs)
    ok = abs(base.mean() - target) <= 0.002
    bad |= not ok
    print('   {:13s} {} sequences, mean success AUC {:.4f} (expected {:.3f})  {}   frames without a usable GT (NaN or w/h <= 0): {} of {} ({:.2f}%)'.format(
        ds, len(seqs), base.mean(), target, 'ok' if ok else 'MISMATCH', nan_frames, tot, 100. * nan_frames / tot))
    data[ds] = (seqs, gt, P, base)
if bad:
    print('STOP: baseline differs from the expected value by more than 0.002')
    sys.exit(1)


def variants(g, p):
    usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
    gcx, gcy = g[:, 0] + g[:, 2] / 2., g[:, 1] + g[:, 3] / 2.
    pcx, pcy = p[:, 0] + p[:, 2] / 2., p[:, 1] + p[:, 3] / 2.
    err = np.hypot(pcx - gcx, pcy - gcy)
    found = usable & (err <= PX)
    lost = usable & (err > PX)
    size = p.copy()                      # tracker centre, GT width and height
    size[usable, 2], size[usable, 3] = g[usable, 2], g[usable, 3]
    size[usable, 0], size[usable, 1] = pcx[usable] - g[usable, 2] / 2., pcy[usable] - g[usable, 3] / 2.
    cen = p.copy()                       # GT centre, tracker width and height
    cen[usable, 0], cen[usable, 1] = gcx[usable] - p[usable, 2] / 2., gcy[usable] - p[usable, 3] / 2.
    c = p.copy(); c[found] = size[found]
    d = p.copy(); d[lost] = size[lost]
    return {'a size oracle, all frames': size, 'b centre oracle, all frames': cen,
            'c size oracle, found frames only': c, 'd size oracle, lost frames only': d}, found, lost, usable


# ---------------------------------------------------------------- steps 2-3
print('\nsteps 2-3: success AUC of the variants and paired bootstrap against the baseline')
rows = []
for ds, run, _ in SETS:
    seqs, gt, P, base = data[ds]
    res = {}
    nf = nl = 0
    for s in seqs:
        v, found, lost, usable = variants(gt[s], P[s])
        nf += int(found.sum()); nl += int(lost.sum())
        for k, box in v.items():
            res.setdefault(k, []).append(auc_of(gt[s], box))
    print('\n=== {} (folder {}, {} sequences; frames: found {} lost {} of {})'.format(ds, run, len(seqs), nf, nl, sum(len(gt[s]) for s in seqs)))
    print('   baseline {:.4f}'.format(base.mean()))
    rows.append([ds, run, 'baseline', len(seqs), base.mean(), 0, 0, 0, 0, 0, 0, ''])
    for k, v in res.items():
        v = np.array(v)
        d = v - base
        m, lo, hi = boot(d)
        verdict = 'differs' if (lo > 0 or hi < 0) else 'within noise'
        print('   {:36s} {:.4f}  gain {:+.4f} [{:+.4f}, {:+.4f}]  better {:3d} worse {:3d} equal {:3d}  {}'.format(
            k, v.mean(), m, lo, hi, int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), verdict))
        rows.append([ds, run, k, len(seqs), v.mean(), m, lo, hi, int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum()), verdict])

# ---------------------------------------------------------------- step 4
print('\nstep 4: baseline box size on found frames (t >= 1, usable GT, centre error <= 20 px)')
srows = []
pool = {'iou': [], 'wr': [], 'hr': [], 'ar': []}
for ds, run, _ in SETS + [('ALL', '', 0)]:
    if ds == 'ALL':
        iou, wr, hr, ar = [np.concatenate(pool[k]) for k in ('iou', 'wr', 'hr', 'ar')]
    else:
        seqs, gt, P, base = data[ds]
        I, W, H, A = [], [], [], []
        for s in seqs:
            g, p = gt[s], P[s]
            _, found, _, _ = variants(g, p)
            found[0] = False
            I.append(iou_of(g, p)[found])
            W.append(p[found, 2] / g[found, 2]); H.append(p[found, 3] / g[found, 3])
            A.append((p[found, 2] * p[found, 3]) / (g[found, 2] * g[found, 3]))
        iou, wr, hr, ar = [np.concatenate(x) for x in (I, W, H, A)]
        for k, x in zip(('iou', 'wr', 'hr', 'ar'), (iou, wr, hr, ar)):
            pool[k].append(x)
    q = lambda x: (np.median(x), np.percentile(x, 5), np.percentile(x, 95))
    print('   {:13s} found frames {:7d}  mean IoU {:.3f}  width ratio median {:.3f} [5th {:.3f}, 95th {:.3f}]  height ratio median {:.3f} [{:.3f}, {:.3f}]  area > 1.5x GT {:.1f}%  area < 0.67x GT {:.1f}%'.format(
        ds, len(iou), iou.mean(), *(q(wr) + q(hr)), 100. * (ar > 1.5).mean(), 100. * (ar < 0.67).mean()))
    srows.append([ds, len(iou), iou.mean(), *(q(wr) + q(hr)), np.median(ar), (ar > 1.5).mean(), (ar < 0.67).mean()])

for name, header, body in (
        ('E20_size_oracle.csv', ['dataset', 'folder', 'variant', 'n_sequences', 'success_auc', 'gain_vs_baseline', 'lo', 'hi', 'better', 'worse', 'equal', 'verdict'], rows),
        ('E20_size_ratio.csv', ['dataset', 'found_frames', 'mean_iou', 'width_ratio_median', 'width_ratio_p5', 'width_ratio_p95', 'height_ratio_median',
                                'height_ratio_p5', 'height_ratio_p95', 'area_ratio_median', 'share_area_gt_1.5', 'share_area_lt_0.67'], srows),
        ('E20_folder_check.csv', ['dataset', 'folder_a', 'folder_b', 'n_sequences', 'mean_auc_a', 'mean_auc_b', 'max_abs_auc_diff', 'n_sequences_differing', 'max_abs_box_diff', 'note'], chk)):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    print('wrote', os.path.join(OUT, name))
