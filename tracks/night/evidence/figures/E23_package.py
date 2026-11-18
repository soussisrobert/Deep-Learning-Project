"""E23: per-video CSVs (official pysot OPE code through the NATDataset adapter), cross-checks, figures for night_evidence/. CPU only, existing result folders, no tracking.
Run from tools/:  python -I illumination/E23_package.py
Models (all first.pth, results/<dataset>/<folder>):
  original      no enhancement                         NAT2021 first1.2 | NAT2021L first1.2 | UD135 test/val first1.2_none
  gamma         gamma 0.6 on both crops                NAT2021 first1.2_gamma0.6_180 | NAT2021L first1.2_gamma0.6_L | UD135 test/val first1.2_g06
  gamma_anchor  gamma 0.6 + anchor 0.2, last 3 frames  NAT2021 / NAT2021L first1.2_m_anchor | UD135 test/val first1.2_ag
Numbers: toolkit/evaluation/ope_benchmark.py OPEBenchmark.eval_success / eval_precision (unmodified in git since the first commit that has it) applied to the saved boxes through
toolkit/datasets/nat.py NATDataset (the adapter, written for this project). success_auc = mean of the 21-point success curve; precision20 = precision curve at 20 px.
Cross-checks: (1) tools/eval_nat.py run as a subprocess on the same folders (per-video AUC CSV, 4 decimals; table with success / precision means); (2) own formula (E15 / E20 auc_of).
Figures: headline bar chart; before/after frame strips with the selection rule fixed in advance (see strips()).
"""
import os
import sys
import csv
import json
import shutil
import subprocess
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.getcwd())
from pysot.core.config import cfg
cfg.merge_from_file('../experiments/config.yaml')
cfg.CUDA = False
from toolkit.datasets import NATDataset
from toolkit.evaluation import OPEBenchmark
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.enhance import make_enhancer

ROOT = '../test_dataset'
EV = '../night_evidence'
BENCH = [('NAT2021', 'NAT2021', {'original': 'first1.2', 'gamma': 'first1.2_gamma0.6_180', 'gamma_anchor': 'first1.2_m_anchor'}, 180),
         ('NAT2021L', 'NAT2021L', {'original': 'first1.2', 'gamma': 'first1.2_gamma0.6_L', 'gamma_anchor': 'first1.2_m_anchor'}, 23),
         ('UAVDark135_test', 'NATUD135TEST', {'original': 'first1.2_none', 'gamma': 'first1.2_g06', 'gamma_anchor': 'first1.2_ag'}, 23),
         ('UAVDark135_val', 'NATUD135VAL', {'original': 'first1.2_none', 'gamma': 'first1.2_g06', 'gamma_anchor': 'first1.2_ag'}, 16)]
LOGGED = {'NAT2021': (0.372, 0.417, 0.453), 'NAT2021L': (0.299, 0.316, 0.377), 'UAVDark135_test': (0.352, 0.434, 0.443), 'UAVDark135_val': (0.4216, None, 0.4934)}
MODELS = ['original', 'gamma', 'gamma_anchor']
THR = np.arange(0, 1.05, 0.05)


def own_auc(gt, pred):
    ax2, ay2, bx2, by2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3], pred[:, 0] + pred[:, 2], pred[:, 1] + pred[:, 3]
    iou = np.ones(len(gt)) * -1
    m = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(gt[:, 0], pred[:, 0]))[m]
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(gt[:, 1], pred[:, 1]))[m]
    inter = iw * ih
    iou[m] = np.clip(inter / (gt[m, 2] * gt[m, 3] + pred[m, 2] * pred[m, 3] - inter), 0, 1)
    return float(np.mean([(iou > t).sum() / float(len(iou)) for t in THR]))


per_video = {}
rows_chk, rows_abs = [], []
for bname, ds, folders, nvid in BENCH:
    for mname, folder in folders.items():
        p = os.path.join('results', ds, folder)
        assert os.path.isdir(p), p
    dataset = NATDataset(ds, os.path.join(ROOT, ds))
    assert len(dataset.videos) == nvid, (ds, len(dataset.videos))
    trackers = [folders[m] for m in MODELS]
    for t in trackers:
        n = len([f for f in os.listdir(os.path.join('results', ds, t)) if f.endswith('.txt')])
        assert n >= nvid, (ds, t, n)
    dataset.set_tracker('results/' + ds, trackers)
    bm = OPEBenchmark(dataset)
    succ, prec = {}, {}
    for t in trackers:
        succ.update(bm.eval_success(t))
        prec.update(bm.eval_precision(t))
    seqs = sorted(dataset.videos)
    # cross-check 1: eval_nat.py as a subprocess
    csv_path = os.path.join('illumination', 'E23_evalnat_{}.csv'.format(ds))
    out = subprocess.run([sys.executable, 'eval_nat.py', '--dataset', ds, '--trackers'] + trackers + ['--csv', csv_path], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    en = {}
    for r in csv.DictReader(open(csv_path)):
        en[r['seq']] = r
    table = {}
    for line in out.stdout.splitlines():
        if line.startswith('|') and 'Tracker' not in line:
            c = [x.strip() for x in line.strip('|').split('|')]
            if len(c) == 4:
                table[c[0]] = [float(x) for x in c[1:]]
    rows = []
    for mname in MODELS:
        f = folders[mname]
        auc = np.array([float(np.mean(succ[f][s])) for s in seqs])
        p20 = np.array([float(prec[f][s][20]) for s in seqs])
        # own formula
        own = np.array([own_auc(np.array(dataset.videos[s].gt_traj, dtype=float), np.loadtxt(os.path.join('results', ds, f, s + '.txt'), delimiter=',', ndmin=2)) for s in seqs])
        evn = np.array([float(en[s][f]) for s in seqs])
        for s, a, q in zip(seqs, auc, p20):
            rows.append([bname, s, len(dataset.videos[s].gt_traj), mname, 0, '{:.6f}'.format(a), '{:.6f}'.format(q), 'official'])
        # precision with absent GT frames counted as misses (own variant)
        def prec_absent_miss(s):
            g = np.array(dataset.videos[s].gt_traj, dtype=float)
            pr = np.loadtxt(os.path.join('results', ds, f, s + '.txt'), delimiter=',', ndmin=2)
            gc = np.array([g[:, 0] + (g[:, 2] - 1) / 2, g[:, 1] + (g[:, 3] - 1) / 2]).T
            pc = np.array([pr[:, 0] + (pr[:, 2] - 1) / 2, pr[:, 1] + (pr[:, 3] - 1) / 2]).T
            ok = np.isfinite(gc).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
            d = np.hypot(*(gc - pc).T)
            return float(((d <= 20) & ok).sum() / len(g))
        pm = np.array([prec_absent_miss(s) for s in seqs])
        nfr = np.array([len(dataset.videos[s].gt_traj) for s in seqs], float)
        absent = np.array([int((~(np.isfinite(np.array(dataset.videos[s].gt_traj, dtype=float)).all(axis=1) & (np.array(dataset.videos[s].gt_traj, dtype=float)[:, 2] > 0) & (np.array(dataset.videos[s].gt_traj, dtype=float)[:, 3] > 0))).sum()) for s in seqs])
        tab = table.get(f, [np.nan] * 3)
        rows_chk.append([bname, mname, f, len(seqs), auc.mean(), p20.mean(), np.abs(auc - evn).max(), np.abs(auc - own).max(), tab[0], tab[2], abs(auc.mean() - tab[0]), abs(p20.mean() - tab[2])])
        rows_abs.append([bname, mname, f, int(absent.sum()), int(nfr.sum()), absent.sum() / nfr.sum(), p20.mean(), pm.mean(), p20.mean() - pm.mean()])
        print('{:16s} {:13s} {:24s} mean success {:.4f} (logged {})  precision20 {:.4f} | max|official - eval_nat CSV| {:.1e}  max|official - own| {:.1e}  | eval_nat table success {} precision {}'.format(
            bname, mname, f, auc.mean(), LOGGED[bname][MODELS.index(mname)], p20.mean(), np.abs(auc - evn).max(), np.abs(auc - own).max(), tab[0], tab[2]))
        lg = LOGGED[bname][MODELS.index(mname)]
        if lg is not None and abs(auc.mean() - lg) > 0.002:
            print('MISMATCH with the logged mean for', bname, mname)
            sys.exit(1)
    per_video[bname] = rows
    with open(os.path.join(EV, 'per_video_{}.csv'.format(bname)), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['benchmark', 'video', 'frames', 'model', 'seed', 'success_auc', 'precision20', 'toolkit'])
        w.writerows(rows)
    print('wrote', os.path.join(EV, 'per_video_{}.csv'.format(bname)), len(rows), 'rows')

with open(os.path.join(EV, 'official_vs_own_check.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['benchmark', 'model', 'folder', 'n_videos', 'mean_success_official', 'mean_precision20_official', 'max_abs_diff_vs_eval_nat_csv', 'max_abs_diff_vs_own_formula',
                'eval_nat_table_success', 'eval_nat_table_precision', 'abs_diff_mean_success', 'abs_diff_mean_precision'])
    w.writerows(rows_chk)
with open(os.path.join(EV, 'absent_gt_handling.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['benchmark', 'model', 'folder', 'frames_without_usable_gt', 'frames', 'share', 'mean_precision20_official', 'mean_precision20_absent_counted_as_miss', 'difference'])
    w.writerows(rows_abs)
print('max over all rows: |official - eval_nat CSV| {:.2e}, |official - own formula| {:.2e}, |mean success - eval_nat table| {:.4f}, |mean precision - eval_nat table| {:.4f}'.format(
    max(r[6] for r in rows_chk), max(r[7] for r in rows_chk), max(r[10] for r in rows_chk), max(r[11] for r in rows_chk)))

# ------------------------------------------------------------------ figures
# headline bars (NAT2021, no oracle bar)
a = {m: np.mean([float(r[5]) for r in per_video['NAT2021'] if r[3] == m]) for m in MODELS}
fig, ax = plt.subplots(figsize=(5.2, 4))
bars = ax.bar(['original', 'gamma 0.6', 'gamma 0.6 + anchor'], [a[m] for m in MODELS], color=['#9aa0a6', '#4c78a8', '#2a9d8f'])
for b, m in zip(bars, MODELS):
    ax.text(b.get_x() + b.get_width() / 2, a[m] + 0.006, '{:.3f}'.format(a[m]), ha='center', fontsize=10)
ax.set_ylabel('mean success AUC, NAT2021 (180 videos)')
ax.set_ylim(0, 0.5)
ax.set_title('first.pth, test-time settings only')
plt.tight_layout()
plt.savefig(os.path.join(EV, 'figures', 'fig_headline_bars.png'), dpi=130)
plt.close()
print('wrote figures/fig_headline_bars.png', {m: round(v, 4) for m, v in a.items()})

# strips. Selection rule, fixed in advance (NAT2021, per-video AUC gain = gamma_anchor - original, official numbers):
#   3 largest gains among videos with >= 300 frames; the video at rank n // 2 of the gains sorted ascending (median gain); the video with the smallest gain (largest loss).
#   Frames: the longest run of consecutive frames where IoU(original box, gamma_anchor box) < 0.5 (the two models differ); 4 frames evenly spaced over that run (np.linspace, rounded).
#   Crop: square of side 3 * max(GT w, h) of that frame (at least 48 px) around the GT centre; top row raw, bottom row the same crop after gamma 0.6; GT green, original red, gamma_anchor blue.
rows = per_video['NAT2021']
vids = sorted({r[1] for r in rows})
gain = {v: float([r for r in rows if r[1] == v and r[3] == 'gamma_anchor'][0][5]) - float([r for r in rows if r[1] == v and r[3] == 'original'][0][5]) for v in vids}
nfr = {v: int([r for r in rows if r[1] == v][0][2]) for v in vids}
order = sorted(vids, key=lambda v: gain[v])
big = [v for v in sorted(vids, key=lambda v: -gain[v]) if nfr[v] >= 300][:3]
pick = [('gain%d' % (i + 1), v) for i, v in enumerate(big)] + [('median', order[len(order) // 2]), ('loss', order[0])]
print('strip videos:', [(k, v, round(gain[v], 3), nfr[v]) for k, v in pick])


class _M(object):
    def eval(self):
        return self


tracker = HiFTTracker(_M())
enh = make_enhancer('gamma', 0.6, 3.0, 'none')
meta = json.load(open(os.path.join(ROOT, 'NAT2021', 'NAT2021.json')))


def iou1(a, b):
    iw = np.maximum(0, np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0]))
    ih = np.maximum(0, np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1]))
    inter = iw * ih
    return inter / (a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter + 1e-12)


strip_info = []
for key, v in pick:
    g = np.array(meta[v]['gt_rect'], dtype=float)
    po = np.loadtxt(os.path.join('results', 'NAT2021', 'first1.2', v + '.txt'), delimiter=',', ndmin=2)
    pg = np.loadtxt(os.path.join('results', 'NAT2021', 'first1.2_m_anchor', v + '.txt'), delimiter=',', ndmin=2)
    diff = iou1(po, pg) < 0.5
    best, cur_s, bs, bl = None, None, 0, 0
    t = 0
    while t < len(diff):
        if diff[t]:
            e = t
            while e + 1 < len(diff) and diff[e + 1]:
                e += 1
            if e - t + 1 > bl:
                bs, bl = t, e - t + 1
            t = e + 1
        else:
            t += 1
    if bl == 0:
        bs, bl = int(np.argmin(iou1(po, pg))), 1
    frames = sorted(set(int(round(x)) for x in np.linspace(bs, bs + bl - 1, 4)))
    avg = np.mean(cv2.imread(os.path.join(ROOT, 'NAT2021', meta[v]['img_names'][0])), axis=(0, 1))
    fig, axes = plt.subplots(2, len(frames), figsize=(3.0 * len(frames), 6.4), squeeze=False)
    last = None
    for j, f in enumerate(frames):
        ok = np.isfinite(g[f]).all() and g[f, 2] > 0 and g[f, 3] > 0
        if ok:
            last = (g[f, 0] + g[f, 2] / 2., g[f, 1] + g[f, 3] / 2., g[f, 2], g[f, 3])
        ctr = last if last is not None else (po[f, 0] + po[f, 2] / 2., po[f, 1] + po[f, 3] / 2., po[f, 2], po[f, 3])
        side = int(max(48, round(3 * max(ctr[2], ctr[3]))))
        img = cv2.imread(os.path.join(ROOT, 'NAT2021', meta[v]['img_names'][f]))
        patch = np.ascontiguousarray(tracker.get_subwindow(img, np.array(ctr[:2]), side, side, avg, None).numpy()[0].transpose(1, 2, 0)).astype(np.uint8)
        xm, ym = np.floor(ctr[0] - (side + 1) / 2. + 0.5), np.floor(ctr[1] - (side + 1) / 2. + 0.5)
        for r_, im in enumerate((patch, enh(np.ascontiguousarray(patch)))):
            axx = axes[r_][j]
            axx.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if ok:
                axx.add_patch(Rectangle((g[f, 0] - xm, g[f, 1] - ym), g[f, 2], g[f, 3], fill=False, ec='#00cc00', lw=1.6))
            axx.add_patch(Rectangle((po[f, 0] - xm, po[f, 1] - ym), po[f, 2], po[f, 3], fill=False, ec='#ff2222', lw=1.3))
            axx.add_patch(Rectangle((pg[f, 0] - xm, pg[f, 1] - ym), pg[f, 2], pg[f, 3], fill=False, ec='#2255ff', lw=1.3, ls='--'))
            axx.set_xticks([]); axx.set_yticks([])
            axx.set_title(('frame %d' % f) + (' (GT absent)' if not ok else '') + ('  raw' if r_ == 0 else '  gamma 0.6'), fontsize=8)
    fig.suptitle('{} ({}): AUC original {:.3f} -> gamma_anchor {:.3f}; GT green, original red, gamma_anchor blue (a box outside the crop is not visible); differing run frames {}-{}'.format(
        v, key, float([r for r in rows if r[1] == v and r[3] == 'original'][0][5]), float([r for r in rows if r[1] == v and r[3] == 'gamma_anchor'][0][5]), bs, bs + bl - 1), fontsize=8)
    plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    fn = 'strip_{}_{}.png'.format(key, v)
    plt.savefig(os.path.join(EV, 'figures', fn), dpi=90)
    plt.close()
    strip_info.append([key, v, nfr[v], gain[v], bs, bl, ' '.join(str(x) for x in frames), fn])
with open(os.path.join(EV, 'figures', 'strips_index.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['kind', 'video', 'frames', 'auc_gain', 'differing_run_start', 'differing_run_length', 'frames_shown', 'file'])
    w.writerows(strip_info)
shutil.copy('illumination/E23_package.py', os.path.join(EV, 'figures', 'E23_package.py'))
print('wrote strips', [r[7] for r in strip_info])
