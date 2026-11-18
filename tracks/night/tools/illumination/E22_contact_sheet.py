"""E22 D: contact sheet of 60 stretch starts with the tracker's own view. CPU only. Run from tools/: python -I illumination/E22_contact_sheet.py
Selection (assumption, the original selection rule was not in the task text I received): 30 stretch starts whose category from A is 'runner-up' (2d or 2x logged peak within 20 px of the GT)
and 30 whose category is 'not a logged peak', drawn at random (RandomState(7)) among starts with t >= 6 and t + 5 <= last frame.
One PNG per start in illumination/E22_contact/. Top row: frames t-5, t-1, t, t+5, a crop of 3 x max(GT w, h) (at least 48 px) around the GT centre of that frame, shown after gamma 0.6
(for visibility), GT green, top-1 red, 2d blue, 2x yellow (logged boxes of that frame). Bottom row: the 287 px search crop of frame t-1 and of frame t exactly as the tracker builds it
(state = the tracker box of the previous frame, get_subwindow with round(s_x), frame-0 channel average, gamma 0.6), at native size, with the GT box and the logged peaks of that frame mapped
into the crop (u = (x - xmin) * 287 / S, xmin = floor(centre - (S + 1) / 2 + 0.5)); then the frame-1 template crop (127 px, gamma 0.6) with its GT box; last panel: text.
Verification of the crop reconstruction: the real HiFTTracker.track() code is run up to the network call with a stub model that captures the search crop (state restored from the saved box),
and the captured crop is compared with the reconstruction; the template crop is compared with HiFTTracker.init's own template.
"""
import os
import sys
import csv
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath('..'))
from pysot.core.config import cfg
cfg.merge_from_file('../experiments/config.yaml')
cfg.CUDA = False
cfg.TRACK.ENH_TEMPLATE = 'gamma'
cfg.TRACK.ENH_SEARCH = 'gamma'
cfg.TRACK.ENH_GAMMA = 0.6
cfg.TRACK.ENH_CLIP = 3.0
cfg.TRACK.ENH_DENOISE = 'none'
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.enhance import make_enhancer
from pysot.utils.bbox import get_axis_aligned_bbox

DS = 'NAT2021'
ROOT = '../test_dataset/' + DS
RUN = 'first1.2_m_anchor'
OUT = 'illumination'
PNG_DIR = os.path.join(OUT, 'E22_contact')
PX = 20.0


class Stub(object):
    """model stub: records the template / search crop that the tracker code builds"""
    def eval(self):
        return self

    def template(self, z):
        self.z = z

    def track(self, x):
        self.x = x
        raise RuntimeError('captured')


stub = Stub()
tracker = HiFTTracker(stub)
enh = make_enhancer('gamma', 0.6, 3.0, 'none')


def to_img(t):
    return np.ascontiguousarray(t.numpy()[0].transpose(1, 2, 0)).astype(np.uint8)


def search_geom(p_prev, scaleaa, H, W):
    w, h = p_prev[2], p_prev[3]
    pos = np.array([p_prev[0] + w / 2., p_prev[1] + h / 2.])
    wz = w + cfg.TRACK.CONTEXT_AMOUNT * (w + h); hz = h + cfg.TRACK.CONTEXT_AMOUNT * (w + h)
    sz = np.sqrt(wz * hz)
    if w * h > 0.5 * H * W:
        sz = scaleaa
    sx = sz * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    return pos, round(sx)


def search_crop(img, p_prev, avg, scaleaa):
    H, W = img.shape[:2]
    pos, S = search_geom(p_prev, scaleaa, H, W)
    return to_img(tracker.get_subwindow(img, pos, cfg.TRACK.INSTANCE_SIZE, S, avg, enh)), pos, S


def xmin_of(pos, S):
    return np.floor(pos[0] - (S + 1) / 2. + 0.5), np.floor(pos[1] - (S + 1) / 2. + 0.5)


meta = json.load(open(os.path.join(ROOT, DS + '.json')))
seqs = sorted(meta)
S_ = {}
for s in seqs:
    g = np.array(meta[s]['gt_rect'], dtype=float)
    p = np.loadtxt(os.path.join('results', DS, RUN, s + '.txt'), delimiter=',', ndmin=2)
    usable = np.isfinite(g).all(axis=1) & (g[:, 2] > 0) & (g[:, 3] > 0)
    gcx, gcy = g[:, 0] + g[:, 2] / 2., g[:, 1] + g[:, 3] / 2.
    rows = list(csv.DictReader(open(os.path.join('margin', DS, RUN, s + '.csv'))))
    pk = {}
    for k, key in (('1', 'cx1'), ('2d', 'cx2d'), ('2x', 'cx2x')):
        suffix = key[2:]
        pk[k] = np.full((len(g), 4), np.nan)
        pk[k][1:] = np.array([[float(r['cx' + suffix]), float(r['cy' + suffix]), float(r['w' + suffix]), float(r['h' + suffix])] for r in rows])
    S_[s] = dict(g=g, p=p, usable=usable, gcx=gcx, gcy=gcy, pk=pk, n=len(g), names=meta[s]['img_names'])

stretches = []
for r in csv.DictReader(open(os.path.join(OUT, 'E15_stretches.csv'))):
    if r['dataset'] == DS:
        stretches.append((r['seq'], int(r['start']), int(r['length'])))


def cat_of(s, t):
    d = S_[s]
    e = {k: np.hypot(d['pk'][k][t, 0] - d['gcx'][t], d['pk'][k][t, 1] - d['gcy'][t]) for k in d['pk']}
    return 'runner-up' if (e['2d'] <= PX or e['2x'] <= PX) else 'not-a-logged-peak'


cand = {'runner-up': [], 'not-a-logged-peak': []}
for s, t, L in stretches:
    d = S_[s]
    if t >= 6 and t + 5 <= d['n'] - 1 and d['usable'][t]:
        cand[cat_of(s, t)].append((s, t, L))
rng = np.random.RandomState(7)
chosen = []
for c in cand:
    idx = rng.permutation(len(cand[c]))[:30]
    chosen += [(c,) + cand[c][i] for i in sorted(idx)]
print('eligible starts: runner-up {}, not-a-logged-peak {}; chosen {}'.format(len(cand['runner-up']), len(cand['not-a-logged-peak']), len(chosen)))

# ---------------------------------------------------------------- verification with the tracker's own code
print('\nverification of the crop reconstruction (real HiFTTracker code, stub model captures the crop):')
n_ok = n_tot = 0
for c, s, t, L in chosen[:6] + chosen[-3:]:
    d = S_[s]
    img0 = cv2.imread(os.path.join(ROOT, d['names'][0]))
    cx0, cy0, w0, h0 = get_axis_aligned_bbox(d['g'][0])
    box0 = [cx0 - (w0 - 1) / 2, cy0 - (h0 - 1) / 2, w0, h0]
    tracker.init(img0, box0)
    z_tracker = stub.z.numpy()
    # reconstruction of the template crop
    avg = np.mean(img0, axis=(0, 1))
    sz0 = round(np.sqrt((w0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0)) * (h0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0))))
    z_mine = tracker.get_subwindow(img0, np.array([box0[0] + (box0[2] - 1) / 2, box0[1] + (box0[3] - 1) / 2]), cfg.TRACK.EXEMPLAR_SIZE, sz0, avg, enh).numpy()
    ok_z = np.array_equal(z_tracker, z_mine)
    for f in (t - 1, t):
        img = cv2.imread(os.path.join(ROOT, d['names'][f]))
        p_prev = d['p'][f - 1]
        tracker.center_pos = np.array([p_prev[0] + p_prev[2] / 2., p_prev[1] + p_prev[3] / 2.])
        tracker.size = np.array([p_prev[2], p_prev[3]])
        try:
            tracker.track(img)
        except RuntimeError:
            pass
        x_tracker = stub.x.numpy()[0].transpose(1, 2, 0)
        x_mine = search_crop(img, p_prev, avg, tracker.scaleaa)[0]
        ok = np.array_equal(x_tracker.astype(np.uint8), x_mine)
        n_tot += 1
        n_ok += int(ok)
        print('   {} frame {}: search crop identical to the tracker code\'s crop: {}; template crop identical: {}'.format(s, f, ok, ok_z))
print('   {} of {} search crops identical (state restored from the saved box)'.format(n_ok, n_tot))
# float round trip of the state: how close are crop positions to a rounding boundary?
near = tot = 0
for s in seqs:
    d = S_[s]
    for f in range(1, d['n']):
        pr = d['p'][f - 1]
        pos = np.array([pr[0] + pr[2] / 2., pr[1] + pr[3] / 2.])
        _, S = search_geom(pr, 0, 1e9, 1e9)
        for v in pos:
            tot += 1
            r = (v - (S + 1) / 2. + 0.5) % 1.0
            near += int(min(r, 1 - r) < 1e-9)
print('   float round trip: {} of {} crop-position coordinates lie within 1e-9 of a rounding boundary (a flip would need that)'.format(near, tot))
# consistency of the geometry: the logged top-1 box centre mapped into the crop vs the grid cell it came from
dev = []
for c, s, t, L in chosen:
    d = S_[s]
    img = cv2.imread(os.path.join(ROOT, d['names'][t]))
    img0_ = cv2.imread(os.path.join(ROOT, d['names'][0]))
    avg = np.mean(img0_, axis=(0, 1))
    cx0_, cy0_, w0_, h0_ = get_axis_aligned_bbox(d['g'][0])
    sa_ = round(np.sqrt((w0_ + cfg.TRACK.CONTEXT_AMOUNT * (w0_ + h0_)) * (h0_ + cfg.TRACK.CONTEXT_AMOUNT * (w0_ + h0_))))
    _, pos, Ssz = search_crop(img, d['p'][t - 1], avg, sa_)
    xm, ym = xmin_of(pos, Ssz)
    f_ = cfg.TRACK.INSTANCE_SIZE / float(Ssz)
    u = (d['pk']['1'][t, 0] - xm) * f_
    v = (d['pk']['1'][t, 1] - ym) * f_
    dev.append((u, v))
dev = np.array(dev)
print('   mapped top-1 centres in the 287 crop: u in [{:.0f}, {:.0f}], v in [{:.0f}, {:.0f}] (all inside the crop: {})'.format(
    dev[:, 0].min(), dev[:, 0].max(), dev[:, 1].min(), dev[:, 1].max(), bool(((dev >= 0) & (dev <= 287)).all())))

# ---------------------------------------------------------------- figures
os.makedirs(PNG_DIR, exist_ok=True)
COL = {'gt': '#00cc00', '1': '#ff2222', '2d': '#2255ff', '2x': '#ffcc00'}


def draw_boxes(ax, d, f, to_xy, scale, show_peaks=True):
    if d['usable'][f]:
        u, v = to_xy(d['gcx'][f], d['gcy'][f])
        w, h = d['g'][f, 2] * scale, d['g'][f, 3] * scale
        ax.add_patch(Rectangle((u - w / 2, v - h / 2), w, h, fill=False, ec=COL['gt'], lw=1.6))
    if show_peaks:
        for k in ('1', '2d', '2x'):
            b = d['pk'][k][f]
            if np.isfinite(b).all():
                u, v = to_xy(b[0], b[1])
                ax.add_patch(Rectangle((u - b[2] * scale / 2, v - b[3] * scale / 2), b[2] * scale, b[3] * scale, fill=False, ec=COL[k], lw=1.2,
                                       ls='-' if k == '1' else '--'))


index = []
for c, s, t, L in chosen:
    d = S_[s]
    img0 = cv2.imread(os.path.join(ROOT, d['names'][0]))
    H, W = img0.shape[:2]
    avg = np.mean(img0, axis=(0, 1))
    cx0, cy0, w0, h0 = get_axis_aligned_bbox(d['g'][0])
    scaleaa = round(np.sqrt((w0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0)) * (h0 + cfg.TRACK.CONTEXT_AMOUNT * (w0 + h0))))
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.0), dpi=100)
    last = (d["gcx"][t], d["gcy"][t])
    for ax, f in zip(axes[0], (t - 5, t - 1, t, t + 5)):
        ctr = (d['gcx'][f], d['gcy'][f]) if d['usable'][f] else last
        last = ctr
        side = int(max(48, round(3 * max(d['g'][t, 2], d['g'][t, 3]))))
        img = cv2.imread(os.path.join(ROOT, d['names'][f]))
        patch = to_img(tracker.get_subwindow(img, np.array(ctr), side, side, avg, enh))
        xm, ym = xmin_of(np.array(ctr), side)
        ax.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        draw_boxes(ax, d, f, lambda x, y: (x - xm, y - ym), 1.0)
        ax.set_title('frame {} (t{:+d})'.format(f, f - t), fontsize=9)
        ax.axis('off')
    for ax, f in zip(axes[1][:2], (t - 1, t)):
        img = cv2.imread(os.path.join(ROOT, d['names'][f]))
        crop_, pos, Ssz = search_crop(img, d['p'][f - 1], avg, scaleaa)
        xm, ym = xmin_of(pos, Ssz)
        fs = cfg.TRACK.INSTANCE_SIZE / float(Ssz)
        ax.imshow(cv2.cvtColor(crop_, cv2.COLOR_BGR2RGB), interpolation='nearest')
        draw_boxes(ax, d, f, lambda x, y: ((x - xm) * fs, (y - ym) * fs), fs)
        ax.set_xlim(0, 287); ax.set_ylim(287, 0)
        ax.set_title('search crop of frame {}: S={} ({:.2f}x)'.format(f, Ssz, fs), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    # frame-1 template crop with its GT box
    box0 = [cx0 - (w0 - 1) / 2, cy0 - (h0 - 1) / 2, w0, h0]
    pos0 = np.array([box0[0] + (box0[2] - 1) / 2, box0[1] + (box0[3] - 1) / 2])
    z = to_img(tracker.get_subwindow(img0, pos0, cfg.TRACK.EXEMPLAR_SIZE, scaleaa, avg, enh))
    xm, ym = xmin_of(pos0, scaleaa)
    fz = cfg.TRACK.EXEMPLAR_SIZE / float(scaleaa)
    ax = axes[1][2]
    ax.imshow(cv2.cvtColor(z, cv2.COLOR_BGR2RGB), interpolation='nearest')
    ax.add_patch(Rectangle(((d['gcx'][0] - xm) * fz - d['g'][0, 2] * fz / 2, (d['gcy'][0] - ym) * fz - d['g'][0, 3] * fz / 2), d['g'][0, 2] * fz, d['g'][0, 3] * fz, fill=False, ec=COL['gt'], lw=1.6))
    ax.set_xlim(0, 127); ax.set_ylim(127, 0)
    ax.set_title('frame-1 template crop (127, gamma 0.6)', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    pkv = {k: np.hypot(d['pk'][k][t, 0] - d['gcx'][t], d['pk'][k][t, 1] - d['gcy'][t]) for k in d['pk']}
    axes[1][3].axis('off')
    axes[1][3].text(0, 1, 'green GT, red top-1 (solid), blue 2d, yellow 2x (dashed)\n\nerror to GT at t:\n top-1 {:.0f} px\n 2d {:.0f} px\n 2x {}\n\nstretch length {}\nGT size {:.0f}x{:.0f} px'.format(
        pkv['1'], pkv['2d'], '{:.0f} px'.format(pkv['2x']) if np.isfinite(pkv['2x']) else 'none', L, d['g'][t, 2], d['g'][t, 3]), va='top', fontsize=9, family='monospace')
    fig.suptitle('{}  frame {}  stretch length {}  category: {}'.format(s, t, L, c), fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    name = 'E22_{}_{}_f{:04d}_L{}.png'.format('R' if c == 'runner-up' else 'N', s, t, L)
    fig.savefig(os.path.join(PNG_DIR, name))
    plt.close(fig)
    index.append([name, c, s, t, L])
with open(os.path.join(OUT, 'E22_D_index.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['file', 'category', 'seq', 'start_frame', 'stretch_length'])
    w.writerows(index)
print('wrote {} PNGs in {} and illumination/E22_D_index.csv'.format(len(index), PNG_DIR))
