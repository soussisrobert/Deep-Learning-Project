"""Phase 1 plan: sequence splits and the UAVDT track selection, plus the measured crop511 size per dataset.

UAVDark135: whole linked groups (union of the same-footage pairs found by tools/leak_one.py, and sequences that share a
   base name before '_<n>', e.g. person3_1, person3_2), split ~75/10/15 train/val/test by frame count, stratified by
   mean gray (3 brightness bands of groups; within each band the seed (0..999) with the closest fractions is taken).
UAV123: folders whose name prefix (suffix after _ and trailing digits removed) is group, wakeboard or boat are the
   held-out daytime test set; every other folder is train. The long-term (UAV20L) sequences are only used for frames not
   already covered by the 123 list.
UAVDT: official train sequences only; tracks of at least 30 frames after dropping frames with a box short side < 8 px,
   out-of-view != 1 or a box mostly inside an ignore region; night sequences and day sequences get the same object-frame
   quota (30k each, cap ~60k in all), tracks added round robin over sequences so every sequence is represented; fog is not used.
Writes tools/splits/UAVDark135_{train,val,test}.txt (+ _groups.csv), UAV123_{train,test}.txt, UAVDT_selection.json.
Measures the JPEG size (q95, 'x' crop 511 only, which is all the loader reads) of 100 sampled crops per dataset.
"""
import os
import re
import glob
import json
import random
import collections
import numpy as np
import cv2

random.seed(0)
np.random.seed(0)
UD = 'training_dataset/UAVDark135/extracted/UAVDark135_TSP_out'
U123 = 'training_dataset/UAV123/extracted/UAV123'
UDT = 'training_dataset/UAVDT/extracted'
os.makedirs('tools/splits', exist_ok=True)


def crop_x511(im, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    s_z = np.sqrt((h + 0.5 * (w + h)) * (w + 0.5 * (w + h)))
    s_x = s_z + 2 * ((511 - 127) / 2) / (127 / s_z)
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    b = [cx - s_x / 2, cy - s_x / 2, cx + s_x / 2, cy + s_x / 2]
    a_, b_ = 510 / (b[2] - b[0]), 510 / (b[3] - b[1])
    m = np.array([[a_, 0, -a_ * b[0]], [0, b_, -b_ * b[1]]], dtype=np.float64)
    return cv2.warpAffine(im, m, (511, 511), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=tuple(float(v) for v in np.mean(im, axis=(0, 1))))


def jb(img):
    return len(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])


# ======================================================================== UAVDark135
print('=' * 20, 'UAVDark135')
seqs = sorted(os.listdir(UD + '/Sequences'))
info = {}
for s in seqs:
    fr = sorted(glob.glob('{}/Sequences/{}/*.jpg'.format(UD, s)))
    a = np.loadtxt('{}/anno_revise/{}.txt'.format(UD, s), delimiter=',', ndmin=2)
    idx = np.linspace(0, len(fr) - 1, 8).astype(int)
    gray = float(np.mean([cv2.cvtColor(cv2.imread(fr[i]), cv2.COLOR_BGR2GRAY).mean() for i in idx]))
    n = min(len(fr), len(a))
    ok = ~np.isnan(a[:n]).any(axis=1) & (a[:n, 2] >= 4) & (a[:n, 3] >= 4)
    info[s] = dict(frames=len(fr), rows=len(a), usable=int(ok.sum()), gray=gray)
mism = {s: (v['frames'], v['rows']) for s, v in info.items() if v['frames'] != v['rows']}
print('sequences {}, frames {}, annotation rows {}, usable (box present, >=4 px) {}'.format(
    len(seqs), sum(v['frames'] for v in info.values()), sum(v['rows'] for v in info.values()), sum(v['usable'] for v in info.values())))
print('sequences where jpg count != annotation rows: {} (alignment assumed row i = i-th sorted jpg; extra frames/rows ignored): {}'.format(len(mism), dict(list(mism.items())[:12])))
parent = {s: s for s in seqs}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    parent[find(a)] = find(b)


for l in list(open('tools/p0/leak_UAVDark135_within.csv'))[1:]:
    a, b = l.split(',')[:2]
    union(a, b)
for s in seqs:
    m = re.match(r'^(.*)_\d+$', s)
    if m:
        base = m.group(1)
        for t in seqs:
            if t != s and (t == base or re.match(r'^' + re.escape(base) + r'_\d+$', t)):
                union(s, t)
groups = collections.defaultdict(list)
for s in seqs:
    groups[find(s)].append(s)
G = [dict(members=sorted(m), frames=sum(info[x]['frames'] for x in m),
          gray=float(np.average([info[x]['gray'] for x in m], weights=[info[x]['frames'] for x in m]))) for m in groups.values()]
print('groups {} (sizes: {} singletons, {} with 2+ members; largest {} members / {} frames)'.format(
    len(G), sum(len(g['members']) == 1 for g in G), sum(len(g['members']) > 1 for g in G),
    max(len(g['members']) for g in G), max(g['frames'] for g in G)))
tot = sum(g['frames'] for g in G)
G.sort(key=lambda g: g['gray'])
bands, acc, cur = [], 0, []
for g in G:
    cur.append(g)
    acc += g['frames']
    if acc >= tot / 3 and len(bands) < 2:
        bands.append(cur); cur, acc = [], 0
bands.append(cur)
assign = {}
for bi, band in enumerate(bands):
    bt = sum(g['frames'] for g in band)
    best = None
    for seed in range(1000):
        r = random.Random(seed)
        order = band[:]
        r.shuffle(order)
        te, va = 0, 0
        lab = {}
        for g in order:
            if te < 0.15 * bt:
                lab[id(g)] = 'test'; te += g['frames']
            elif va < 0.10 * bt:
                lab[id(g)] = 'val'; va += g['frames']
            else:
                lab[id(g)] = 'train'
        err = abs(te / bt - 0.15) + abs(va / bt - 0.10)
        if best is None or err < best[0]:
            best = (err, lab)
    for g in band:
        assign[id(g)] = best[1][id(g)]
parts = {'train': [], 'val': [], 'test': []}
for g in G:
    parts[assign[id(g)]].append(g)
for k, v in parts.items():
    with open('tools/splits/UAVDark135_{}.txt'.format(k), 'w') as f:
        f.write('\n'.join(sorted(m for g in v for m in g['members'])) + '\n')
    fr = sum(g['frames'] for g in v)
    print('{:5s} {:3d} groups {:3d} sequences {:6d} frames ({:4.1f}%)  mean gray {:.1f}'.format(
        k, len(v), sum(len(g['members']) for g in v), fr, 100. * fr / tot,
        np.average([g['gray'] for g in v], weights=[g['frames'] for g in v])))
with open('tools/splits/UAVDark135_groups.csv', 'w') as f:
    f.write('group_members,frames,gray,split\n')
    for g in G:
        f.write('"{}",{},{:.1f},{}\n'.format(' '.join(g['members']), g['frames'], g['gray'], assign[id(g)]))
print('multi-member groups:', [(' '.join(g['members']), assign[id(g)]) for g in G if len(g['members']) > 1])
gr = [g['gray'] for g in G]
print('mean gray over sequences (min/25/50/75/max): {}'.format(np.percentile([info[s]['gray'] for s in seqs], [0, 25, 50, 75, 100]).round(1)))
ud_pool = []
for s in seqs:
    if s in set(m for g in parts['train'] + parts['val'] for m in g['members']):
        a = np.loadtxt('{}/anno_revise/{}.txt'.format(UD, s), delimiter=',', ndmin=2)
        fr = sorted(glob.glob('{}/Sequences/{}/*.jpg'.format(UD, s)))
        for i in range(min(len(fr), len(a))):
            if not np.isnan(a[i]).any() and a[i][2] >= 4 and a[i][3] >= 4:
                ud_pool.append((fr[i], a[i]))
random.shuffle(ud_pool)
bs = [jb(crop_x511(cv2.imread(p), (a[0], a[1], a[0] + a[2], a[1] + a[3]))) for p, a in ud_pool[:100]]
print('MEASURED crop511 (x only, q95): {:.1f} KB mean over 100 -> {} usable train+val frames = {:.1f} GB'.format(np.mean(bs) / 1e3, len(ud_pool), len(ud_pool) * np.mean(bs) / 1e9))
est = {'UAVDark135 train+val': (len(ud_pool), float(np.mean(bs)))}

# ======================================================================== UAV123
print('=' * 20, 'UAV123')
cfg = open(U123 + '/configSeqs.m', errors='ignore').read()
structs = re.findall(r"struct\('name','([^']+)','path','[^']*\\([^\\']+)\\','startFrame',(\d+),'endFrame',(\d+)", cfg)
n123 = len(glob.glob(U123 + '/anno/UAV123/*.txt'))
folders = sorted(d for d in os.listdir(U123 + '/data_seq/UAV123') if os.path.isdir(U123 + '/data_seq/UAV123/' + d))
prefix = lambda d: re.sub(r'\d+$', '', re.sub(r'_[^_]*$', '', d))
test_f = [d for d in folders if prefix(d) in ('group', 'wakeboard', 'boat')]
train_f = [d for d in folders if d not in test_f]
nfr = {d: len(glob.glob('{}/data_seq/UAV123/{}/*.jpg'.format(U123, d))) for d in folders}
for k, v in (('train', train_f), ('test', test_f)):
    with open('tools/splits/UAV123_{}.txt'.format(k), 'w') as f:
        f.write('\n'.join(v) + '\n')
    print('{:5s} {:2d} folders {:6d} frames ({:.1f}%)'.format(k, len(v), sum(nfr[d] for d in v), 100. * sum(nfr[d] for d in v) / sum(nfr.values())))
covered, tracks = set(), []  # tracks: (folder, name, {frame: box})
names123 = {os.path.basename(f)[:-4] for f in glob.glob(U123 + '/anno/UAV123/*.txt')}
for pass_ in ('UAV123', 'UAV20L'):
    for name, folder, st, en in structs:
        d = 'UAV123' if os.path.exists('{}/anno/UAV123/{}.txt'.format(U123, name)) else 'UAV20L'
        if d != pass_ or folder not in train_f:
            continue
        a = np.loadtxt('{}/anno/{}/{}.txt'.format(U123, d, name), delimiter=',', ndmin=2)
        boxes = {}
        for i in range(len(a)):
            fr = int(st) + i
            if (folder, fr) in covered or np.isnan(a[i]).any() or a[i][2] < 4 or a[i][3] < 4:
                continue
            boxes[fr] = [float(a[i][0]), float(a[i][1]), float(a[i][0] + a[i][2]), float(a[i][1] + a[i][3])]
        if pass_ == 'UAV20L' and len(boxes) < 30:
            continue
        for fr in boxes:
            covered.add((folder, fr))
        if boxes:
            tracks.append((folder, name, boxes))
n_u123 = sum(len(t[2]) for t in tracks)
print('train folders: {} tracks, {} usable annotated frames (frames in both lists counted once)'.format(len(tracks), n_u123))
json.dump({'{}'.format(t[0]) + '/' + t[1]: {str(k): v for k, v in t[2].items()} for t in tracks}, open('tools/splits/UAV123_train_tracks.json', 'w'))
pool = [(f, fr, b) for f, n, bx in tracks for fr, b in bx.items()]
random.shuffle(pool)
bs = []
for f, fr, b in pool[:100]:
    im = cv2.imread('{}/data_seq/UAV123/{}/{:06d}.jpg'.format(U123, f, fr))
    bs.append(jb(crop_x511(im, b)))
print('MEASURED crop511: {:.1f} KB mean over 100 -> {} frames = {:.1f} GB'.format(np.mean(bs) / 1e3, n_u123, n_u123 * np.mean(bs) / 1e9))
est['UAV123 train'] = (n_u123, float(np.mean(bs)))

# ======================================================================== UAVDT
print('=' * 20, 'UAVDT-M selection (train sequences only)')
tk = glob.glob(UDT + '/toolkit/*')[0] + '/GT'
bm = glob.glob(UDT + '/benchmark_M/*')[0]
at = glob.glob(UDT + '/attributes/*')[0]
tr = {}
for f in glob.glob(at + '/train/*_attr.txt'):
    v = [int(x) for x in open(f).read().strip().split(',')]
    tr[os.path.basename(f).replace('_attr.txt', '').strip()] = ['daylight', 'night', 'fog'][int(np.argmax(v[:3]))]
cand = {'night': {}, 'daylight': {}}
for s, w in sorted(tr.items()):
    if w not in cand:
        continue
    a = np.loadtxt('{}/{}_gt_whole.txt'.format(tk, s), delimiter=',', ndmin=2)
    ig = np.loadtxt('{}/{}_gt_ignore.txt'.format(tk, s), delimiter=',', ndmin=2) if os.path.getsize('{}/{}_gt_ignore.txt'.format(tk, s)) > 0 else np.zeros((0, 9))
    igf = collections.defaultdict(list)
    for r in ig:
        igf[int(r[0])].append(r[2:6])
    byid = collections.defaultdict(list)
    for r in a:
        if min(r[4], r[5]) < 8 or int(r[6]) != 1:   # too small or (partly) out of view
            continue
        bad = False
        for g in igf.get(int(r[0]), []):
            ix = max(0, min(r[2] + r[4], g[0] + g[2]) - max(r[2], g[0]))
            iy = max(0, min(r[3] + r[5], g[1] + g[3]) - max(r[3], g[1]))
            if ix * iy > 0.5 * r[4] * r[5]:
                bad = True; break
        if not bad:
            byid[int(r[1])].append(r)
    cand[w][s] = {i: rows for i, rows in byid.items() if len(rows) >= 30}
quota = 30000
sel, counts = {}, {'night': 0, 'daylight': 0}
for w in ('night', 'daylight'):
    allw = sum(len(r) for s in cand[w].values() for r in s.values())
    q = min(quota, allw)
    print('{}: {} train sequences, {} candidate tracks, {} candidate object-frames, quota {}'.format(
        w, len(cand[w]), sum(len(s) for s in cand[w].values()), allw, q))
    order = {s: random.sample(sorted(t), len(t)) for s, t in cand[w].items()}
    ptr = {s: 0 for s in order}
    progress = True
    while counts[w] < q and progress:
        progress = False
        for s in sorted(order):
            if ptr[s] < len(order[s]) and counts[w] < q:
                i = order[s][ptr[s]]
                ptr[s] += 1
                sel.setdefault(s, {})[i] = {'weather': w, 'frames': {str(int(r[0])): [float(r[2]), float(r[3]), float(r[2] + r[4]), float(r[3] + r[5])] for r in cand[w][s][i]}, 'class': int(cand[w][s][i][0][8])}
                counts[w] += len(cand[w][s][i])
                progress = True
json.dump(sel, open('tools/splits/UAVDT_selection.json', 'w'))
tot_sel = sum(counts.values())
print('selected: night {} + daylight {} = {} object-frames in {} tracks over {} sequences (fog not used)'.format(
    counts['night'], counts['daylight'], tot_sel, sum(len(v) for v in sel.values()), len(sel)))
print('classes of selected tracks (1 car, 2 truck, 3 bus):', dict(collections.Counter(t['class'] for s in sel.values() for t in s.values())))
pool = [(s, int(f), b) for s, ts in sel.items() for t in ts.values() for f, b in t['frames'].items()]
random.shuffle(pool)
bs = []
for s, f, b in pool[:100]:
    im = cv2.imread('{}/{}/img{:06d}.jpg'.format(bm, s, f))
    bs.append(jb(crop_x511(im, b)))
print('MEASURED crop511: {:.1f} KB mean over 100 -> {} object-frames = {:.1f} GB'.format(np.mean(bs) / 1e3, tot_sel, tot_sel * np.mean(bs) / 1e9))
est['UAVDT selected'] = (tot_sel, float(np.mean(bs)))
print('=' * 20, 'TOTAL')
tt = 0
for k, (n, by) in est.items():
    print('   {:22s} {:7d} crops x {:5.1f} KB = {:5.2f} GB'.format(k, n, by / 1e3, n * by / 1e9))
    tt += n * by
print('   total {:.1f} GB'.format(tt / 1e9))
