"""Phase 0 inventory of the labeled training datasets (read-only; nothing is written except small csv/txt/png under tools/p0/).

Prints: per dataset sequences / frames / disk size / one annotation line / how an absent target is marked; DarkTrack2021
split groups; UAVDT train/test and weather labels; UAV123 sequence structure and the name-prefix groups; first.pth format;
numpy/torch versions; and the crop511 disk estimate (100 sampled boxes per dataset cropped in memory, JPEG bytes counted).
"""
import os
import re
import glob
import json
import random
import subprocess
import collections
import numpy as np
import cv2

os.makedirs('tools/p0', exist_ok=True)
random.seed(0)
np.random.seed(0)
DT = 'training_dataset/DarkTrack2021/extracted'
U123 = 'training_dataset/UAV123/extracted/UAV123'
UDT = 'training_dataset/UAVDT/extracted'


def du_mb(p):
    return int(subprocess.check_output(['du', '-sm', p]).split()[0])


def crop_x511(im, box):  # same crop as training_dataset/vid/par_crop.py crop_like_SiamFC (x crop only), float instead of np.float
    x1, y1, x2, y2 = box
    pos = ((x1 + x2) / 2., (y1 + y2) / 2.)
    w, h = x2 - x1, y2 - y1
    s_z = np.sqrt((h + 0.5 * (w + h)) * (w + 0.5 * (w + h)))
    scale_z = 127 / s_z
    s_x = s_z + 2 * ((511 - 127) / 2) / scale_z
    b = [pos[0] - s_x / 2, pos[1] - s_x / 2, pos[0] + s_x / 2, pos[1] + s_x / 2]
    a_ = 510 / (b[2] - b[0])
    b_ = 510 / (b[3] - b[1])
    m = np.array([[a_, 0, -a_ * b[0]], [0, b_, -b_ * b[1]]], dtype=np.float64)
    pad = tuple(float(v) for v in np.mean(im, axis=(0, 1)))
    return cv2.warpAffine(im, m, (511, 511), borderMode=cv2.BORDER_CONSTANT, borderValue=pad)


def jpeg_bytes(img):
    return len(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])


est = {}  # dataset -> (n_crops_total, mean_bytes)

# ------------------------------------------------------------------ DarkTrack2021
print('=' * 30, 'DarkTrack2021')
seqs = sorted(os.listdir(DT + '/data_seq'))
nfr, nvalid, nnan, ntiny = 0, 0, 0, 0
samples = []
for s in seqs:
    fr = sorted(glob.glob('{}/data_seq/{}/*.jpg'.format(DT, s)))
    a = np.loadtxt('{}/anno/{}.txt'.format(DT, s), delimiter=',', ndmin=2)
    nfr += len(fr)
    isnan = np.isnan(a).any(axis=1)
    tiny = ~isnan & ((a[:, 2] < 4) | (a[:, 3] < 4))
    nnan += int(isnan.sum()); ntiny += int(tiny.sum())
    ok = ~isnan & ~tiny
    nvalid += int(ok.sum())
    for i in np.where(ok)[0]:
        samples.append((fr[i], a[i]))
print('sequences {}, frames {}, disk {} MB (frames+annotations)'.format(len(seqs), nfr, du_mb(DT)))
print('annotation example (anno/building_1.txt line 1):', open(DT + '/anno/building_1.txt').readline().strip(), '  format x,y,w,h per frame, one row per frame')
print('absent target: NaN rows {}, tiny (<4 px) rows {}, usable frames {}'.format(nnan, ntiny, nvalid))
nan_ex = [(s, ) for s in seqs if np.isnan(np.loadtxt('{}/anno/{}.txt'.format(DT, s), delimiter=',', ndmin=2)).any()][:1]
if nan_ex:
    for l in open('{}/anno/{}.txt'.format(DT, nan_ex[0][0])):
        if 'nan' in l.lower():
            print('absent-target example line ({}):'.format(nan_ex[0][0]), l.strip()); break
groups = collections.defaultdict(lambda: collections.defaultdict(list))
split_of = {}
for k in ('train', 'val', 'test'):
    for l in open('tools/splits/DarkTrack2021_{}.txt'.format(k)):
        if l.strip():
            split_of[l.strip()] = k
for s in seqs:
    g = re.sub(r'[_\d]+$', '', s)
    groups[g][split_of[s]].append(s)
print('DarkTrack2021 split (tools/make_split.py, seed 0) is PER SEQUENCE, stratified by mean gray, not per scene or group.')
print('name groups (name minus trailing _digits) and members per split:')
for g, d in sorted(groups.items()):
    print('   {:9s} train {:2d} val {:2d} test {:2d}  {}'.format(g, len(d['train']), len(d['val']), len(d['test']),
          'SPANS SPLITS' if sum(1 for k in d if d[k]) > 1 else ''))
print('(names are class + running number, e.g. car_12; they do not encode the scene, so name groups are class groups)')
random.shuffle(samples)
b = []
for p, a in samples[:100]:
    im = cv2.imread(p)
    b.append(jpeg_bytes(crop_x511(im, (a[0], a[1], a[0] + a[2], a[1] + a[3]))))
est['DarkTrack2021'] = (nvalid, float(np.mean(b)))
print('crop511 x-crop JPEG q95: mean {:.1f} KB over 100 samples -> {} usable frames x = {:.1f} GB'.format(np.mean(b) / 1e3, nvalid, nvalid * np.mean(b) / 1e9))

# ------------------------------------------------------------------ UAV123
print('=' * 30, 'UAV123 (original, undarkened frames)')
cfg_txt = open(U123 + '/configSeqs.m', errors='ignore').read()
blocks = re.split(r'seq(UAV20L|UAV123\w*)\s*=', cfg_txt)
structs = re.findall(r"struct\('name','([^']+)','path','[^']*\\([^\\']+)\\','startFrame',(\d+),'endFrame',(\d+)", cfg_txt)
print('struct entries in configSeqs.m:', len(structs))
folders = sorted(d for d in os.listdir(U123 + '/data_seq/UAV123') if os.path.isdir(U123 + '/data_seq/UAV123/' + d))
nfr_folder = {d: len(glob.glob('{}/data_seq/UAV123/{}/*.jpg'.format(U123, d))) for d in folders}
print('frame folders {}, jpg on disk {}, disk {} MB'.format(len(folders), sum(nfr_folder.values()), du_mb(U123)))
annos = sorted(glob.glob(U123 + '/anno/UAV123/*.txt')) + sorted(glob.glob(U123 + '/anno/UAV20L/*.txt'))
print('annotation files: UAV123 {}, UAV20L {}'.format(len(glob.glob(U123 + '/anno/UAV123/*.txt')), len(glob.glob(U123 + '/anno/UAV20L/*.txt'))))
a0 = annos[0]
print('annotation example ({}):'.format(os.path.basename(a0)), open(a0).readline().strip(), '  format x,y,w,h per frame')
nan_seqs, nan_rows, tot_rows, mism = 0, 0, 0, []
seq_frames = {}
for name, folder, st, en in structs:
    f = '{}/anno/UAV123/{}.txt'.format(U123, name)
    if not os.path.exists(f):
        f = '{}/anno/UAV20L/{}.txt'.format(U123, name)
    if not os.path.exists(f):
        continue
    a = np.loadtxt(f, delimiter=',', ndmin=2)
    n = int(en) - int(st) + 1
    seq_frames[name] = (folder, int(st), int(en))
    if len(a) != n:
        mism.append((name, len(a), n))
    nr = int(np.isnan(a).any(axis=1).sum())
    nan_rows += nr; tot_rows += len(a); nan_seqs += nr > 0
print('annotation rows {}; absent-target rows (NaN) {} in {} sequences; rows != endFrame-startFrame+1: {}'.format(tot_rows, nan_rows, nan_seqs, mism[:6]))
for name in list(seq_frames)[:400]:
    f = '{}/anno/UAV123/{}.txt'.format(U123, name)
    if os.path.exists(f) and any('nan' in l.lower() for l in open(f)):
        print('absent-target example ({}):'.format(name), next(l for l in open(f) if 'nan' in l.lower()).strip()); break
print('UAV123 sequences (UAV123 list + UAV20L list; the 20L ones reuse the same frame folders): {}'.format(len(seq_frames)))
print('how the darkened copy was made: NO darkened copy exists on disk and no darkening was run on any dataset. The only darkening')
print('   code is pysot/utils/darken.py (written this session, validated on UAV123 samples, not applied to files).')
ugroups = collections.defaultdict(lambda: [0, 0])
for d in folders:
    g = re.sub(r'\d+$', '', re.sub(r'_[^_]*$', '', d))  # drop _suffix, then trailing digits
    ugroups[g][0] += 1
    ugroups[g][1] += nfr_folder[d]
print('UAV123 frame folders grouped by name prefix (suffix after _ and trailing digits removed): group, folders, frames')
tot = sum(v[1] for v in ugroups.values())
for g, v in sorted(ugroups.items(), key=lambda kv: -kv[1][1]):
    print('   {:12s} {:3d} folders {:7d} frames {:5.1f}%'.format(g, v[0], v[1], 100. * v[1] / tot))
pool = []
for name, (folder, st, en) in seq_frames.items():
    f = '{}/anno/UAV123/{}.txt'.format(U123, name)
    if not os.path.exists(f):
        continue
    a = np.loadtxt(f, delimiter=',', ndmin=2)
    for i in range(len(a)):
        if not np.isnan(a[i]).any() and a[i][2] >= 4 and a[i][3] >= 4:
            pool.append((folder, st + i, a[i]))
random.shuffle(pool)
b = []
for folder, fi, a in pool[:100]:
    im = cv2.imread('{}/data_seq/UAV123/{}/{:06d}.jpg'.format(U123, folder, fi))
    if im is not None:
        b.append(jpeg_bytes(crop_x511(im, (a[0], a[1], a[0] + a[2], a[1] + a[3]))))
est['UAV123'] = (len(pool), float(np.mean(b)))
print('crop511 estimate: mean {:.1f} KB over {} samples; {} annotated frames x = {:.1f} GB (all UAV123+UAV20L rows counted; the 20L rows repeat frames, so true count is lower)'.format(
    np.mean(b) / 1e3, len(b), len(pool), len(pool) * np.mean(b) / 1e9))

# ------------------------------------------------------------------ UAVDT
print('=' * 30, 'UAVDT-M')
tk = glob.glob(UDT + '/toolkit/*')[0]
at = glob.glob(UDT + '/attributes/*')[0]
bm = glob.glob(UDT + '/benchmark_M/*')[0]
for md in glob.glob(tk + '/*.md') + glob.glob(tk + '/readme*'):
    print('--- toolkit doc', os.path.basename(md))
    print(open(md, errors='ignore').read()[:1500])
gtdir = [d for d in glob.glob(tk + '/**/GT', recursive=True)][0]
print('GT files:', len(os.listdir(gtdir)), 'e.g.', sorted(os.listdir(gtdir))[:4])
useqs = sorted(d for d in os.listdir(bm) if os.path.isdir(bm + '/' + d))
print('sequences {}, frames {}, disk {} MB (frames) + toolkit'.format(len(useqs), sum(len(glob.glob('{}/{}/*.jpg'.format(bm, s))) for s in useqs), du_mb(bm)))
attr = {}
for part in ('train', 'test'):
    for f in sorted(glob.glob('{}/{}/*_attr.txt'.format(at, part)) + glob.glob('{}/*/{}/*_attr.txt'.format(at, part))):
        name = os.path.basename(f).replace('_attr.txt', '').strip()
        attr[name] = (part, [int(x) for x in open(f).read().strip().split(',')])
print('attribute files read:', len(attr), ' (order: daylight, night, fog; low/medium/high alt; front/side/bird view; long-term)')
cnt = collections.defaultdict(lambda: [0, 0])
for name, (part, v) in sorted(attr.items()):
    w = ['daylight', 'night', 'fog'][int(np.argmax(v[:3]))] if sum(v[:3]) == 1 else 'unclear'
    nf = len(glob.glob('{}/{}/*.jpg'.format(bm, name)))
    cnt[(part, w)][0] += 1
    cnt[(part, w)][1] += nf
for k, v in sorted(cnt.items()):
    print('   {:5s} {:9s} {:2d} sequences {:6d} frames'.format(k[0], k[1], v[0], v[1]))
print('train sequences by weather label:')
for name, (part, v) in sorted(attr.items()):
    if part == 'train':
        print('   {} {} frames {}'.format(name, ['daylight', 'night', 'fog'][int(np.argmax(v[:3]))], len(glob.glob('{}/{}/*.jpg'.format(bm, name)))), end=';')
print()
g0 = sorted(os.listdir(gtdir))
for f in [x for x in g0 if x.endswith('_gt.txt')][:1] + [x for x in g0 if x.endswith('_gt_whole.txt')][:1] + [x for x in g0 if 'ignore' in x][:1]:
    print('GT example', f, ':', open(gtdir + '/' + f).readline().strip())
# track statistics on train sequences (gt_whole), tracks >=30 frames with boxes >= 8 px on the short side
tot_obj, kept_obj, kept_tracks = 0, 0, 0
pool = []
for name, (part, v) in sorted(attr.items()):
    if part != 'train':
        continue
    f = '{}/{}_gt_whole.txt'.format(gtdir, name)
    if not os.path.exists(f):
        continue
    a = np.loadtxt(f, delimiter=',', ndmin=2)
    tot_obj += len(a)
    byid = collections.defaultdict(list)
    for r in a:
        byid[int(r[1])].append(r)
    for i, rows in byid.items():
        rows = [r for r in rows if min(r[4], r[5]) >= 8]
        if len(rows) >= 30:
            kept_tracks += 1
            kept_obj += len(rows)
            pool += [(name, r) for r in rows]
print('train sequences, gt_whole: {} object-frames; tracks with >=30 frames of short side >=8 px: {} tracks, {} object-frames (every one would be one crop)'.format(tot_obj, kept_tracks, kept_obj))
random.shuffle(pool)
b = []
for name, r in pool[:100]:
    im = cv2.imread('{}/{}/img{:06d}.jpg'.format(bm, name, int(r[0])))
    if im is not None:
        b.append(jpeg_bytes(crop_x511(im, (r[2], r[3], r[2] + r[4], r[3] + r[5]))))
est['UAVDT'] = (kept_obj, float(np.mean(b)) if b else 0.0)
print('crop511 estimate: mean {:.1f} KB over {} samples; {} object-frames x = {:.1f} GB'.format(np.mean(b) / 1e3 if b else 0, len(b), kept_obj, kept_obj * (np.mean(b) if b else 0) / 1e9))

# ------------------------------------------------------------------ code facts
print('=' * 30, 'environment and checkpoint')
import torch
print('numpy', np.__version__, 'has np.float:', hasattr(np, 'float'), 'torch', torch.__version__, 'cv2', cv2.__version__)
ck = torch.load('tools/snapshot/first.pth', map_location='cpu')
print('first.pth type', type(ck).__name__, 'top-level keys', list(ck.keys())[:6] if isinstance(ck, dict) else '')
sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
print('first.pth tensors', len(sd), 'example keys', list(sd.keys())[:3], 'has optimizer state', isinstance(ck, dict) and 'optimizer' in ck, 'epoch', ck.get('epoch') if isinstance(ck, dict) else None)
print('=' * 30, 'crop511 total estimate')
for k, (n, by) in est.items():
    print('   {:14s} {:8d} crops x {:5.1f} KB = {:5.1f} GB'.format(k, n, by / 1e3, n * by / 1e9))
print('   total {:.1f} GB (all datasets, every usable frame). Home now {} MB.'.format(sum(n * by for n, by in est.values()) / 1e9, du_mb(os.path.expanduser('~'))))
