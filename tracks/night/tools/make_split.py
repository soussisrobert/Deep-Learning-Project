"""Inspect a night tracking dataset laid out as <root>/data_seq/<seq>/*.jpg + <root>/anno/<seq>.txt (x,y,w,h per frame)
and write a sequence-level train/val/test split (about 65/10/25), stratified by mean gray, fixed seed.

usage: python -I tools/make_split.py <root> <name> [seed]
writes tools/splits/<name>_stats.csv and tools/splits/<name>_{train,val,test}.txt (one sequence name per line).
Whole sequences only; the split is decided from brightness alone, never from tracker results.
"""
import os
import sys
import csv
import glob
import random
import numpy as np
import cv2

root, name = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
seqs = sorted(os.listdir(os.path.join(root, 'data_seq')))
rows = []
for s in seqs:
    frames = sorted(glob.glob(os.path.join(root, 'data_seq', s, '*.jpg')))
    ann = np.loadtxt(os.path.join(root, 'anno', s + '.txt'), delimiter=',', ndmin=2)
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    grays, sizes = [], None
    for i in idx:
        im = cv2.imread(frames[i])
        grays.append(float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()))
        sizes = im.shape[:2]
    nan = int(np.isnan(ann).any(axis=1).sum())
    zero = int(((ann[:, 2] <= 0) | (ann[:, 3] <= 0)).sum())
    rows.append(dict(seq=s, frames=len(frames), anno=len(ann), nan_rows=nan, zero_size=zero, h=sizes[0], w=sizes[1],
                     gray=round(float(np.mean(grays)), 1), box_w=round(float(np.nanmedian(ann[:, 2])), 1),
                     box_h=round(float(np.nanmedian(ann[:, 3])), 1)))
os.makedirs('tools/splits', exist_ok=True)
with open('tools/splits/{}_stats.csv'.format(name), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('{} sequences, {} frames; anno/frame mismatches: {}; sequences with bad boxes: {}'.format(
    len(rows), sum(r['frames'] for r in rows), sum(r['frames'] != r['anno'] for r in rows),
    sum(r['nan_rows'] + r['zero_size'] > 0 for r in rows)))
g = np.array([r['gray'] for r in rows])
print('mean gray quartiles', np.percentile(g, [0, 25, 50, 75, 100]).round(1))

# stratify: sort by brightness, cut into blocks of 20 sequences, split each block 13/2/5 at random
rng = random.Random(seed)
order = sorted(rows, key=lambda r: r['gray'])
parts = {'train': [], 'val': [], 'test': []}
for b in range(0, len(order), 20):
    blk = order[b:b + 20]
    rng.shuffle(blk)
    n = len(blk)
    nt, nv = round(0.25 * n), round(0.10 * n)
    parts['test'] += blk[:nt]
    parts['val'] += blk[nt:nt + nv]
    parts['train'] += blk[nt + nv:]
for k, v in parts.items():
    with open('tools/splits/{}_{}.txt'.format(name, k), 'w') as f:
        f.write('\n'.join(sorted(r['seq'] for r in v)) + '\n')
    print('{:5s} {:3d} seqs {:6d} frames mean gray {:.1f}'.format(
        k, len(v), sum(r['frames'] for r in v), np.mean([r['gray'] for r in v])))
