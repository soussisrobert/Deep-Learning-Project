"""Check whether training sequences contain footage that is also in NAT2021 / NAT2021L (the test sets).

The first version used a 64-bit dHash; it was too weak for night frames (smooth, vignetted, dark images collide:
daytime UAV123 frames "matched" NAT at distance 0), so it is replaced.

Stage 1: every NAT2021 and NAT2021L frame (all 194,379) -> 24x24 grayscale thumbnail, z-scored (removes gain/offset, so
         a darker or brighter copy still matches). A query frame's score against a NAT frame is the Pearson correlation.
Stage 2: the 5 best stage-1 candidates are compared again at 160x90 after removing low frequencies (image minus its
         Gaussian blur, sigma 3), so a shared vignette or gradient does not count, only shared structure/noise pattern.
A query frame is a MATCH if low-res correlation >= LO and high-pass correlation >= HI for some candidate.
Queries: 5 frames per sequence (first, 25%, 50%, 75%, last); a sequence is dropped if any of them matches.
UAV123 (daytime, other lab) is run as a negative control: it should come out with no matches.

usage: python -I tools/overlap_check.py <LO> <HI> <label>=<seq_root> [<label>=<seq_root> ...]
writes tools/splits/overlap_<label>.csv (one row per query frame) and tools/splits/overlap_drop_<label>.txt
"""
import os
import sys
import glob
import numpy as np
import cv2
from multiprocessing import Pool

NAT_ROOTS = ['test_dataset/NAT2021/data_seq', 'test_dataset/NAT2021L/data_seq']


def zs(a):
    a = a.astype(np.float32).ravel()
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 1e-6 else a * 0


def thumb(path):
    im = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if im is None:
        return np.zeros(576, np.float32)
    return zs(cv2.resize(im, (24, 24), interpolation=cv2.INTER_AREA))


def hp(path):
    im = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    im = cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    return zs(im - cv2.GaussianBlur(im, (0, 0), 3))


if __name__ == '__main__':
    lo, hi = float(sys.argv[1]), float(sys.argv[2])
    sets = [a.split('=', 1) for a in sys.argv[3:]]
    nat_files = []
    for r in NAT_ROOTS:
        for s in sorted(os.listdir(r)):
            for f in sorted(glob.glob(os.path.join(r, s, '*.jpg'))):
                nat_files.append((r.split('/')[1], s, f))
    print('NAT frames:', len(nat_files), flush=True)
    cache = 'tools/splits/nat_thumb24.npy'
    if os.path.exists(cache) and len(np.load(cache, mmap_mode='r')) == len(nat_files):
        nat = np.load(cache)
    else:
        with Pool(8) as p:
            nat = np.array(p.map(thumb, [f for _, _, f in nat_files], chunksize=500), dtype=np.float16)
        np.save(cache, nat)
    nat = nat.astype(np.float32)
    print('NAT thumbnails ready', flush=True)
    for label, root in sets:
        seqs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        qs = []
        for s in seqs:
            frames = sorted(glob.glob(os.path.join(root, s, '*.jpg')))
            for k, i in enumerate(np.linspace(0, len(frames) - 1, 5).astype(int)):
                qs.append((s, int(i), 'first' if k == 0 else 'sampled', frames[i]))
        Q = np.stack([thumb(q[3]) for q in qs])
        S = Q @ nat.T  # (queries, NAT frames) correlations
        rows, drop = [], set()
        for qi, q in enumerate(qs):
            cand = np.argsort(-S[qi])[:5]
            qh = hp(q[3])
            best = (-1.0, -1.0, cand[0])
            for j in cand:
                h = float(qh @ hp(nat_files[j][2]))
                if h > best[1]:
                    best = (float(S[qi, j]), h, j)
            lowc, hpc, j = best
            match = lowc >= lo and hpc >= hi
            rows.append((q[0], q[1], q[2], round(float(S[qi].max()), 4), round(hpc, 4), int(match), nat_files[j][0], nat_files[j][1], os.path.basename(nat_files[j][2])))
            if match:
                drop.add(q[0])
        with open('tools/splits/overlap_{}.csv'.format(label), 'w') as f:
            f.write('seq,frame_idx,kind,best_lowres_corr,hp_corr_of_best_candidate,match,nat_set,nat_seq,nat_frame\n')
            for r in rows:
                f.write(','.join(map(str, r)) + '\n')
        with open('tools/splits/overlap_drop_{}.txt'.format(label), 'w') as f:
            f.write('\n'.join(sorted(drop)) + ('\n' if drop else ''))
        lc = np.array([r[3] for r in rows])
        hc = np.array([r[4] for r in rows])
        print('{}: {} sequences, {} query frames; lowres corr pct (5/50/95/max) {}; hp corr pct {}; matched frames {}; flagged sequences {}: {}'.format(
            label, len(seqs), len(rows), np.percentile(lc, [5, 50, 95, 100]).round(3), np.percentile(hc, [5, 50, 95, 100]).round(3),
            sum(r[5] for r in rows), len(drop), sorted(drop)), flush=True)
