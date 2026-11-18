"""Check the synthetic darkening against NAT2021: brightness distribution and noise-vs-brightness after JPEG.

Darkens 3 frames of each of the 91 UAV123 frame folders (seeded), for noise_k in the given list, and prints
 (1) mean-gray quartiles/range of the darkened frames next to NAT2021's (14.8/21.6/34.0, 3.6-68.7),
 (2) the block-noise curve (same estimator as tools/measure_nat_noise.py, q=0.5) per brightness bin, darkened vs NAT.
Saves a few before/after crops to tools/splits/darken_examples/ for looking at.

usage: python -I tools/validate_darken.py <noise_k> [<noise_k> ...]
"""
import os
import sys
import glob
import numpy as np
import cv2
sys.path.insert(0, os.getcwd())
from pysot.utils.darken import DarkenParams, darken, NAT_MU, NAT_SIGMA

ROOT = 'training_dataset/UAV123/extracted/UAV123/data_seq/UAV123'
K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6.0


def block_curve(img_u8, acc):
    g = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = cv2.filter2D(g, -1, K)[1:-1, 1:-1]
    g = g[1:-1, 1:-1]
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    mu = g[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3)).ravel()
    sd = r[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3)).ravel()
    for lo in range(0, 160, 8):
        m = (mu >= lo) & (mu < lo + 8)
        if m.sum() >= 20:
            acc.setdefault(lo + 4, []).append(float(np.median(sd[m])))


seqs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
os.makedirs('tools/splits/darken_examples', exist_ok=True)
for k in [float(a) for a in sys.argv[1:]]:
    rng = np.random.default_rng(0)
    grays, acc = [], {}
    for si, s in enumerate(seqs):
        fr = sorted(glob.glob(os.path.join(ROOT, s, '*.jpg')))
        pick = [cv2.imread(fr[i]) for i in np.linspace(0, len(fr) - 1, 3).astype(int)]
        p = DarkenParams(pick, rng, noise_k=k)
        for j, im in enumerate(pick):
            d = darken(im, p, rng)
            grays.append(float(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).mean()))
            block_curve(d, acc)
            if k == float(sys.argv[1]) and si % 15 == 0 and j == 1:
                h, w = im.shape[:2]
                cv2.imwrite('tools/splits/darken_examples/{}_k{}.jpg'.format(s, k), np.hstack([im[:h // 2, :w // 2], d[:h // 2, :w // 2]]))
    print('noise_k {}: darkened mean gray quartiles (min/25/50/75/max) {}   [NAT: 3.6 / 14.8 / 21.6 / 34.0 / 68.7]'.format(
        k, np.percentile(grays, [0, 25, 50, 75, 100]).round(1)), flush=True)
    print('   bin: darkened vs NAT block sigma (median block):', flush=True)
    print('   ' + ' '.join('{}:{:.2f}/{:.2f}'.format(m, np.median(acc[m]), NAT_SIGMA[list(NAT_MU).index(m)]) for m in sorted(acc) if m in list(NAT_MU) and m < 100), flush=True)
