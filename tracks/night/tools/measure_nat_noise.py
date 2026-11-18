"""Measure how noise grows with brightness in NAT2021 (test set, only used here for calibration statistics).

Per sampled frame: grayscale, 8x8 blocks, block mean mu (8-bit code) and noise sigma from a high-pass residual
(kernel [1,-2,1;-2,4,-2;1,-2,1]/6, std of residual over the block). In each frame only the flattest 30% of blocks per
brightness bin are kept (low residual std = little texture), so texture is not counted as noise.
Outputs tools/splits/nat_noise_bins_q<Q>.csv (mu bin -> median sigma code), tools/splits/nat_noise_seq_q<Q>.csv (per sequence)
and prints the fit  var_lin = a*x + b  (x = linear intensity 0..1) and the per-sequence spread of noise around it.

usage: python -I tools/measure_nat_noise.py [frames_per_seq]
"""
import os
import sys
import glob
import numpy as np
import cv2
from multiprocessing import Pool

ROOT = 'test_dataset/NAT2021/data_seq'
K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6.0


def lin(c):  # 8-bit sRGB code -> linear 0..1
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


Q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3  # block-noise quantile kept per brightness bin


def frame_stats(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    r = cv2.filter2D(g, -1, K)[1:-1, 1:-1]
    g = g[1:-1, 1:-1]
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    mu = g[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3)).ravel()
    sd = r[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3)).ravel()
    out = []
    for lo in range(0, 160, 8):
        m = (mu >= lo) & (mu < lo + 8)
        if m.sum() >= 20:
            s = np.sort(sd[m])[:max(5, int(Q * m.sum()))]
            out.append((lo + 4, float(np.median(s)), int(m.sum())))
    return out


def seq_job(args):
    s, n = args
    fr = sorted(glob.glob(os.path.join(ROOT, s, '*.jpg')))
    idx = np.linspace(0, len(fr) - 1, n).astype(int)
    acc = []
    for i in idx:
        acc += frame_stats(fr[i])
    return s, acc


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seqs = sorted(os.listdir(ROOT))
    with Pool(4) as p:
        res = p.map(seq_job, [(s, n) for s in seqs])
    bins = {}
    for s, acc in res:
        for mu, sg, cnt in acc:
            bins.setdefault(mu, []).append(sg)
    os.makedirs('tools/splits', exist_ok=True)
    mus = sorted(bins)
    med = [float(np.median(bins[m])) for m in mus]
    with open('tools/splits/nat_noise_bins_q{}.csv'.format(Q) + '', 'w') as f:
        f.write('mu_code,sigma_code_median,n_obs\n')
        for m, v in zip(mus, med):
            f.write('{},{:.3f},{}\n'.format(m, v, len(bins[m])))
    # fit in linear domain: sigma_lin = sigma_code * d(lin)/d(code); var_lin = a*x + b
    x = lin(mus)
    dlin = (lin(np.array(mus) + 0.5) - lin(np.array(mus) - 0.5))
    var = (np.array(med) * dlin) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, var, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    print('bins:', list(zip(mus, np.round(med, 2))))
    print('fit var_lin = a*x + b with a={:.3e}, b={:.3e} (b clipped at 0: {:.3e})'.format(a, b, max(b, 0)))
    pred = np.sqrt(np.maximum(a * x + b, 1e-12)) / dlin
    print('fit residual in code units: max |pred-meas| = {:.2f}'.format(np.abs(pred - np.array(med)).max()))
    # per-sequence noise multiplier relative to the global curve
    ratios = []
    with open('tools/splits/nat_noise_seq_q{}.csv'.format(Q) + '', 'w') as f:
        f.write('seq,mean_ratio\n')
        for s, acc in res:
            r = []
            for mu, sg, cnt in acc:
                xx = lin(mu)
                dd = lin(mu + 0.5) - lin(mu - 0.5)
                pr = np.sqrt(max(a * xx + b, 1e-12)) / dd
                r.append(sg / pr)
            if r:
                ratios.append(float(np.exp(np.mean(np.log(np.maximum(r, 1e-3))))))
                f.write('{},{:.3f}\n'.format(s, ratios[-1]))
    ratios = np.array(ratios)
    print('per-sequence noise multiplier vs global curve: percentiles 5/25/50/75/95 = {}; log-std {:.3f}'.format(
        np.percentile(ratios, [5, 25, 50, 75, 95]).round(2), np.log(ratios).std()))
