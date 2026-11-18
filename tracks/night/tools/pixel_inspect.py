"""Print pixel-level statistics for one NAT frame and save enhanced views.

Outputs (in --out): <seq>_<frame>.png = [original | gamma 0.4 | CLAHE] with the GT box drawn.
"""
import os
import argparse

import cv2
import numpy as np

np.set_printoptions(linewidth=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NAT2021')
    parser.add_argument('--dataset_root', default='../test_dataset')
    parser.add_argument('--seq', required=True)
    parser.add_argument('--frame', type=int, default=1, help='1-based frame number')
    parser.add_argument('--out', default='illumination')
    args = parser.parse_args()

    root = os.path.join(args.dataset_root, args.dataset)
    path = os.path.join(root, 'data_seq', args.seq, '{:06d}.jpg'.format(args.frame))
    img = cv2.imread(path)
    gt = [float(v) for v in open(os.path.join(root, 'anno', args.seq + '.txt')).read().split('\n')[args.frame - 1].split(',')]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    print('=== {} frame {} ({}x{}) gt box x,y,w,h = {} ==='.format(args.seq, args.frame, W, H, gt))

    print('\n-- per channel (B, G, R) --')
    for i, c in enumerate('BGR'):
        ch = img[:, :, i]
        print('{}: mean {:.1f} std {:.1f} min {} max {}'.format(c, ch.mean(), ch.std(), ch.min(), ch.max()))

    print('\n-- gray percentiles --')
    ps = [1, 5, 25, 50, 75, 95, 99, 99.9]
    print(dict(zip(ps, np.percentile(gray, ps).round(1))))
    print('fraction of pixels  <10: {:.1%}  <30: {:.1%}  <60: {:.1%}  >200: {:.1%}  ==255: {:.2%}'.format(
        (gray < 10).mean(), (gray < 30).mean(), (gray < 60).mean(), (gray > 200).mean(), (gray == 255).mean()))
    print('distinct gray levels used: {} / 256'.format(len(np.unique(gray))))

    print('\n-- gray histogram (16 bins of 16 levels, % of pixels) --')
    hist = np.histogram(gray, bins=16, range=(0, 256))[0] / gray.size * 100
    for i, h in enumerate(hist):
        print('{:3d}-{:3d} {:6.2f}% {}'.format(i * 16, i * 16 + 15, h, '#' * int(round(h / 2))))

    x, y, w, h = [int(round(v)) for v in gt]
    x, y = max(x, 0), max(y, 0)
    roi = gray[y:y + h, x:x + w]
    mask = np.ones_like(gray, bool)
    mask[y:y + h, x:x + w] = False
    bg = gray[mask]
    print('\n-- target (GT box) vs background --')
    print('ROI : mean {:.1f} std {:.1f} min {} max {}  p5 {:.0f} p95 {:.0f}'.format(
        roi.mean(), roi.std(), roi.min(), roi.max(), np.percentile(roi, 5), np.percentile(roi, 95)))
    print('BG  : mean {:.1f} std {:.1f}'.format(bg.mean(), bg.std()))
    ring = gray[max(y - 40, 0):y + h + 40, max(x - 40, 0):x + w + 40].astype(float)
    ring_mask = np.ones_like(ring, bool)
    ring_mask[40 if y >= 40 else y:(40 if y >= 40 else y) + h, 40 if x >= 40 else x:(40 if x >= 40 else x) + w] = False
    print('local contrast (ROI mean - 40px surround mean): {:.1f}'.format(roi.mean() - ring[ring_mask].mean()))
    print('ROI area {:.2%} of frame, ROI/frame mean-brightness ratio {:.2f}'.format(roi.size / gray.size, roi.mean() / gray.mean()))

    print('\n-- noise estimate (std of Laplacian) --')
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    dark = gray < 20
    print('dark pixels (<20): {:.2f}   bright pixels (>=20): {:.2f}'.format(lap[dark].std(), lap[~dark].std()))
    print('ROI: {:.2f}'.format(lap[y:y + h, x:x + w].std()))

    print('\n-- 9x16 grid of block-mean gray over the whole frame --')
    grid = cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA)
    print(grid)

    cy, cx = y + h // 2, x + w // 2
    print('\n-- raw gray 12x12 patch at the GT box centre ({}, {}) --'.format(cx, cy))
    print(gray[cy - 6:cy + 6, cx - 6:cx + 6])
    print('\n-- raw BGR of 4 pixels at the centre --')
    print(img[cy, cx - 2:cx + 2])
    ys, xs = np.unravel_index(np.argmax(roi), roi.shape)
    print('\n-- raw gray 12x12 patch around the brightest pixel inside the box ({}, {}) value {} --'.format(
        x + xs, y + ys, roi.max()))
    by, bx = y + ys, x + xs
    print(gray[max(by - 6, 0):by + 6, max(bx - 6, 0):bx + 6])

    def enhance(g):
        gam = (255 * (g / 255.0) ** 0.4).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return gam, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gam_gray, clahe_img = enhance(gray)
    gam_img = (255 * (img / 255.0) ** 0.4).astype(np.uint8)
    panels = []
    for p in (img, gam_img, clahe_img):
        p = p.copy()
        cv2.rectangle(p, (x, y), (x + w, y + h), (0, 255, 0), 2)
        panels.append(cv2.resize(p, (W // 2, H // 2)))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, '{}_{:06d}.png'.format(args.seq, args.frame))
    cv2.imwrite(out, np.hstack(panels))
    print('\nsaved', out)


if __name__ == '__main__':
    main()
