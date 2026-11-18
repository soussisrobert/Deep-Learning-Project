"""Sample crops and augmentation statistics for the night fine-tuning data (CPU only).

Saves 20 panels (template + search, without and with the night augmentation) to tools/night_ft/samples/ and prints
  * mean gray quartiles and block noise of the augmented search crops per dataset next to NAT2021's (14.8 / 21.6 / 34.0, noise curve),
  * the share of distractor requests that produced a valid distractor pair, label counts of positive / negative pairs,
  * the default_collate shapes and dtypes of a small batch.
usage: python -I tools/night_samples.py <config yaml> [<more yaml>]
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import cv2
import torch
from torch.utils.data import default_collate
from pysot.core.config import cfg
from pysot.datasets.night_dataset import NightTrkDataset
from pysot.datasets.night_aug import enhance_gamma
from pysot.utils.darken import NAT_MU, NAT_SIGMA

for f in sys.argv[1:]:
    cfg.merge_from_file(f)
cfg.DATASET.VIDEOS_PER_EPOCH = 6000
os.makedirs('tools/night_ft/samples', exist_ok=True)
np.random.seed(0)
ds = NightTrkDataset('train', n_epochs=1, start_epoch=0, seed=0)
by_kind = {}
for i, (j, v) in enumerate(ds.picks):
    by_kind.setdefault(ds.kinds[j], []).append(i)
print('pairs by dataset:', {k: len(v) for k, v in by_kind.items()})

K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6.0


def gray(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def block_sigma(img, acc):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = cv2.filter2D(g, -1, K)[1:-1, 1:-1]
    g = g[1:-1, 1:-1]
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    mu = g[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3)).ravel()
    sd = r[:h, :w].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3)).ravel()
    for lo in range(0, 80, 8):
        m = (mu >= lo) & (mu < lo + 8)
        if m.sum() >= 5:
            acc.setdefault(lo + 4, []).append(float(np.median(sd[m])))


enh_cfg = ds.enh_p
print('=== darkening only (enhancement switched off), search crops, 400 pairs per dataset')
ds.enh_p = 0.0
for kind, idxs in by_kind.items():
    grays_raw, grays_all, grays_dk, acc, n_dark = [], [], [], {}, 0
    for i in idxs[:400]:
        out, dbg, raw = ds.make(i)
        s = out['search'].transpose(1, 2, 0).astype(np.uint8)
        grays_raw.append(gray(raw[1]))
        grays_all.append(gray(s))
        if 'darken_s' in dbg:
            grays_dk.append(gray(s))
            block_sigma(s, acc)
        n_dark += 'darken_s' in dbg
    n = len(idxs[:400])
    q = lambda a: np.percentile(a, [0, 25, 50, 75, 100]).round(1) if len(a) else '-'
    print('{:8s} mean gray (min/25/50/75/max): original {} | after, all crops {} | darkened crops only ({:.0f}% of crops) {}'.format(
        kind, q(grays_raw), q(grays_all), 100. * n_dark / n, q(grays_dk)))
    if acc:
        print('         block sigma by brightness bin, darkened crops (augmented / NAT): ' + ' '.join(
            '{}:{:.2f}/{:.2f}'.format(m, np.median(acc[m]), NAT_SIGMA[list(NAT_MU).index(m)]) for m in sorted(acc) if m in list(NAT_MU)))
print('NAT2021 mean gray quartiles for comparison: 14.8 / 21.6 / 34.0 (min 3.6, max 68.7)')
print('=== enhancement alone (gamma uniform in [0.4, 1.0], p forced to 1) on real night crops (UAVDARK)')
ds.enh_p = 1.0
gb, ga = [], []
for i in by_kind['UAVDARK'][:400]:
    out, dbg, raw = ds.make(i)
    gb.append(gray(raw[1]))
    ga.append(gray(out['search'].transpose(1, 2, 0).astype(np.uint8)))
print('UAVDARK search crop mean gray (min/25/50/75/max): before {} -> after enhancement {}'.format(
    np.percentile(gb, [0, 25, 50, 75, 100]).round(1), np.percentile(ga, [0, 25, 50, 75, 100]).round(1)))
ds.enh_p = enh_cfg

print('=== labels and pair types')
pos = ds.make(by_kind['UAVDARK'][0], force_pair='pos')[0]
neg = ds.make(by_kind['UAVDARK'][1], force_pair='neg')[0]
print('positive pair: label==1 cells {}, label==0 cells {}, weightxff sum {:.2f}'.format(int((pos['label_cls1'] == 1).sum()), int((pos['label_cls1'] == 0).sum()), pos['weightxff'].sum()))
print('negative pair: label==1 cells {}, label==0 cells {}, weightxff sum {:.2f}, labelcls2 max {:.2f}'.format(int((neg['label_cls1'] == 1).sum()), int((neg['label_cls1'] == 0).sum()), neg['weightxff'].sum(), neg['labelcls2'].max()))
req, got = 0, 0
for i in by_kind['UAVDT'][:300]:
    out, dbg, raw = ds.make(i, force_pair='hard')
    req += 1
    got += dbg['pair'] == 'hard'
print('distractor pairs requested on UAVDT items: {}, valid (template object outside the search window after augmentation): {} ({:.0f}%), rest fell back to positive pairs'.format(req, got, 100. * got / req))
batch = default_collate([ds[i] for i in range(8)])
print('collate:', {k: (tuple(v.shape), str(v.dtype)) for k, v in batch.items()})

print('=== writing 20 sample panels')
plan = [('UAVDARK', None)] * 4 + [('UAV123', None)] * 6 + [('UAVDT', None)] * 6 + [('UAVDT', 'hard')] * 4
used = {}
tiles = []
for n, (kind, force) in enumerate(plan):
    k = used.get(kind, 0) + (50 if force else 0)
    used[kind] = used.get(kind, 0) + 1
    for i in by_kind[kind][k:k + 60]:
        out, dbg, raw = ds.make(i, force_pair=force)
        if force is None or dbg['pair'] == 'hard':
            break
    t_aug = out['template'].transpose(1, 2, 0).astype(np.uint8)
    s_aug = out['search'].transpose(1, 2, 0).astype(np.uint8)
    t_raw, s_raw = raw
    up = lambda a: cv2.resize(a, (287, 287), interpolation=cv2.INTER_NEAREST)
    row_raw = np.hstack([up(t_raw), s_raw])
    row_aug = np.hstack([up(t_aug), s_aug])
    panel = np.vstack([row_raw, row_aug])
    cap = '{} {} | ' .format(kind, dbg['pair']) + ' '.join('{}:{}'.format(k2, v2) for k2, v2 in dbg.items() if k2 not in ('dataset', 'pair'))
    cv2.putText(panel, cap[:90], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(panel, 'without night augmentation (template x2 | search)', (4, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel, 'with night augmentation', (4, 304), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite('tools/night_ft/samples/sample_{:02d}_{}_{}.png'.format(n + 1, kind, dbg['pair']), panel)
    tiles.append(cv2.resize(panel, (287, 287)))
rows = [np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(4)]
cv2.imwrite('tools/night_ft/samples/contact_sheet.png', np.vstack(rows))
print('saved 20 panels + contact_sheet.png in tools/night_ft/samples/')
