"""Mean gray level (0-255, cv2 BGR2GRAY mean, as in illumination.py) and mean HSV V per UAVDT-M sequence, read straight from
the zip (every Nth frame), next to the daylight / night / fog label. Compare with NAT2021 (illumination/NAT2021_illumination.csv).
Nothing is extracted.
"""
import os
import csv
import zipfile
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--zip', default='training_dataset/UAVDT/benchmark_M/UAV-benchmark-M.zip')
parser.add_argument('--attr_zip', default='training_dataset/UAVDT/attributes/M_attr.zip')
parser.add_argument('--nat_csv', default='tools/illumination/NAT2021_illumination.csv')
parser.add_argument('--stride', type=int, default=20)
parser.add_argument('--out_csv', default='tools/illumination/UAVDT_M_illumination.csv')
args = parser.parse_args()

# labels: daylight, night, fog, low-alt, medium-alt, high-alt, front-view, side-view, bird-view, long-term
az = zipfile.ZipFile(args.attr_zip)
label = {}
for n in az.namelist():
    if n.endswith('_attr.txt'):
        seq = os.path.basename(n).replace(' ', '')[:5]
        flags = [int(x) for x in az.read(n).decode().strip().split(',')]
        split = n.split('/')[1]
        label[seq] = ('daylight' if flags[0] else 'night' if flags[1] else 'fog' if flags[2] else '?', split)

zf = zipfile.ZipFile(args.zip)
frames = {}
for n in zf.namelist():
    if n.endswith('.jpg'):
        frames.setdefault(n.split('/')[1], []).append(n)

rows = []
for seq in sorted(frames):
    names = sorted(frames[seq])[::args.stride]
    g, v = [], []
    for n in names:
        img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
        g.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        v.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    h, w = img.shape[:2]
    lab, split = label.get(seq, ('?', '?'))
    rows.append([seq, split, lab, len(frames[seq]), '{}x{}'.format(w, h), np.mean(g), np.mean(v)])
    print('{} {:5s} {:8s} frames {:5d} {}x{} gray {:.1f} v {:.1f}'.format(seq, split, lab, len(frames[seq]), w, h, np.mean(g), np.mean(v)), flush=True)

with open(args.out_csv, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['seq', 'split', 'label', 'n_frames', 'size', 'gray_mean', 'v_mean'])
    wr.writerows(rows)

print('\nmean gray per label (sequence level):')
for lab in ('daylight', 'night', 'fog'):
    g = np.array([r[5] for r in rows if r[2] == lab])
    if len(g):
        print('  {:8s} n {:2d}  mean {:.1f}  median {:.1f}  min {:.1f}  max {:.1f}'.format(lab, len(g), g.mean(), np.median(g), g.min(), g.max()))

with open(args.nat_csv) as f:
    nat = np.array([float(r['gray_mean']) for r in csv.DictReader(f)])
print('\nNAT2021 test (180 sequences): mean gray mean {:.1f} median {:.1f} min {:.1f} max {:.1f}; 25th / 75th percentile {:.1f} / {:.1f}'.format(
    nat.mean(), np.median(nat), nat.min(), nat.max(), *np.percentile(nat, [25, 75])))
print('wrote ' + args.out_csv)
