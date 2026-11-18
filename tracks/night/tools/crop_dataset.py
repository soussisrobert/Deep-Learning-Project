"""Convert one labeled dataset to the PySOT training format: crop511 'x' images + a json the loader (pysot/datasets/dataset.py) reads.

Same crop as training_dataset/vid/par_crop.py crop_like_SiamFC (x crop only: the loader reads only '<frame>.<track>.x.jpg'),
with float64 instead of np.float (numpy 2 has no np.float). Padding = mean colour of the frame. JPEG quality 95 (cv2 default).
Json: {video: {track: {"000123": [x1, y1, x2, y2]}}} with boxes in ORIGINAL frame pixels (the loader only uses width and height).
Frames with an absent target (NaN), width or height < 4 px are skipped at plan time. Existing crops are not rewritten.

usage: python -I tools/crop_dataset.py <uavdark|uav123|uavdt> <train|val> <out_dir> <max_home_mb> <est_mb> [workers]
writes <out_dir>/crop511/<video>/<frame>.<track>.x.jpg, <out_dir>/<split>.json (+ <out_dir>/meta.json for UAVDT: class per track)
"""
import os
import sys
import glob
import json
import subprocess
import numpy as np
import cv2
from multiprocessing import Pool

ds, split, out, max_mb, est_mb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
workers = int(sys.argv[6]) if len(sys.argv) > 6 else 5
UD = 'training_dataset/UAVDark135/extracted/UAVDark135_TSP_out'
U123 = 'training_dataset/UAV123/extracted/UAV123'
BM = glob.glob('training_dataset/UAVDT/extracted/benchmark_M/*')[0] if ds == 'uavdt' else ''


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


def plan():
    """-> list of (video, track, frame, image_path, box) and meta"""
    items, meta = [], {}
    if ds == 'uavdark':
        for s in [l.strip() for l in open('tools/splits/UAVDark135_{}.txt'.format(split)) if l.strip()]:
            fr = sorted(glob.glob('{}/Sequences/{}/*.jpg'.format(UD, s)))
            a = np.loadtxt('{}/anno_revise/{}.txt'.format(UD, s), delimiter=',', ndmin=2)
            for i in range(min(len(fr), len(a))):
                if not np.isnan(a[i]).any() and a[i][2] >= 4 and a[i][3] >= 4:
                    items.append((s, '00', i + 1, fr[i], [float(a[i][0]), float(a[i][1]), float(a[i][0] + a[i][2]), float(a[i][1] + a[i][3])]))
    elif ds == 'uav123':
        assert split == 'train'
        for key, frames in json.load(open('tools/splits/UAV123_train_tracks.json')).items():
            folder, name = key.split('/')
            for f, b in frames.items():
                items.append((folder, name, int(f), '{}/data_seq/UAV123/{}/{:06d}.jpg'.format(U123, folder, int(f)), b))
    elif ds == 'uavdt':
        assert split == 'train'
        for s, tr in json.load(open('tools/splits/UAVDT_selection.json')).items():
            for i, t in tr.items():
                meta.setdefault(s, {})['{:02d}'.format(int(i))] = {'class': t['class'], 'weather': t['weather']}
                for f, b in t['frames'].items():
                    items.append((s, '{:02d}'.format(int(i)), int(f), '{}/{}/img{:06d}.jpg'.format(BM, s, int(f)), b))
    return items, meta


def work(chunk):
    done = 0
    for video, track, frame, path, box in chunk:
        o = '{}/crop511/{}/{:06d}.{}.x.jpg'.format(out, video, frame, track)
        if os.path.exists(o) and os.path.getsize(o) > 0:
            done += 1
            continue
        im = cv2.imread(path)
        if im is None:
            continue
        os.makedirs(os.path.dirname(o), exist_ok=True)
        if cv2.imwrite(o, crop_x511(im, box)):
            done += 1
    return done


if __name__ == '__main__':
    used = int(subprocess.check_output(['du', '-sm', os.path.expanduser('~')]).split()[0])
    print('home before {} MB; estimated output {} MB; limit {} MB'.format(used, est_mb, max_mb), flush=True)
    if used + est_mb > max_mb:
        sys.exit('would exceed the limit, not cropping')
    items, meta = plan()
    print('{} {}: {} crops planned'.format(ds, split, len(items)), flush=True)
    os.makedirs(out + '/crop511', exist_ok=True)
    k = max(1, len(items) // (workers * 20))
    chunks = [items[i:i + k] for i in range(0, len(items), k)]
    done = 0
    with Pool(workers) as p:
        for n, d in enumerate(p.imap_unordered(work, chunks), 1):
            done += d
            if n % 50 == 0:
                print('   {}/{} chunks, {} crops'.format(n, len(chunks), done), flush=True)
    js = {}
    for video, track, frame, path, box in items:
        js.setdefault(video, {}).setdefault(track, {})['{:06d}'.format(frame)] = box
    json.dump(js, open('{}/{}.json'.format(out, split), 'w'))
    if meta:
        json.dump(meta, open(out + '/meta.json', 'w'))
    on_disk = len(glob.glob(out + '/crop511/*/*.x.jpg'))
    print('crops written/present {} of {} planned; on disk in {}: {}'.format(done, len(items), out + '/crop511', on_disk))
    print('videos {}, tracks {}'.format(len(js), sum(len(v) for v in js.values())))
    print('home after {} MB'.format(int(subprocess.check_output(['du', '-sm', os.path.expanduser('~')]).split()[0])))
    if done != len(items):
        sys.exit('MISSING CROPS')
