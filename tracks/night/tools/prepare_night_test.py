"""Build pysot-toolkit json files (NATDataset format) for the held-out night / daytime evaluation sets.
Images stay where they are (absolute paths in img_names; os.path.join(root, absolute) is the absolute path).

  NATUD135VAL   UAVDark135 val split (tools/splits/UAVDark135_val.txt)        used to choose the checkpoint
  NATUD135TEST  UAVDark135 test split                                         final test
  NATUAV123HELD UAV123 held-out daytime set: sequences of the 123 list whose frame folder has the prefix group, wakeboard or boat
                (tools/splits/UAV123_test.txt), undarkened
Absent-target rows are kept as NaN, as in the original annotation (the toolkit counts those frames as failures for every tracker).
usage: python -I tools/prepare_night_test.py
"""
import os
import re
import json
import glob
import numpy as np

UD = 'training_dataset/UAVDark135/extracted/UAVDark135_TSP_out'
U123 = 'training_dataset/UAV123/extracted/UAV123'


def write(name, meta):
    d = os.path.join('test_dataset', name)
    os.makedirs(d, exist_ok=True)
    json.dump(meta, open(os.path.join(d, name + '.json'), 'w'))
    n = sum(len(v['img_names']) for v in meta.values())
    nan = sum(int(np.isnan(np.array(v['gt_rect'])).any(axis=1).sum()) for v in meta.values())
    print('{}: {} sequences, {} frames, {} absent-target frames (NaN), written to {}/{}.json'.format(name, len(meta), n, nan, d, name))


for name, split in (('NATUD135VAL', 'val'), ('NATUD135TEST', 'test')):
    meta = {}
    for s in [l.strip() for l in open('tools/splits/UAVDark135_{}.txt'.format(split)) if l.strip()]:
        fr = [os.path.abspath(f) for f in sorted(glob.glob('{}/Sequences/{}/*.jpg'.format(UD, s)))]
        gt = np.loadtxt('{}/anno_revise/{}.txt'.format(UD, s), delimiter=',', ndmin=2)
        assert len(fr) == len(gt), s
        assert not np.isnan(gt[0]).any(), 'first box is NaN: ' + s
        meta[s] = {'video_dir': s, 'init_rect': gt[0].tolist(), 'img_names': fr, 'gt_rect': gt.tolist(), 'attr': []}
    write(name, meta)

cfg = open(U123 + '/configSeqs.m', errors='ignore').read()
structs = re.findall(r"struct\('name','([^']+)','path','[^']*\\([^\\']+)\\','startFrame',(\d+),'endFrame',(\d+)", cfg)
held = set(l.strip() for l in open('tools/splits/UAV123_test.txt') if l.strip())
meta, skipped = {}, []
for name, folder, st, en in structs:
    f = '{}/anno/UAV123/{}.txt'.format(U123, name)
    if folder not in held or not os.path.exists(f):   # the long-term (UAV20L) list repeats the same frames
        continue
    gt = np.loadtxt(f, delimiter=',', ndmin=2)
    fr = ['{}/data_seq/UAV123/{}/{:06d}.jpg'.format(os.path.abspath(U123), folder, i) for i in range(int(st), int(en) + 1)]
    assert len(fr) == len(gt), name
    if np.isnan(gt[0]).any():
        skipped.append(name)
        continue
    meta[name] = {'video_dir': folder, 'init_rect': gt[0].tolist(), 'img_names': fr, 'gt_rect': gt.tolist(), 'attr': []}
print('UAV123 held-out sequences skipped because the first box is absent:', skipped)
write('NATUAV123HELD', meta)
