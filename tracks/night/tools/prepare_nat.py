"""Build <name>.json (pysot-toolkit format) for an extracted NAT2021 / NAT2021L folder.

Expected layout under dataset_root:
    anno/<seq>.txt        gt boxes, one 'x,y,w,h' per frame
    anno/att/<seq>.txt    12 comma separated 0/1 attribute flags
    data_seq/<seq>/*.jpg  frames
"""
import os
import re
import json
import argparse
from glob import glob


def read_boxes(path):
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                boxes.append([float(v) for v in re.split(r'[,\s]+', line)])
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--name', required=True, help="'NAT2021' or 'NAT2021L'")
    args = parser.parse_args()

    root = args.dataset_root
    meta = {}
    for anno in sorted(glob(os.path.join(root, 'anno', '*.txt'))):
        seq = os.path.splitext(os.path.basename(anno))[0]
        frames = sorted(glob(os.path.join(root, 'data_seq', seq, '*.jpg')))
        if not frames:
            print('no frames for', seq)
            continue
        gt = read_boxes(anno)
        if len(gt) != len(frames):
            print('{}: {} boxes vs {} frames'.format(seq, len(gt), len(frames)))
        attr = []
        att_file = os.path.join(root, 'anno', 'att', seq + '.txt')
        if os.path.exists(att_file):
            with open(att_file) as f:
                flags = [int(v) for v in re.split(r'[,\s]+', f.read().strip()) if v]
            attr = ['ATT{:02d}'.format(i + 1) for i, v in enumerate(flags) if v]
        meta[seq] = {
            'video_dir': os.path.join('data_seq', seq),
            'init_rect': gt[0],
            'img_names': [os.path.relpath(x, root) for x in frames],
            'gt_rect': gt,
            'attr': attr,
        }
    out = os.path.join(root, args.name + '.json')
    with open(out, 'w') as f:
        json.dump(meta, f)
    print('wrote', out, len(meta), 'videos')


if __name__ == '__main__':
    main()
