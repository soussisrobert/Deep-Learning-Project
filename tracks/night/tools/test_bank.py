"""E18: oracle template bank (upper-bound test). Same loop as test.py with HiFTBankTracker.

After frame t has been tracked and its box recorded, the ground-truth box of frame t is passed to tracker.add_gt (frames whose GT is
NaN or has w <= 0 / h <= 0 are skipped); a template added there is first used on frame t + 1. Frame 0 is not passed (its candidate is the frame-1
template). Boxes go to results/<dataset>/first1.2<tag>/<seq>.txt as in test.py. Per frame log (bank/<dataset>/first1.2<tag>/<seq>.csv):
frame, bank size, winner index, then for every bank slot 0..K-1 the peak pscore and the centre error (px, vs the GT centre x + w / 2, y + h / 2;
NaN if the slot is empty or the GT is not usable). Additions per sequence: bank/<dataset>/first1.2<tag>/additions.csv (seq, frame whose GT was used, max similarity).
usage (from tools/): python test_bank.py --dataset NAT2021 --enh_template gamma --enh_search gamma --gamma 0.6 --bank_k 5 --bank_thr 0.52 --tag _bank_K5
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker_bank import HiFTBankTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory


def main():
    parser = argparse.ArgumentParser(description='HiFT tracking with an oracle template bank')
    parser.add_argument('--dataset', default='NAT2021', type=str)
    parser.add_argument('--config', default='./../experiments/config.yaml', type=str)
    parser.add_argument('--snapshot', default='./snapshot/first.pth', type=str)
    parser.add_argument('--enh_template', default='none', type=str)
    parser.add_argument('--enh_search', default='none', type=str)
    parser.add_argument('--denoise', default='none', type=str)
    parser.add_argument('--gamma', default=0.4, type=float)
    parser.add_argument('--clip', default=3.0, type=float)
    parser.add_argument('--bank_k', default=5, type=int)
    parser.add_argument('--bank_thr', default=0.52, type=float)
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--video_list', default='', type=str)
    parser.add_argument('--every', default=1, type=int)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cfg.merge_from_file(args.config)
    cfg.TRACK.ENH_TEMPLATE = args.enh_template
    cfg.TRACK.ENH_SEARCH = args.enh_search
    cfg.TRACK.ENH_GAMMA = args.gamma
    cfg.TRACK.ENH_CLIP = args.clip
    cfg.TRACK.ENH_DENOISE = args.denoise

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../test_dataset', args.dataset)
    model = load_pretrain(ModelBuilder(), args.snapshot).cuda().eval()
    tracker = HiFTBankTracker(model, k=args.bank_k, thr=args.bank_thr)
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)
    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1) + args.tag
    video_names = None
    if args.video_list:
        video_names = set(l.strip() for l in open(args.video_list) if l.strip())

    res_dir = os.path.join('results', args.dataset, model_name)
    bank_dir = os.path.join('bank', args.dataset, model_name)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(bank_dir, exist_ok=True)
    K = args.bank_k
    add_rows = []
    time_rows = []
    for v_idx, video in enumerate(dataset):
        if v_idx % args.every != 0:
            continue
        if video_names is not None and video.name not in video_names:
            continue
        toc = 0
        pred_bboxes = []
        log = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            g = np.array(gt_bbox, dtype=float)
            usable = bool(np.all(np.isfinite(g)) and g[2] > 0 and g[3] > 0)
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(g)
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bboxes.append(gt_bbox_)
            else:
                out = tracker.track(img)
                pred_bboxes.append(out['bbox'])
                last = tracker.last
                row = [idx, len(tracker.bank), last['winner']]
                for s in range(K):
                    if s < len(last['info']):
                        p, px, py = last['info'][s]
                        err = float(np.hypot(px - (g[0] + g[2] / 2.), py - (g[1] + g[3] / 2.))) if usable else float('nan')
                        row += [p, err]
                    else:
                        row += [float('nan'), float('nan')]
                log.append(row)
                if usable:
                    cx, cy, w, h = get_axis_aligned_bbox(g)
                    tracker.add_gt(img, [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h])
            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        time_rows.append([video.name, idx + 1, toc])
        with open(os.path.join(res_dir, video.name + '.txt'), 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        with open(os.path.join(bank_dir, video.name + '.csv'), 'w') as f:
            f.write(','.join(['frame', 'bank_size', 'winner'] + ['p%d,err%d' % (s, s) for s in range(K)]) + '\n')
            for row in log:
                f.write(','.join('{:.6g}'.format(v) for v in row) + '\n')
        for fr, sim in tracker.additions:
            add_rows.append([video.name, fr, sim])
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps  bank {}'.format(
            v_idx + 1, video.name, toc, idx / toc, len(tracker.bank)), flush=True)
    with open(os.path.join(bank_dir, "times.csv"), "w") as f:
        f.write("seq,frames,seconds\n")
        for r in time_rows:
            f.write("{},{},{:.3f}\n".format(*r))
    with open(os.path.join(bank_dir, 'additions.csv'), 'w') as f:
        f.write('seq,frame,max_sim\n')
        for r in add_rows:
            f.write('{},{},{:.5f}\n'.format(*r))


if __name__ == '__main__':
    main()
