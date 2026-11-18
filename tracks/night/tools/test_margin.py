"""Same as test.py / test_tmpl.py, but the tracker also logs the top-1 / top-2 peak of the final score map per frame.

--tmpl_update none      plain HiFT (the E0 / E4 trackers); pixel | feature as in test_tmpl.py (E7 / E8 / E10).
Boxes are written exactly as before to results/<dataset>/<name>/<seq>.txt (they must equal the runs without logging);
the margin log goes to margin/<dataset>/<name>/<seq>.csv.
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
from pysot.tracker.hift_tracker_margin import HiFTMarginTracker, HiFTMarginTemplateTracker, MARGIN_COLUMNS
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory


def main():
    parser = argparse.ArgumentParser(description='HiFT tracking with score margin logging')
    parser.add_argument('--dataset', default='NAT2021', type=str)
    parser.add_argument('--config', default='./../experiments/config.yaml', type=str)
    parser.add_argument('--snapshot', default='./snapshot/first.pth', type=str)
    parser.add_argument('--enh_template', default='none', type=str)
    parser.add_argument('--enh_search', default='none', type=str)
    parser.add_argument('--denoise', default='none', type=str)
    parser.add_argument('--gamma', default=0.4, type=float)
    parser.add_argument('--clip', default=3.0, type=float)
    parser.add_argument('--tag', default='', type=str, help='suffix appended to the result folder name')
    parser.add_argument('--tmpl_update', default='none', type=str, choices=['none', 'pixel', 'feature'])
    parser.add_argument('--tmpl_k', default=3, type=int)
    parser.add_argument('--tmpl_alpha', default=-1.0, type=float)
    parser.add_argument('--tmpl_gate', default='none', type=str, choices=['none', 'median'])
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

    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cuda().eval()

    if args.tmpl_update == 'none':
        tracker = HiFTMarginTracker(model)
    else:
        tracker = HiFTMarginTemplateTracker(model, mode=args.tmpl_update, k=args.tmpl_k,
                                            alpha=None if args.tmpl_alpha < 0 else args.tmpl_alpha,
                                            gate=args.tmpl_gate)

    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]+str(cfg.TRACK.w1)+args.tag

    video_names = None
    if args.video_list:
        video_names = set(l.strip() for l in open(args.video_list) if l.strip())

    for v_idx, video in enumerate(dataset):
        if v_idx % args.every != 0:
            continue
        if video_names is not None and video.name not in video_names:
            continue
        toc = 0
        pred_bboxes = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bboxes.append(gt_bbox_)
            else:
                outputs = tracker.track(img)
                pred_bboxes.append(outputs['bbox'])
            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, '{}.txt'.format(video.name)), 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')

        margin_path = os.path.join('margin', args.dataset, model_name)
        if not os.path.isdir(margin_path):
            os.makedirs(margin_path)
        with open(os.path.join(margin_path, '{}.csv'.format(video.name)), 'w') as f:
            f.write(','.join(MARGIN_COLUMNS)+'\n')
            for row in tracker.margin_log:
                f.write(','.join(['{:.6g}'.format(v) for v in row])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc), flush=True)


if __name__ == '__main__':
    main()
