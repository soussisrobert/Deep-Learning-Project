"""E21: test_tmpl.py (template feature average, anchor, gamma) with the three size-related tracker parameters passable on the command line.
--lr / --penalty_k / --window_influence override cfg.TRACK.LR / PENALTY_K / WINDOW_INFLUENCE after the config file is merged (default: leave the config value).
Boxes go to results/<dataset>/first1.2<tag>/<seq>.txt exactly as test_tmpl.py; per-sequence frame counts and seconds to timing/<dataset>/first1.2<tag>/times.csv.
usage (from tools/): python test_tmpl_sweep.py --dataset NATUD135VAL --enh_template gamma --enh_search gamma --gamma 0.6 --tmpl_update feature --tmpl_k 3 --tmpl_alpha 0.2 --lr 0.3 --penalty_k 0.08 --window_influence 0.42 --tag _E21_...
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
from pysot.tracker.hift_tracker_tmpl import HiFTTemplateTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory


def main():
    parser = argparse.ArgumentParser(description='HiFT tracking with size-parameter overrides')
    parser.add_argument('--dataset', default='NATUD135VAL', type=str)
    parser.add_argument('--config', default='./../experiments/config.yaml', type=str)
    parser.add_argument('--snapshot', default='./snapshot/first.pth', type=str)
    parser.add_argument('--enh_template', default='none', type=str)
    parser.add_argument('--enh_search', default='none', type=str)
    parser.add_argument('--denoise', default='none', type=str)
    parser.add_argument('--gamma', default=0.4, type=float)
    parser.add_argument('--clip', default=3.0, type=float)
    parser.add_argument('--tmpl_update', default='feature', type=str, choices=['pixel', 'feature'])
    parser.add_argument('--tmpl_k', default=3, type=int)
    parser.add_argument('--tmpl_alpha', default=-1.0, type=float)
    parser.add_argument('--tmpl_gate', default='none', type=str, choices=['none', 'median'])
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--penalty_k', default=None, type=float)
    parser.add_argument('--window_influence', default=None, type=float)
    parser.add_argument('--tag', default='', type=str)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cfg.merge_from_file(args.config)
    cfg.TRACK.ENH_TEMPLATE = args.enh_template
    cfg.TRACK.ENH_SEARCH = args.enh_search
    cfg.TRACK.ENH_GAMMA = args.gamma
    cfg.TRACK.ENH_CLIP = args.clip
    cfg.TRACK.ENH_DENOISE = args.denoise
    if args.lr is not None:
        cfg.TRACK.LR = args.lr
    if args.penalty_k is not None:
        cfg.TRACK.PENALTY_K = args.penalty_k
    if args.window_influence is not None:
        cfg.TRACK.WINDOW_INFLUENCE = args.window_influence
    print('TRACK.LR {} PENALTY_K {} WINDOW_INFLUENCE {}'.format(cfg.TRACK.LR, cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE), flush=True)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../test_dataset', args.dataset)
    model = load_pretrain(ModelBuilder(), args.snapshot).cuda().eval()
    tracker = HiFTTemplateTracker(model, mode=args.tmpl_update, k=args.tmpl_k,
                                  alpha=None if args.tmpl_alpha < 0 else args.tmpl_alpha, gate=args.tmpl_gate)
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)
    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1) + args.tag
    res_dir = os.path.join('results', args.dataset, model_name)
    time_dir = os.path.join('timing', args.dataset, model_name)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    rows = []
    for v_idx, video in enumerate(dataset):
        toc = 0
        pred_bboxes = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bboxes.append(gt_bbox_)
            else:
                pred_bboxes.append(tracker.track(img)['bbox'])
            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        with open(os.path.join(res_dir, video.name + '.txt'), 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        rows.append((video.name, idx + 1, toc))
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx + 1, video.name, toc, idx / toc), flush=True)
    with open(os.path.join(time_dir, 'times.csv'), 'w') as f:
        f.write('seq,frames,seconds\n')
        for r in rows:
            f.write('{},{},{:.3f}\n'.format(*r))


if __name__ == '__main__':
    main()
