# Copyright (c) SenseTime. All Rights Reserved.

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
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

def main():
    parser = argparse.ArgumentParser(description='HiFT tracking')
    parser.add_argument('--dataset', default='DTB70',type=str,
            help='datasets')
    parser.add_argument('--config', default='./../experiments/config.yaml', type=str,
            help='config file')
    parser.add_argument('--snapshot', default='./snapshot/first.pth', type=str,
            help='snapshot of models to eval')
    parser.add_argument('--video', default='', type=str,
            help='eval one special video')
    parser.add_argument('--vis', default='',action='store_true',
            help='whether visualzie result')
    parser.add_argument('--enh_template', default='none', type=str,
            help='none | gamma | clahe | gamma_clahe, applied to the template crop')
    parser.add_argument('--enh_search', default='none', type=str,
            help='none | gamma | clahe | gamma_clahe, applied to the search crop')
    parser.add_argument('--denoise', default='none', type=str,
            help='none | gauss3 | gauss5 | bilateral | nlm, applied before the enhancement to both crops')
    parser.add_argument('--gamma', default=0.4, type=float)
    parser.add_argument('--clip', default=3.0, type=float, help='CLAHE clip limit')
    parser.add_argument('--tag', default='', type=str,
            help='suffix appended to the result folder name')
    parser.add_argument('--reinit_every', default=0, type=int,
            help='DIAGNOSTIC (uses ground truth): reset the tracker box/scale to the ground truth every N frames, 0 = never')
    parser.add_argument('--reinit_mode', default='box', type=str, choices=['box', 'full'],
            help="box: keep the original template; full: also rebuild the template at the reset")
    parser.add_argument('--video_list', default='', type=str,
            help='text file with one sequence name per line; only those are run')
    parser.add_argument('--every', default=1, type=int,
            help='only run every Nth video (sorted by name), 1 = all')
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

    tracker = HiFTTracker(model)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]+str(cfg.TRACK.w1)+args.tag

    video_names = None
    if args.video_list:
        video_names = set(l.strip() for l in open(args.video_list) if l.strip())

    for v_idx, video in enumerate(dataset):

            if v_idx % args.every != 0:
                continue
            if video_names is not None and video.name not in video_names:
                continue

            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                # diagnostic only (uses the ground truth): every N frames put the tracker's box and scale
                # back on the ground truth AFTER this frame's prediction was recorded. 'box' keeps the
                # original template, 'full' also rebuilds the template from this frame.
                if args.reinit_every > 0 and idx > 0 and idx % args.reinit_every == 0:
                    gt_arr = np.array(gt_bbox, dtype=float)
                    if np.all(np.isfinite(gt_arr)) and gt_arr[2] > 0 and gt_arr[3] > 0:
                        rcx, rcy, rw, rh = get_axis_aligned_bbox(gt_arr)
                        gt_reset = [rcx-(rw-1)/2, rcy-(rh-1)/2, rw, rh]
                        if args.reinit_mode == 'full':
                            tracker.init(img, gt_reset)
                        else:
                            tracker.reset_box(gt_reset)
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis:
                    try:
                        gt_bbox = list(map(int, gt_bbox))
                    except:
                        gt_bbox=[0,0,0,0]
                        pred_bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                    (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                    (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow(video.name, img)
                        cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
        
            model_path = os.path.join('results', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))

if __name__ == '__main__':
    main()
