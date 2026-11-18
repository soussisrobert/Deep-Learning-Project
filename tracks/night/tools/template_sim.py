"""E17 step 1 (GPU, inference only): template-feature similarity to frame 1 at every 10th frame of NAT2021.

For every sequence: frame 0 is the reference. At frames 0, 10, 20, ... with a usable GT box (finite, w > 0, h > 0) a template is cropped at the GT box
with the tracker's own template rule (HiFTTracker.init / HiFTTemplateTracker.track: centre x + (w - 1) / 2 after get_axis_aligned_bbox, s_z = round(sqrt((w + p)(h + p))),
p = CONTEXT_AMOUNT (w + h); box area > half the image: s_z of frame 0; channel average of frame 0; gamma 0.6 enhancement, as in first_ag),
passed through the backbone, and the cosine similarity between the flattened frame-0 features and this frame's features is written.
The backbone output (a tuple of tensors) is concatenated after flattening; no pooling.
usage (from tools/): python template_sim.py
output: illumination/E17_template_sim.csv  (dataset, seq, frame, sim)
"""
import os
import sys
import csv
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('..'))
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

EVERY = 10


def flat(zf):
    if isinstance(zf, (list, tuple)):
        return torch.cat([f.reshape(-1) for f in zf])
    return zf.reshape(-1)


def main():
    cfg.merge_from_file('../experiments/config.yaml')
    cfg.TRACK.ENH_TEMPLATE = 'gamma'
    cfg.TRACK.ENH_SEARCH = 'gamma'
    cfg.TRACK.ENH_GAMMA = 0.6
    cfg.TRACK.ENH_CLIP = 3.0
    cfg.TRACK.ENH_DENOISE = 'none'
    torch.set_num_threads(1)
    model = load_pretrain(ModelBuilder(), 'snapshot/first.pth').cuda().eval()
    tracker = HiFTTracker(model)
    dataset = DatasetFactory.create_dataset(name='NAT2021', dataset_root='../test_dataset/NAT2021', load_img=False)
    rows = []
    for v_idx, video in enumerate(dataset):
        f0 = None
        for idx, (img, gt) in enumerate(video):
            if idx % EVERY != 0:
                continue
            g = np.array(gt, dtype=float)
            if not (np.all(np.isfinite(g)) and g[2] > 0 and g[3] > 0):
                continue
            cx, cy, w, h = get_axis_aligned_bbox(g)
            box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            if idx == 0:
                tracker.init(img, box)
                f0 = flat(model.zf)
                rows.append(['NAT2021', video.name, 0, 1.0])
                continue
            if f0 is None:
                continue
            size = np.array([box[2], box[3]])
            centre = np.array([box[0] + (box[2] - 1) / 2, box[1] + (box[3] - 1) / 2])
            w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
            h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
            s_z = round(np.sqrt(w_z * h_z))
            if size[0] * size[1] > 0.5 * img.shape[0] * img.shape[1]:
                s_z = tracker.scaleaa
            z = tracker.get_subwindow(img, centre, cfg.TRACK.EXEMPLAR_SIZE, s_z, tracker.channel_average, tracker.enh_t)
            model.template(z)
            f = flat(model.zf)
            sim = float(torch.nn.functional.cosine_similarity(f0, f, dim=0))
            rows.append(['NAT2021', video.name, idx, sim])
        print('({:3d}) {} {} samples so far'.format(v_idx + 1, video.name, len(rows)), flush=True)
    with open('illumination/E17_template_sim.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'seq', 'frame', 'sim'])
        w.writerows(rows)
    print('wrote illumination/E17_template_sim.csv', len(rows))


if __name__ == '__main__':
    main()
