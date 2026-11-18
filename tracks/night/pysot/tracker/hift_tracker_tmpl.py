from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import deque

import numpy as np
import torch

from pysot.core.config import cfg
from pysot.tracker.hift_tracker import HiFTTracker


def _mean_feat(fs):
    """Element wise mean of a list of backbone outputs (a tensor, or a list / tuple of tensors)."""
    first = fs[0]
    if isinstance(first, (list, tuple)):
        return type(first)(torch.stack([f[i] for f in fs]).mean(0) for i in range(len(first)))
    return torch.stack(fs).mean(0)


def _blend(f0, f1, alpha):
    """(1 - alpha) * f0 + alpha * f1 for backbone outputs."""
    if isinstance(f0, (list, tuple)):
        return type(f0)((1 - alpha) * f0[i] + alpha * f1[i] for i in range(len(f0)))
    return (1 - alpha) * f0 + alpha * f1


class HiFTTemplateTracker(HiFTTracker):
    """HiFT with a template that is rebuilt after frames from the last k template crops.

    mode 'pixel':   average the last k 127x127 template crops (frame 1 crop included until it drops out), then run the backbone
    mode 'feature': run the backbone on every crop and average the last k feature sets
    alpha (feature mode only, None = off): anchor on the frame 1 template,
        template = (1 - alpha) * frame 1 features + alpha * mean of the last k features
    gate ('none' or 'median'): with 'median' the template is only updated when this frame's best score is at least the
        median of the best scores of all earlier frames of the sequence (after 'warmup' frames; before that every frame
        updates). The scale of the score is not known, so the gate is relative to the sequence itself.
    The crop is taken at the new box with the same context rule and the same enhancement as the original template.
    With alpha=None and gate='none' this is exactly the E7 tracker.
    """

    def __init__(self, model, mode='feature', k=3, alpha=None, gate='none', warmup=10):
        super(HiFTTemplateTracker, self).__init__(model)
        assert mode in ('pixel', 'feature'), mode
        assert gate in ('none', 'median'), gate
        assert alpha is None or mode == 'feature', 'the anchor is only implemented for feature averaging'
        self.mode = mode
        self.k = k
        self.alpha = alpha
        self.gate = gate
        self.warmup = warmup

    def init(self, img, bbox):
        super(HiFTTemplateTracker, self).init(img, bbox)
        self.z_hist = deque([self.template], maxlen=self.k)
        self.f_hist = deque([self.model.zf], maxlen=self.k)
        self.f0 = self.model.zf
        self.score_hist = []
        self.n_updates = 0

    def track(self, img):
        out = super(HiFTTemplateTracker, self).track(img)
        score = float(out['best_score'])
        accept = True
        if self.gate == 'median' and len(self.score_hist) >= self.warmup:
            accept = score >= float(np.median(self.score_hist))
        self.score_hist.append(score)
        if not accept:
            return out
        self.n_updates += 1
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scaleaa
        z = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,
                               s_z, self.channel_average, self.enh_t)
        if self.mode == 'pixel':
            self.z_hist.append(z)
            self.model.template(torch.stack(list(self.z_hist)).mean(0))
        else:
            self.model.template(z)
            self.f_hist.append(self.model.zf)
            avg = _mean_feat(list(self.f_hist))
            self.model.zf = avg if self.alpha is None else _blend(self.f0, avg, self.alpha)
        return out
