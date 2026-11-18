from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from pysot.core.config import cfg
from pysot.tracker.hift_tracker import HiFTTracker


def flat(zf):
    """Same definition as tools/template_sim.py (E17): the backbone tuple, every tensor reshaped to 1-D and concatenated."""
    if isinstance(zf, (list, tuple)):
        return torch.cat([f.reshape(-1) for f in zf])
    return zf.reshape(-1)


class HiFTBankTracker(HiFTTracker):
    """E18: oracle template bank. Plain HiFT (no anchored / averaged template update) with a bank of up to K templates.

    bank[0] is the frame-1 template, built by HiFTTracker.init. After a frame has been tracked, add_gt() crops a candidate at the
    GROUND-TRUTH box of that frame (E17 rule) and appends it when its largest cosine similarity to the bank is below `thr` and the bank
    holds fewer than K entries (nothing is ever evicted). A template appended after frame t is first used on frame t + 1.
    track(): the search backbone runs once, the head once per bank template; every template gives pscore (penalty * score mixed with the
    window, the array the argmax is taken over); the template with the highest pscore peak wins (ties: the lower index) and the box and
    state update are computed from that template's output only, with the unchanged formulas of HiFTTracker.track.
    With K = 1 this is HiFTTracker.track.
    """

    def __init__(self, model, k=5, thr=0.52):
        super(HiFTBankTracker, self).__init__(model)
        self.k = k
        self.thr = thr

    def init(self, img, bbox):
        super(HiFTBankTracker, self).init(img, bbox)
        self.bank = [self.model.zf]
        self.bank_flat = [flat(self.model.zf)]
        self.additions = []
        self.frame_no = 0
        self.last = None

    def add_gt(self, img, box):
        """box = (x, y, w, h) in the same convention as init (centre x + (w - 1) / 2). Call after track() of the same frame."""
        if len(self.bank) >= self.k:
            return
        centre = np.array([box[0] + (box[2] - 1) / 2, box[1] + (box[3] - 1) / 2])
        size = np.array([box[2], box[3]])
        w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        s_z = round(np.sqrt(w_z * h_z))
        if size[0] * size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scaleaa
        z = self.get_subwindow(img, centre, cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average, self.enh_t)
        with torch.no_grad():
            zf = self.model.backbone(z)
        f = flat(zf)
        sims = [float(torch.nn.functional.cosine_similarity(b, f, dim=0)) for b in self.bank_flat]
        if max(sims) < self.thr:
            self.bank.append(zf)
            self.bank_flat.append(f)
            self.additions.append((self.frame_no, max(sims)))

    def track(self, img):
        self.frame_no += 1
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        if self.size[0] * self.size[1] > 0.5 * img.shape[0] * img.shape[1]:
            s_z = self.scaleaa
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average, self.enh_s)

        with torch.no_grad():
            xf = self.model.backbone(x_crop)

        def change(r):
            return np.maximum(r, 1. / (r + 1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        cands = []
        for zf in self.bank:
            with torch.no_grad():
                loc, cls1, cls2 = self.model.grader(xf, zf)
            pred_bbox = self.generate_anchor(loc).transpose()
            score1 = self._convert_score(cls1) * cfg.TRACK.w2
            score2 = cls2.view(-1).cpu().detach().numpy() * cfg.TRACK.w3
            score = (score1 + score2) / 2
            s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                         (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
            r_c = change((self.size[0] / (self.size[1] + 1e-5)) /
                         (pred_bbox[2, :] / (pred_bbox[3, :] + 1e-5)))
            penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
            pscore = penalty * score
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx = np.argmax(pscore)
            cands.append((float(pscore[best_idx]), best_idx, pred_bbox, penalty, score))

        win = int(np.argmax([c[0] for c in cands]))
        # per template: peak pscore and the centre its argmax box decodes to (image coordinates), for the diagnostics
        info = []
        for p, bi, pb, _, _ in cands:
            b = pb[:, bi] / scale_z
            info.append((p, float(b[0] + self.center_pos[0]), float(b[1] + self.center_pos[1])))
        self.last = {'winner': win, 'info': info}

        _, best_idx, pred_bbox, penalty, score = cands[win]
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2, cy - height / 2, width, height]
        return {'bbox': bbox, 'best_score': score[best_idx]}
