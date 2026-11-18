from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.tracker.hift_tracker_tmpl import HiFTTemplateTracker


class HiFTMarginTracker(HiFTTracker):
    """HiFTTracker whose track() is a copy of the original with logging added; the boxes it returns must be identical.

    Per tracked frame the log keeps, for the final score map (pscore = penalty * score, then the window mix, exactly
    what the argmax is taken over):
      p1    the top-1 value,
      p2a   the second highest value anywhere (usually the cell next to the top-1, so mostly the same peak),
      p2d   the highest value among cells at least 2 grid cells away from the top-1 (Chebyshev distance); such a cell is
            far in the grid but often decodes to nearly the same box,
      p2x   the highest value among cells whose decoded box centre is at least 0.3 x target size from the top-1 box
            centre (0 if there is no such cell), n_distinct = how many cells qualify; this is the "different peak",
      the grid cell (row, col) of the first two, the raw (unsmoothed) box each of the three decodes to, in image coordinates.
    The map is score_size x score_size (cfg.TRAIN.OUTPUT_SIZE), one cell every 8 px of the search crop.
    """

    def init(self, img, bbox):
        super(HiFTMarginTracker, self).init(img, bbox)
        self.margin_log = []
        self.frame_no = 0

    def track(self, img):
        self.frame_no += 1
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        if self.size[0]*self.size[1] >0.5*img.shape[0]*img.shape[1]:
            s_z=self.scaleaa
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z

        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average, self.enh_s)

        outputs = self.model.track(x_crop)
        pred_bbox=self.generate_anchor(outputs['loc']).transpose()
        score1 = self._convert_score(outputs['cls1'])*cfg.TRACK.w2
        score2=(outputs['cls2']).view(-1).cpu().detach().numpy()*cfg.TRACK.w3
        score=(score1+score2)/2

        def change(r):
            return np.maximum(r, 1. / (r+1e-5))

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/(self.size[1]+1e-5)) /
                     (pred_bbox[2, :]/(pred_bbox[3, :]+1e-5)))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        # ---- logging only, nothing below depends on it
        g = self.score_size
        n_cells = len(pscore)
        order = np.argsort(-pscore)
        idx_a = int(order[1])
        rows = np.arange(n_cells) // g
        cols = np.arange(n_cells) % g
        r1, c1 = divmod(int(best_idx), g)
        far = np.maximum(np.abs(rows - r1), np.abs(cols - c1)) >= 2
        idx_d = int(np.argmax(np.where(far, pscore, -np.inf)))

        def img_box(i):
            b = pred_bbox[:, i] / scale_z
            return [float(b[0] + self.center_pos[0]), float(b[1] + self.center_pos[1]), float(b[2]), float(b[3])]

        box_1 = img_box(int(best_idx))
        box_d = img_box(idx_d)

        # cells cannot be told apart by their grid position alone: cells 16 px apart often regress to nearly the same
        # box. A peak is a different hypothesis when the box it decodes to has its centre at least 0.3 x the target
        # size (mean of w and h of the top-1 box) away from the centre of the top-1 box.
        off = pred_bbox[:2, :] / scale_z
        dist = np.hypot(off[0] - off[0][best_idx], off[1] - off[1][best_idx])
        distinct = dist >= 0.3 * 0.5 * (box_1[2] + box_1[3])
        n_distinct = int(distinct.sum())
        if n_distinct:
            idx_x = int(np.argmax(np.where(distinct, pscore, -np.inf)))
            p2x = float(pscore[idx_x])
            box_x = img_box(idx_x)
        else:
            p2x = 0.0
            box_x = [float('nan')] * 4
        self.margin_log.append(
            [self.frame_no, float(pscore[best_idx]), float(pscore[idx_a]), float(pscore[idx_d]),
             r1, c1, idx_d // g, idx_d % g, float(score[best_idx]), float(score[idx_d])] + box_1 + box_d
            + [p2x, n_distinct] + box_x)
        # ---- end logging

        bbox = pred_bbox[:, best_idx] / scale_z

        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score,
               }


MARGIN_COLUMNS = ['frame', 'p1', 'p2a', 'p2d', 'r1', 'c1', 'r2d', 'c2d', 's1', 's2d',
                  'cx1', 'cy1', 'w1', 'h1', 'cx2d', 'cy2d', 'w2d', 'h2d',
                  'p2x', 'n_distinct', 'cx2x', 'cy2x', 'w2x', 'h2x']


class HiFTMarginTemplateTracker(HiFTTemplateTracker, HiFTMarginTracker):
    """The template tracker (E7/E8/E10) on top of the logging track(): MRO is this, template tracker, margin tracker,
    HiFTTracker, so the template tracker's super().track() reaches the logging copy."""
    pass
