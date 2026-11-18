"""NaN-safe candidate selection for the upstream HiFTTracker (isolated inference correction).

Upstream track() computes sqrt((w+pad)*(h+pad)) for every cell's decoded box. Cells predicting
inverted boxes (negative extent; never supervised outside the target region) yield NaN scale
penalties, and np.argmax returns the FIRST NaN, so a garbage cell is selected. This subclass
is the upstream track() with exactly one change: NaN/inf penalized scores are never selected.
If every cell is non-finite it raises rather than guessing. Counts are kept for disclosure.
"""
import numpy as np


def make_nan_safe_tracker(model):
    from pysot.core.config import cfg
    from pysot.tracker.hift_tracker import HiFTTracker

    class NanSafeHiFTTracker(HiFTTracker):
        nonfinite_cells = 0
        frames = 0
        first_nan_would_have_won = 0

        def track(self, img):
            w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z = np.sqrt(w_z * h_z)
            if self.size[0]*self.size[1] > 0.5*img.shape[0]*img.shape[1]:
                s_z = self.scaleaa
            scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE, round(s_x), self.channel_average)
            outputs = self.model.track(x_crop)
            pred_bbox = self.generate_anchor(outputs['loc']).transpose()
            score1 = self._convert_score(outputs['cls1'])*cfg.TRACK.w2
            score2 = (outputs['cls2']).view(-1).cpu().detach().numpy()*cfg.TRACK.w3
            score = (score1+score2)/2

            def change(r):
                return np.maximum(r, 1. / (r+1e-5))

            def sz(w, h):
                pad = (w + h) * 0.5
                with np.errstate(invalid='ignore'):
                    return np.sqrt((w + pad) * (h + pad))

            s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
            r_c = change((self.size[0]/(self.size[1]+1e-5)) / (pred_bbox[2, :]/(pred_bbox[3, :]+1e-5)))
            penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
            pscore = penalty * score
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
            finite = np.isfinite(pscore)
            type(self).frames += 1
            type(self).nonfinite_cells += int((~finite).sum())
            if not finite.all() and not np.isfinite(pscore[np.argmax(pscore)]):
                type(self).first_nan_would_have_won += 1
            if not finite.any():
                raise ValueError("all candidate scores non-finite")
            best_idx = int(np.argmax(np.where(finite, pscore, -np.inf)))  # the only change from upstream
            bbox = pred_bbox[:, best_idx] / scale_z
            lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]
            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
            cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])
            return {'bbox': [cx - width / 2, cy - height / 2, width, height], 'best_score': score[best_idx]}

    return NanSafeHiFTTracker(model)
