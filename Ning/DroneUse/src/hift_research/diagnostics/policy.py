"""Observe unmodified tracker outputs and reconstruct its selection mathematics.

No ground truth, new policy, model modification, or fallback is introduced.
"""
import numpy as np


class RawOutputRecorder:
    """Transparent model proxy: keep original output tensors, return them unchanged."""
    def __init__(self, model):
        self.model = model
        self.latest = None

    def eval(self):
        self.model.eval()
        return self

    def template(self, image):
        return self.model.template(image)

    def track(self, image):
        self.latest = self.model.track(image)
        return self.latest


def decision_maps(boxes, class_probability, second_head, prior_size, scale_z, window,
                  *, w2=1., w3=1., penalty_k=.08, window_influence=.42, learning_rate=.30):
    """Same NumPy expression/order as pinned hift_tracker.py:133-164.

    second_head is either raw logits (B0) or sigmoid(logits) (B1-score), never GT.
    Negative candidate widths are observed, not repaired: original argmax semantics
    (including NaNs) are preserved. Caller validates the emitted tracker box.
    """
    boxes = np.asarray(boxes)
    p, second_head = np.asarray(class_probability), np.asarray(second_head)
    prior_size, window = np.asarray(prior_size), np.asarray(window)
    n = p.size
    if boxes.shape != (4, n) or second_head.shape != p.shape or window.shape != p.shape or prior_size.shape != (2,):
        raise ValueError('policy arrays have incompatible shapes')
    score = (p * w2 + second_head * w3) / 2
    def change(r):
        return np.maximum(r, 1. / (r + 1e-5))
    def sz(w, h):
        pad = (w + h) * .5
        return np.sqrt((w + pad) * (h + pad))
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        scale_penalty = change(sz(boxes[2], boxes[3]) / sz(prior_size[0] * scale_z, prior_size[1] * scale_z))
        aspect_penalty = change((prior_size[0] / (prior_size[1] + 1e-5)) /
                                (boxes[2] / (boxes[3] + 1e-5)))
        penalty = np.exp(-(aspect_penalty * scale_penalty - 1) * penalty_k)
        penalized = penalty * score
        selection = penalized * (1 - window_influence) + window * window_influence
        index = int(np.argmax(selection))
        lr = penalty[index] * score[index] * learning_rate
    return {'score': score, 'penalty': penalty, 'selection': selection,
            'selected_index': index, 'selected_score': float(score[index]),
            'interpolation_lr': float(lr), 'nonfinite_candidates': int((~np.isfinite(selection)).sum()),
            'nonpositive_candidate_sizes': int(((boxes[2] <= 0) | (boxes[3] <= 0)).sum())}


def observe_decision(tracker, recorder, prior_size, scale_z, cfg):
    """Post-timing observation, reconstruct both policies on one shared response/state."""
    raw = recorder.latest
    boxes = tracker.generate_anchor(raw['loc']).transpose()
    probability = tracker._convert_score(raw['cls1'])
    logits = raw['cls2'].detach().view(-1).cpu().numpy()
    sigmoid = raw['cls2'].detach().sigmoid().view(-1).cpu().numpy()
    kwargs = dict(w2=cfg.TRACK.w2, w3=cfg.TRACK.w3, penalty_k=cfg.TRACK.PENALTY_K,
                  window_influence=cfg.TRACK.WINDOW_INFLUENCE, learning_rate=cfg.TRACK.LR)
    original = decision_maps(boxes, probability, logits, prior_size, scale_z, tracker.window, **kwargs)
    corrected = decision_maps(boxes, probability, sigmoid, prior_size, scale_z, tracker.window, **kwargs)
    summary = {}
    for label, value in [('B0', original), ('B1-score', corrected)]:
        summary[label] = {key: value[key] for key in ('selected_index', 'selected_score', 'interpolation_lr',
                                                      'nonfinite_candidates', 'nonpositive_candidate_sizes')}
    summary['same_state_counterfactual_ranking_differs'] = original['selected_index'] != corrected['selected_index']
    summary['raw_logit_min'] = float(logits.min())
    summary['raw_logit_max'] = float(logits.max())
    grids = {'class_probability': probability.copy(), 'bce_logits': logits.copy(),
             'predicted_crop_boxes': boxes.copy(), 'penalty': original['penalty'].copy(),
             'B0_selection': original['selection'].copy(), 'B1_selection': corrected['selection'].copy()}
    return summary, grids
