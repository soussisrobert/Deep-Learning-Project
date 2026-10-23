"""Explicit B2 tiny-positive and identity-checked pair policies.

Derived from pinned pysot/datasets/anchortarget.py:20-111. Ordinary finite
nonempty-support outputs (including RNG consumption) retain upstream behavior.
The minimum-support fallback is a disclosed heuristic, not author intent.
"""
import numpy as np

from .pair_targets import pair_targets


class NegativeIdentityError(ValueError):
    """A requested negative pair has missing or colliding object identities."""


def validate_negative_identity(template_identity, search_identity):
    """Require explicit distinct (dataset, video, track) string keys.

    Different frames of the same object are NOT distinct identities. This only
    checks supplied metadata: cross-dataset duplicates require a provenance map.
    No resampling, relabeling, or image-based identity guess is performed.
    """
    for key in (template_identity, search_identity):
        if not isinstance(key, tuple) or len(key) != 3 or not all(
            isinstance(part, str) and part.strip() for part in key
        ):
            raise NegativeIdentityError("negative pairs require (dataset, video, track) identities")
    if template_identity == search_identity:
        raise NegativeIdentityError("same-object pair must not receive negative supervision")


def robust_pair_targets(anchor_target, bbox, size, *, is_negative,
                        template_identity=None, search_identity=None):
    """Identity-checked pair_targets API; use CorrectedAnchorTarget for positives.

    Missing/colliding negative metadata fails closed, rather than silently
    changing pair prevalence through automatic resampling. Positive pairs need no
    identity arguments; target construction remains the anchor's responsibility.
    """
    if not isinstance(is_negative, (bool, np.bool_)):
        raise ValueError("is_negative must be a boolean")
    if is_negative:
        validate_negative_identity(template_identity, search_identity)
    return pair_targets(anchor_target, bbox, size, is_negative=is_negative)


class CorrectedAnchorTarget:
    """Pinned target recipe plus empty-support guards and stable overflow fallback.

    Retains grid origin=63, stride=16, clipping, radial heuristic, ignore bands,
    random selection/counts, and encoding scale search_size//2. Does not claim
    these retained choices are calibrated or optimally aligned with crop pixels.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        selected = np.arange(num)
        np.random.shuffle(selected)
        selected = selected[:keep_num]
        return tuple(p[selected] for p in position), keep_num

    def get(self, bbox, size):
        cfg = self.cfg
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        if size != cfg.TRAIN.OUTPUTFEATURE_SIZE:
            raise ValueError("size must equal configured OUTPUTFEATURE_SIZE")
        search_size = cfg.TRAIN.SEARCH_SIZE
        label_range = cfg.TRAIN.LABEL_RANGE
        if (isinstance(search_size, bool) or not isinstance(search_size, (int, np.integer))
                or not np.isfinite([search_size, label_range]).all() or search_size < 2 or label_range <= 0
                or label_range > np.sqrt(np.finfo(float).max) / (4 * size)):
            raise ValueError("finite integer search size and numerically bounded positive LABEL_RANGE required")
        target = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            extent = target[2:] - target[:2]
        if (not np.isfinite(target).all() or not np.isfinite(extent).all() or np.any(extent <= 0)
                or np.any(np.abs(target) > np.finfo(np.float32).max / 4)):
            raise ValueError("bbox must have finite positive extents representable by the label encoding")
        if target[2] <= 0 or target[3] <= 0 or target[0] >= search_size or target[1] >= search_size:
            raise ValueError("positive bbox has no overlap with the search crop")

        labelcls1 = np.zeros((1, size, size)) - 1
        pre = (16 * np.linspace(0, size - 1, size) + 63).reshape(-1, 1) - search_size // 2
        pr = np.zeros((size**2, 2))
        pr[:, 0] = np.maximum(0, np.tile(pre, size).T.reshape(-1) + search_size // 2)
        pr[:, 1] = np.maximum(0, np.tile(pre, size).reshape(-1) + search_size // 2)
        labelxff = np.zeros((4, size, size), dtype=np.float32)
        labelcls2 = np.zeros((1, size, size))
        weightxff = np.zeros((1, size, size))
        index = np.int32(np.minimum(size - 1, np.maximum(0, (target - 63) / 16)))
        w = int(index[2] - index[0] + 1)
        h = int(index[3] - index[1] + 1)
        for ii in np.arange(size):
            for jj in np.arange(size):
                weightxff[0, ii, jj] = (
                    ((ii - (index[1] + index[3]) / 2) * label_range)**2
                    + ((jj - (index[0] + index[2]) / 2) * label_range)**2
                )
        cutoff = ((w // 2 + h // 2) * label_range / 1.5)**2
        support = weightxff < cutoff
        se = weightxff[support]
        if se.size:
            weightxff[support] = 1 - (se - se.min()) / (se.max() - se.min() + 1e-4)
            weightxff[weightxff > cutoff] = 0
        else:
            # Collapsed grid cell: retain a positive observation at that cell.
            weightxff.fill(0)
            weightxff[0, (index[1] + index[3]) // 2, (index[0] + index[2]) // 2] = 1
        pos = np.where(weightxff[0] > 0.8)
        pos = self.select(pos, len(pos[0]) // 4)
        weightxff[:, pos[0][0], pos[0][1]] = 1.5

        for ii in np.arange(size):
            for jj in np.arange(size):
                labelcls2[0, ii, jj] = (
                    ((ii - (index[1] + index[3]) / 2) * label_range)**2
                    + ((jj - (index[0] + index[2]) / 2) * label_range)**2
                )
        cutoff = ((w // 2 + h // 2) * label_range / 1.2)**2
        support = labelcls2 < cutoff
        see = labelcls2[support]
        if see.size:
            labelcls2[support] = 1 - (see - see.min()) / (see.max() - see.min() + 1e-4)
            weightcls3 = np.zeros((1, size, size))
            # Deliberately retain upstream's post-normalization mask semantics.
            weightcls3[labelcls2 < cutoff] = 1
            labelcls2 = labelcls2 * weightcls3
        else:
            labelcls2.fill(0)
            labelcls2[0, (index[1] + index[3]) // 2, (index[0] + index[2]) // 2] = 1

        labelxff[0] = (pr[:, 0] - target[0]).reshape(size, size)
        labelxff[1] = (target[2] - pr[:, 0]).reshape(size, size)
        labelxff[2] = (pr[:, 1] - target[1]).reshape(size, size)
        labelxff[3] = (target[3] - pr[:, 1]).reshape(size, size)
        scaled = labelxff / (search_size // 2)
        with np.errstate(over="ignore", invalid="ignore"):
            encoded = (np.exp(scaled) - np.exp(-scaled)) / (np.exp(scaled) + np.exp(-scaled))
        invalid = ~np.isfinite(encoded)
        if invalid.any():
            encoded[invalid] = np.tanh(scaled[invalid])
        labelxff = encoded
        labelcls1[0, index[1] - h//4:index[3] + 1 + h//4,
                  index[0] - w//4:index[2] + 1 + w//4] = -2
        labelcls1[0, index[1] + h//4:index[3] + 1 - h//4,
                  index[0] + w//4:index[2] + 1 - w//4] = 1
        neg2 = self.select(np.where(labelcls1[0] == -1), int(np.count_nonzero(labelcls1 == 1) * 2.5))
        labelcls1[:, neg2[0][0], neg2[0][1]] = 0
        return labelcls1, labelxff, labelcls2, weightxff
