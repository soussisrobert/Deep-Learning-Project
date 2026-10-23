"""Diagnostic geometry only: do not silently alter original tracker heuristics."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CropGeometry:
    """HiFT integer crop bounds and OpenCV half-pixel resize mapping.

    Coordinates are zero-based pixel centers; bbox widths are continuous extents.
    origin is before padding. Out-of-image support is padding, never genuine detail.
    """
    origin_x: int
    origin_y: int
    source_size: int
    model_size: int

    @classmethod
    def from_tracker(cls, center, source_size, model_size):
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError('center must contain two finite coordinates')
        if isinstance(source_size, bool) or int(source_size) != source_size or source_size < 1:
            raise ValueError('source_size must be a positive integer after tracker rounding')
        if int(model_size) != model_size or model_size < 1:
            raise ValueError('model_size must be a positive integer')
        c = (source_size + 1) / 2
        return cls(math.floor(center[0] - c + .5), math.floor(center[1] - c + .5), int(source_size), int(model_size))

    @property
    def scale(self):
        return self.model_size / self.source_size

    def source_to_model(self, points):
        points = np.asarray(points, dtype=float)
        return (points - [self.origin_x, self.origin_y] + .5) * self.scale - .5

    def model_to_source(self, points):
        points = np.asarray(points, dtype=float)
        return (points + .5) / self.scale - .5 + [self.origin_x, self.origin_y]

    def support(self, bbox, image_shape):
        """bbox may be GT in diagnostics only; inference must use prior prediction.

        Pixel-support estimates are geometric proxies, not measured entropy or recoverable detail.
        """
        x, y, w, h = np.asarray(bbox, dtype=float)
        if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
            raise ValueError('bbox must have finite positive extent')
        ih, iw = image_shape[:2]
        if min(ih, iw) <= 0:
            raise ValueError('invalid image shape')
        crop_x, crop_y = self.origin_x, self.origin_y
        # Half-open edge convention for area/containment diagnostics (not pixel-center transform).
        intersect_w = max(0., min(x + w, crop_x + self.source_size, iw) - max(x, crop_x, 0))
        intersect_h = max(0., min(y + h, crop_y + self.source_size, ih) - max(y, crop_y, 0))
        return {'source_width_px': float(w), 'source_height_px': float(h),
                'source_area_px2': float(w * h), 'resized_width_px': float(w * self.scale),
                'resized_height_px': float(h * self.scale), 'aspect_ratio': float(w / h),
                'crop_source_size_px': self.source_size, 'resize_scale': self.scale,
                'source_pixels_per_model_pixel': 1 / self.scale,
                'genuine_visible_support_px2': float(intersect_w * intersect_h),
                'containment_fraction': float(intersect_w * intersect_h / (w * h))}


def stable_sigmoid(logits):
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('logits must be finite')
    out = np.empty_like(values)
    pos = values >= 0
    out[pos] = 1 / (1 + np.exp(-values[pos]))
    exp_values = np.exp(values[~pos])
    out[~pos] = exp_values / (1 + exp_values)
    return out


def blend_scores(class_probability, bce_logits, w2=1., w3=1., corrected=False):
    p = np.asarray(class_probability, dtype=float)
    z = np.asarray(bce_logits, dtype=float)
    if p.shape != z.shape or not np.isfinite(p).all() or not np.isfinite(z).all():
        raise ValueError('finite matching score shapes required')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('classification probabilities outside [0,1]')
    return (w2 * p + w3 * (stable_sigmoid(z) if corrected else z)) / 2
