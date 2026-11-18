"""Night augmentation for fine-tuning HiFT (new file; the default Augmentation is not changed except np.float -> np.float64).

NightAugmentation = Augmentation plus
  * shift/scale that also moves extra boxes (needed to check that a distractor's template object lies outside the search window),
  * darken(): gamma + brightness scale, colour cast, Poisson-Gaussian noise that does not shrink with brightness, JPEG re-compression,
  * enhance(): gamma LUT as in pysot/utils/enhance.py (gamma < 1 brightens), same value for template and search.
All image functions work on uint8 BGR crops (template 127, search 287), i.e. after the crop, like the test-time enhancement.
"""
import cv2
import numpy as np

from pysot.datasets.augmentation import Augmentation
from pysot.utils.bbox import Corner, Center, center2corner, corner2center


class DarkenSpec(object):
    """ranges of one darkening profile (values in [0, 1] intensity units, applied to sRGB code values / 255)"""

    def __init__(self, p, gamma, scale, read=(0.001, 0.006), shot=(0.0, 2e-4), cast=0.06, jpeg=(30, 80)):
        self.p, self.gamma, self.scale = p, gamma, scale
        self.read, self.shot, self.cast, self.jpeg = read, shot, cast, jpeg


def darken(img, spec, rng):
    """img uint8 BGR -> uint8 BGR. x -> x**gamma * scale, colour cast, noise var = shot*x + read**2, JPEG."""
    x = img.astype(np.float32) / 255.0
    gamma = rng.uniform(*spec.gamma)
    scale = rng.uniform(*spec.scale)
    x = np.power(x, gamma) * scale
    x = x * (1.0 + rng.uniform(-spec.cast, spec.cast, size=(1, 1, 3))).astype(np.float32)
    read = rng.uniform(*spec.read)
    shot = rng.uniform(*spec.shot)
    sigma = np.sqrt(shot * np.maximum(x, 0.0) + read ** 2).astype(np.float32)   # does not go to zero at low brightness
    x = x + rng.standard_normal(x.shape).astype(np.float32) * sigma
    out = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    q = int(rng.integers(spec.jpeg[0], spec.jpeg[1] + 1))
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), dict(gamma=round(float(gamma), 2), scale=round(float(scale), 2), read=round(float(read), 4), jpeg=q)


_LUTS = {}


def enhance_gamma(img, gamma):
    """same lookup table as pysot.utils.enhance.make_enhancer('gamma', gamma)"""
    key = round(float(gamma), 3)
    if key not in _LUTS:
        _LUTS[key] = (255.0 * (np.arange(256) / 255.0) ** key).round().astype(np.uint8)
    return cv2.LUT(img, _LUTS[key])


class NightAugmentation(Augmentation):
    def _shift_scale_multi(self, image, bbox, crop_bbox, size, extra):
        """as Augmentation._shift_scale_aug, additionally transforms the Corner boxes in `extra` the same way"""
        im_h, im_w = image.shape[:2]
        c = corner2center(crop_bbox)
        scale_x = scale_y = 1.0
        if self.scale:
            scale_x = 1.0 + Augmentation.random() * self.scale
            scale_y = 1.0 + Augmentation.random() * self.scale
            scale_x = min(scale_x, float(im_w) / c.w)
            scale_y = min(scale_y, float(im_h) / c.h)
            c = Center(c.x, c.y, c.w * scale_x, c.h * scale_y)
        crop_bbox = center2corner(c)
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift
            x1, y1, x2, y2 = crop_bbox
            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))
            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)
        x1, y1 = crop_bbox.x1, crop_bbox.y1

        def tf(b):
            b = Corner(b.x1 - x1, b.y1 - y1, b.x2 - x1, b.y2 - y1)
            return Corner(b.x1 / scale_x, b.y1 / scale_y, b.x2 / scale_x, b.y2 / scale_y)
        out_img = self._crop_roi(image, crop_bbox, size)
        return out_img, tf(bbox), [tf(b) for b in extra]

    def crop(self, image, bbox, size, extra=()):
        """shift/scale/colour/blur like Augmentation.__call__ (flip is off in the config) -> (uint8 crop, bbox, extra boxes)"""
        shape = image.shape
        crop_bbox = center2corner(Center(shape[0] // 2, shape[1] // 2, size - 1, size - 1))
        image, bbox, extra = self._shift_scale_multi(image, bbox, crop_bbox, size, extra)
        if self.color > np.random.random():
            image = self._color_aug(image)
        if self.blur > np.random.random():
            image = self._blur_aug(image)
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return np.clip(image, 0, 255).astype(np.uint8), bbox, extra
