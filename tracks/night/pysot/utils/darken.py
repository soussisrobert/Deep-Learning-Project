"""Synthetic night-time darkening of daytime frames, parameters calibrated on NAT2021 (see experiments_log.md).

Per sequence (DarkenParams): a target mean gray drawn from a log-normal fitted to NAT2021 (quartiles 14.8 / 21.6 / 34.0,
clipped to the observed 3.6-68.7), a tone curve gamma_d in [1.0, 1.5], a noise-strength multiplier and a JPEG quality.
Per frame (darken): sRGB -> linear light, tone curve, exposure scale (solved so the sequence reaches its target gray),
back to sRGB code values, signal-dependent Gaussian noise, 8-bit, JPEG re-encode.

Noise: the first attempt, a Poisson-Gaussian fit var_lin = a*x + b in linear light, did not work. The fit to the NAT2021
measurements had a negative intercept and predicted almost no noise in the dark range, where NAT frames are. So the noise
standard deviation is taken straight from the measured NAT curve (sigma in 8-bit code units against brightness code, block
residual, median block, tools/measure_nat_noise.py 6 0.5), which is heteroscedastic like Poisson-Gaussian noise is. The curve
was measured AFTER NAT's own JPEG compression, so `noise_k` scales it until the darkened-then-JPEG frames match NAT after
compression; it is set by tools/validate_darken.py.
"""
import numpy as np
import cv2

# NAT2021 block-noise curve, median block (q=0.5): brightness code -> sigma in code units
NAT_MU = np.arange(4, 160, 8, dtype=np.float32)
NAT_SIGMA = np.array([0.0, 0.08, 0.08, 0.10, 0.12, 0.20, 0.30, 0.36, 0.42, 0.49, 0.55, 0.64, 0.74, 0.80, 0.84, 0.90,
                      0.97, 1.02, 0.97, 1.06], dtype=np.float32)
NAT_LOGSTD_SIGMA = 0.95    # spread of per-sequence noise sigma around the global curve (q=0.5 run)
GRAY_MED, GRAY_LOGSTD, GRAY_MIN, GRAY_MAX = 21.6, 0.62, 3.6, 68.7

_LUT_TO_LIN = None


def _to_lin_lut():
    global _LUT_TO_LIN
    if _LUT_TO_LIN is None:
        c = np.arange(256) / 255.0
        _LUT_TO_LIN = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return _LUT_TO_LIN


def to_lin(img_u8):
    return _to_lin_lut()[img_u8]


def to_srgb_code(x):
    """linear light (float) -> sRGB code values as float 0..255 (not rounded)"""
    x = np.clip(x, 0.0, 1.0)
    return (np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0).astype(np.float32)


def gray_mean(img_u8):
    return float(cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).mean())


class DarkenParams:
    """Random, per-sequence darkening strength. `ref_frames` are a few frames of the sequence (uint8 BGR) used to solve the
    exposure scale so that the darkened reference frames reach the target mean gray."""

    def __init__(self, ref_frames, rng, noise_k=1.0):
        g = float(np.exp(np.log(GRAY_MED) + GRAY_LOGSTD * rng.standard_normal()))
        self.target_gray = float(np.clip(g, GRAY_MIN, GRAY_MAX))
        self.gamma_d = float(rng.uniform(1.0, 1.5))
        z = float(np.clip(rng.standard_normal(), -2, 2))
        self.noise_mult = float(np.clip(np.exp(NAT_LOGSTD_SIGMA * z), 0.4, 6.0)) * noise_k
        self.jpeg_q = int(rng.integers(85, 96))
        refs = [to_lin(f) for f in ref_frames]
        lo, hi = 1e-4, 4.0
        for _ in range(40):  # bisection on exposure scale (mean gray is monotone in it)
            mid = np.sqrt(lo * hi)
            g_mid = np.mean([gray_mean(np.rint(to_srgb_code(mid * np.power(r, self.gamma_d))).astype(np.uint8)) for r in refs])
            lo, hi = (mid, hi) if g_mid < self.target_gray else (lo, mid)
        self.exposure = float(np.sqrt(lo * hi))


def darken(img_u8, p, rng):
    code = to_srgb_code(p.exposure * np.power(to_lin(img_u8), p.gamma_d))
    luma = code.mean(axis=2, keepdims=True)
    sigma = np.interp(luma, NAT_MU, NAT_SIGMA).astype(np.float32) * p.noise_mult
    # same noise field on the three channels, so the gray residual has exactly this sigma (chroma noise not modelled)
    code = code + rng.standard_normal(luma.shape).astype(np.float32) * sigma
    out = np.clip(np.rint(code), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
