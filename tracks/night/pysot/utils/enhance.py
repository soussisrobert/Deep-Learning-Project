import cv2
import numpy as np

MODES = ('none', 'gamma', 'clahe', 'gamma_clahe')
DENOISERS = ('none', 'gauss3', 'gauss5', 'bilateral', 'nlm')


def _denoiser(name):
    """Return f(uint8 BGR patch) -> uint8 BGR patch, or None for 'none'."""
    if name in (None, '', 'none'):
        return None
    assert name in DENOISERS, 'unknown denoiser {}'.format(name)
    if name == 'gauss3':
        return lambda p: cv2.GaussianBlur(p, (3, 3), 0)
    if name == 'gauss5':
        return lambda p: cv2.GaussianBlur(p, (5, 5), 0)
    if name == 'bilateral':
        # d=5 neighbourhood, intensity sigma 20 (dark noise is a few levels), spatial sigma 5
        return lambda p: cv2.bilateralFilter(p, 5, 20, 5)
    # non-local means: h=3 luminance and colour strength, 7x7 template window, 21x21 search window
    return lambda p: cv2.fastNlMeansDenoisingColored(p, None, 3, 3, 7, 21)


def make_enhancer(mode, gamma=0.4, clip=3.0, denoise='none'):
    """Return f(uint8 BGR patch) -> uint8 BGR patch, or None if nothing is to be done.

    denoise: optional smoothing applied FIRST, before any brightening (brightening amplifies noise)
    gamma: lookup-table brightness curve, out = 255 * (in / 255) ** gamma (gamma < 1 brightens)
    clahe: contrast limited adaptive histogram equalisation on the LAB lightness channel
    gamma_clahe: gamma first, then clahe
    """
    mode = 'none' if mode in (None, '') else mode
    assert mode in MODES, 'unknown enhance mode {}'.format(mode)
    dn = _denoiser(denoise)
    if mode == 'none' and dn is None:
        return None
    lut = (255.0 * (np.arange(256) / 255.0) ** gamma).round().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))

    def enhance(patch):
        if dn is not None:
            patch = dn(patch)
        if mode in ('gamma', 'gamma_clahe'):
            patch = cv2.LUT(patch, lut)
        if mode in ('clahe', 'gamma_clahe'):
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            patch = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return patch

    return enhance
