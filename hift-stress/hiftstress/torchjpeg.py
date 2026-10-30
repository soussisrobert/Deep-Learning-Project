"""JPEG compression written in PyTorch (runs on GPU or CPU, optionally differentiable).

Two uses in this project:
  1. A GPU-side JPEG defence ("tjpeg<q>"): the same compression as libjpeg's baseline mode
     (YCbCr, 4:2:0 chroma subsampling, 8x8 DCT, libjpeg quantisation tables scaled by quality),
     without the CPU round-trip of OpenCV. It is not bit-identical to libjpeg (upsampling and
     rounding details differ), so it is evaluated as its own defence.
  2. A differentiable surrogate for adaptive attacks on the OpenCV JPEG defence. The rounding
     step is replaced by Shin & Song's (2017) cubic approximation
         round(x) + (x - round(x))**3
     which has a useful gradient, unlike round(). This is the standard way to attack JPEG
     defences adaptively (see also Athalye et al., ICML 2018, BPDA).

Input / output: float tensors 1x3xHxW (any batch), BGR, pixel values 0-255.
"""
import math

import torch
import torch.nn.functional as F

# libjpeg standard tables (ITU T.81 Annex K)
_QY = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
       14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
       49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_QC = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
       47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def quant_table(base, quality):
    """libjpeg's jpeg_quality_scaling + clamping to 1..255."""
    q = max(1, min(100, int(quality)))
    s = 5000 / q if q < 50 else 200 - 2 * q
    t = torch.tensor([math.floor((b * s + 50) / 100) for b in base], dtype=torch.float32)
    return t.clamp(1, 255).view(8, 8)


def _dct_matrix(device, dtype):
    n = torch.arange(8, dtype=dtype, device=device)
    k = n.view(-1, 1)
    m = torch.cos((2 * n + 1) * k * math.pi / 16) * math.sqrt(2 / 8)
    m[0] = m[0] / math.sqrt(2)
    return m  # orthonormal: X = M x M^T


def _round(x, mode):
    if mode == "hard":
        return torch.round(x)
    r = torch.round(x).detach()
    if mode == "approx":          # Shin & Song 2017
        return r + (x - r) ** 3
    if mode == "ste":             # straight-through
        return x + (r - x).detach()
    raise ValueError(mode)


def _blocks(x):
    b, c, h, w = x.shape
    return x.view(b, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)  # b c H W 8 8


def _unblocks(x):
    b, c, hb, wb, _, _ = x.shape
    return x.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hb * 8, wb * 8)


def _code(ch, q, M, mode):
    """DCT -> quantise -> dequantise -> IDCT on one or more channels (b c h w, h,w multiples of 8)."""
    B = _blocks(ch - 128.0)
    D = M @ B @ M.T
    D = _round(D / q, mode) * q
    return _unblocks(M.T @ D @ M) + 128.0


def jpeg(x, quality=75, mode="hard", subsample=True):
    """x: Bx3xHxW BGR in [0,255]. Returns the JPEG-compressed image, same shape (float)."""
    dt, dev = x.dtype, x.device
    x = x.float()
    B_, G_, R_ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * R_ + 0.587 * G_ + 0.114 * B_
    cb = -0.168736 * R_ - 0.331264 * G_ + 0.5 * B_ + 128
    cr = 0.5 * R_ - 0.418688 * G_ - 0.081312 * B_ + 128
    h, w = x.shape[-2:]
    mult = 16 if subsample else 8
    ph, pw = (-h) % mult, (-w) % mult
    pad = lambda t: F.pad(t, (0, pw, 0, ph), mode="replicate") if (ph or pw) else t  # noqa: E731
    y, cb, cr = pad(y), pad(cb), pad(cr)
    M = _dct_matrix(dev, torch.float32)
    qy, qc = quant_table(_QY, quality).to(dev), quant_table(_QC, quality).to(dev)
    y = _code(y, qy, M, mode)
    c = torch.cat([cb, cr], 1)
    if subsample:
        c = F.avg_pool2d(c, 2)
        c = _code(c, qc, M, mode)
        c = F.interpolate(c, scale_factor=2, mode="bilinear", align_corners=False)
    else:
        c = _code(c, qc, M, mode)
    cb, cr = c[:, 0:1], c[:, 1:2]
    y, cb, cr = y[..., :h, :w], cb[..., :h, :w], cr[..., :h, :w]
    R = y + 1.402 * (cr - 128)
    G = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    Bc = y + 1.772 * (cb - 128)
    out = torch.cat([Bc, G, R], 1)
    if mode == "hard":
        out = out.round()
    return out.clamp(0, 255).to(dt)
