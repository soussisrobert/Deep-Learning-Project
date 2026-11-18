"""Natural image corruptions (no attacker), applied to every crop the tracker sees.

Severity levels follow ImageNet-C (Hendrycks & Dietterich, ICLR 2019), the protocol UAV-C
(arXiv:2403.11424) adapts to aerial tracking; UAV-C reports HiFT under Gaussian noise and blur.
Differences from UAV-C: we corrupt the 127/287 crops the tracker uses (template and every
search crop) rather than the full frames, and use PyTorch re-implementations.

Spec: "<name>:<severity 1-5>", e.g. "gauss_noise:3". Input/output 1x3xHxW BGR float 0-255.
"""
import math

import torch
import torch.nn.functional as F

LEVELS = {
    "gauss_noise": [0.08, 0.12, 0.18, 0.26, 0.38],      # sigma on the 0-1 scale (ImageNet-C)
    "gauss_blur": [1, 2, 3, 4, 6],                      # sigma in pixels (ImageNet-C)
    "motion_blur": [5, 9, 13, 17, 21],                  # kernel length in pixels, random angle per frame
    "low_light": [0.8, 0.6, 0.45, 0.3, 0.2],            # brightness factor (darker scenes)
    "contrast": [0.4, 0.3, 0.2, 0.1, 0.05],             # ImageNet-C contrast factors
    # synthetic night: darker AND noisier (camera gain), (brightness, noise sigma on 0-1 scale); not real night footage
    "night": [(0.6, 0.01), (0.45, 0.02), (0.35, 0.03), (0.25, 0.04), (0.2, 0.05)],
}


def _conv(x, k):
    c, r = x.shape[1], k.shape[-1] // 2
    x = F.pad(x, (r, r, r, r), mode="replicate")
    return F.conv2d(x, k.to(x.device, x.dtype).expand(c, 1, *k.shape[-2:]), groups=c)


def _gauss_kernel(sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def _motion_kernel(length, angle):
    k = torch.zeros(length, length)
    c = length // 2
    for i in range(length):
        t = i - c
        x = int(round(c + t * math.cos(angle)))
        y = int(round(c + t * math.sin(angle)))
        k[y, x] = 1
    return (k / k.sum())[None, None]


class Corruption:
    def __init__(self, spec, seed=0):
        name, _, sev = spec.partition(":")
        if name not in LEVELS:
            raise SystemExit(f"unknown corruption {name}; choose from {sorted(LEVELS)}")
        self.kind, self.sev = name, int(sev or 3)
        self.level = LEVELS[name][self.sev - 1]
        self.name = f"{name}{self.sev}"
        self.gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def __call__(self, x):
        x = x.float()
        if self.kind == "gauss_noise":
            n = torch.randn(x.shape, generator=self.gen).to(x.device)
            x = x + n * self.level * 255
        elif self.kind == "gauss_blur":
            x = _conv(x, _gauss_kernel(self.level))
        elif self.kind == "motion_blur":
            ang = float(torch.rand(1, generator=self.gen)) * math.pi
            x = _conv(x, _motion_kernel(self.level, ang))
        elif self.kind == "low_light":
            x = x * self.level
        elif self.kind == "night":
            b, sig = self.level
            x = x * b + torch.randn(x.shape, generator=self.gen).to(x.device) * sig * 255
        elif self.kind == "contrast":
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - m) * self.level + m
        return x.clamp(0, 255)


def make_corruption(spec, seed=0):
    return None if spec in (None, "", "none") else Corruption(spec, seed)
