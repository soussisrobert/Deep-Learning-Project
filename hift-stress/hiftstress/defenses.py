"""Input-transform defences applied to the search crop before the network sees it.

None of these needs retraining, which suits a group with little compute. Each one
also costs some clean accuracy and some time per frame; both must be reported.
References: Guo et al., ICLR 2018 (input transformations: JPEG, bit-depth, ...);
Xu et al., NDSS 2018 (feature squeezing: bit-depth, median); Athalye et al.,
ICML 2018 (why defences must be tested against adaptive attacks / BPDA).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class Defense:
    differentiable = False
    name = "base"

    def __call__(self, x):
        raise NotImplementedError

    def straight_through(self, x):
        """Used by adaptive attacks: exact gradient if differentiable, else identity (BPDA)."""
        if self.differentiable:
            return self(x)
        return x + (self(x.detach()) - x).detach()


class GaussianBlur(Defense):
    differentiable = True

    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)
        self.name = f"blur{sigma:g}"
        r = max(1, int(round(3 * self.sigma)))
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        k = torch.exp(-t ** 2 / (2 * self.sigma ** 2))
        self.k = k / k.sum()

    def __call__(self, x):
        k = self.k.to(x.device, x.dtype)
        c, r = x.shape[1], (len(k) - 1) // 2
        x = F.pad(x, (r, r, r, r), mode="replicate")
        x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


class Median(Defense):
    def __init__(self, size=3):
        self.size = int(size)
        self.name = f"median{size}"

    def __call__(self, x):
        r = self.size // 2
        xp = F.pad(x, (r, r, r, r), mode="replicate")
        patches = xp.unfold(2, self.size, 1).unfold(3, self.size, 1)
        return patches.reshape(*patches.shape[:4], -1).median(dim=-1).values


class BitDepth(Defense):
    def __init__(self, bits=5):
        self.bits = int(bits)
        self.name = f"bits{bits}"

    def __call__(self, x):
        step = 256 / (2 ** self.bits)
        return (torch.floor(x / step) * step + step / 2).clamp(0, 255)


class JPEG(Defense):
    """libjpeg (OpenCV) on the CPU: the deployed defence. Not differentiable; adaptive attacks use
    either BPDA (identity gradient) or the differentiable PyTorch surrogate (see surrogate())."""

    def __init__(self, quality=75):
        self.quality = int(quality)
        self.name = f"jpeg{quality}"

    def surrogate(self, x, quality=None):
        """Differentiable stand-in with the same quality (Shin & Song 2017 rounding)."""
        from .torchjpeg import jpeg
        return jpeg(x, self.quality if quality is None else quality, mode="approx")

    def __call__(self, x):
        img = x[0].detach().permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy()
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return torch.from_numpy(dec).permute(2, 0, 1)[None].to(x.device, x.dtype)


class TorchJPEG(JPEG):
    """The same JPEG written in PyTorch: runs wherever the crop already is (GPU), no CPU round-trip.
    Matches OpenCV's output to about 46 dB PSNR (under one grey level), but is evaluated separately."""

    def __init__(self, quality=75):
        super().__init__(quality)
        self.name = f"tjpeg{quality}"

    def __call__(self, x):
        from .torchjpeg import jpeg
        return jpeg(x, self.quality, mode="hard")


def make_defense(spec):
    """'none', 'blur1', 'blur1.5', 'median3', 'bits5', 'jpeg75', 'tjpeg90' ..."""
    if spec in (None, "", "none"):
        return None
    for prefix, cls in (("blur", GaussianBlur), ("median", Median), ("bits", BitDepth), ("tjpeg", TorchJPEG),
                        ("jpeg", JPEG)):
        if spec.startswith(prefix):
            return cls(float(spec[len(prefix):]) if prefix == "blur" else int(spec[len(prefix):]))
    raise SystemExit(f"unknown defence {spec}")
