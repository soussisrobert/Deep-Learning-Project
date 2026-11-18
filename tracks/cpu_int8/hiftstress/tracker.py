"""HiFT tracker with hooks for an attack and a defence on every search crop.

Per frame, the official HiFT tracker crops a 287x287 search region around the last
position and feeds it to the network. This wrapper lets us
    1. perturb that crop (adversarial attack, or random-noise control),
    2. clean it again (input-transform defence),
    3. run any model variant (fp32, fp16, int8, pruned),
while keeping the official pre/post-processing untouched.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import load_cfg

cfg = load_cfg()
cfg.CUDA = False  # we move tensors ourselves; works on cpu, mps and cuda

from pysot.tracker.hift_tracker import HiFTTracker  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def score_map(net, x):
    """Differentiable version of the score HiFT uses to pick the target (1 x 11 x 11)."""
    xf = net.backbone(x)
    _, cls1, cls2 = net.grader(xf, net.zf)
    p = F.softmax(cls1, dim=1)[:, 1]
    return (p * cfg.TRACK.w2 + cls2[:, 0] * cfg.TRACK.w3) / 2


@torch.no_grad()
def confidence_stats(out):
    """Peak and peak-to-sidelobe ratio (PSR) of the fused score map (same fusion HiFT uses to pick the box).
    Used only for failure / attack awareness; does not change tracking."""
    p = F.softmax(out["cls1"], dim=1)[0, 1]
    m = ((p * cfg.TRACK.w2 + out["cls2"][0, 0] * cfg.TRACK.w3) / 2).cpu()
    peak = float(m.max())
    i, j = divmod(int(m.argmax()), m.shape[1])
    mask = torch.ones_like(m, dtype=torch.bool)
    mask[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = False
    side = m[mask]
    return peak, float((peak - side.mean()) / (side.std() + 1e-6))


class TrackingModel:
    """Stands in for ModelBuilder inside HiFTTracker."""

    def __init__(self, net, device, dtype=torch.float32, attack=None, defense=None, attack_net=None, corruption=None):
        self.net, self.device, self.dtype = net, device, dtype
        self.corruption = corruption  # natural corruption of every crop (part of the scene, not timed)
        self.attack, self.defense = attack, defense
        self.attack_net = attack_net if attack_net is not None else net
        self.stats = {"attack_s": 0.0, "defense_s": 0.0, "model_s": 0.0, "frames": 0}
        self.frame_conf = []   # per tracked frame: (peak, psr) of the fused 11x11 score map HiFT selects from

    def eval(self):
        return self

    def template(self, z):
        z = z.to(self.device)
        if self.corruption is not None:
            z = self.corruption(z)
        with torch.no_grad():
            self.net.template(z.to(self.dtype))
            if self.attack is not None and self.attack_net is not self.net:
                self.attack_net.template(z.to(next(self.attack_net.parameters()).device))

    def track(self, x):
        x = x.to(self.device)
        if self.corruption is not None:
            x = self.corruption(x)
        if self.attack is not None:
            t0 = time.perf_counter()
            adev = next(self.attack_net.parameters()).device
            if adev != x.device:   # e.g. attacker computes on the GPU against an int8 model on the CPU
                x = self.attack(x.to(adev), lambda inp: score_map(self.attack_net, inp), self.defense).to(self.device)
                _sync(adev)
            else:
                x = self.attack(x, lambda inp: score_map(self.attack_net, inp), self.defense)
            _sync(self.device)
            self.stats["attack_s"] += time.perf_counter() - t0
        if self.defense is not None:
            t0 = time.perf_counter()
            with torch.no_grad():
                x = self.defense(x)
            _sync(self.device)
            self.stats["defense_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.net.track(x.to(self.dtype))
        _sync(self.device)
        self.stats["model_s"] += time.perf_counter() - t0
        self.stats["frames"] += 1
        out = {k: v.float() for k, v in out.items()}
        self.frame_conf.append(confidence_stats(out))
        return out


class HookedTracker(HiFTTracker):
    def __init__(self, model, device):
        super().__init__(model)
        self.device = device

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        return super().get_subwindow(im, pos, model_sz, original_sz, avg_chans).to(self.device)


def run_sequence(seq, model, device):
    """One-pass evaluation: initialise on frame 0 with ground truth, never re-initialise."""
    tracker = HookedTracker(model, device)
    preds, scores, times = [], [], []
    for i in range(len(seq)):
        img = seq.image(i)
        t0 = time.perf_counter()
        if i == 0:
            x, y, w, h = seq.gt[0]
            tracker.init(img, [x, y, w, h])
            preds.append([x, y, w, h])
            scores.append(float("nan"))
        else:
            out = tracker.track(img)
            preds.append([float(v) for v in out["bbox"]])
            scores.append(float(out["best_score"]))
        times.append(time.perf_counter() - t0)
    return np.array(preds), np.array(scores), np.array(times)
