"""Attacks on the search crop (pixel values 0-255, BGR, shape 1x3x287x287).

Threat model: the attacker can add a small perturbation to the image the tracker
sees each frame (digital, white-box: it knows the weights of `attack_net`).
eps is the L-infinity budget in pixel units: every pixel moves by at most eps out of 255.

Objective ("cooling"): push down HiFT's target score near the centre of the
search crop, where the target is expected to be, so the tracker jumps elsewhere.
This is the core idea of the Cooling-Shrinking Attack (Yan et al., CVPR 2020),
done with plain FGSM / PGD instead of a trained generator.
"""
import numpy as np
import torch

from .model import load_cfg

cfg = load_cfg()


def centre_weights(size, device):
    w = np.outer(np.hanning(size + 2)[1:-1], np.hanning(size + 2)[1:-1])
    w = w / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)[None]


class RandomSignNoise:
    """Control: random +-eps noise with the same budget. If this hurts as much as
    the attack, the 'attack' is just noise sensitivity."""

    name = "noise"

    def __init__(self, eps, seed=0):
        self.eps = float(eps)
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, x, scorer=None, defense=None):
        signs = torch.randint(0, 2, x.shape, generator=self.gen).float().mul_(2).sub_(1)
        return (x + self.eps * signs.to(x.device)).clamp_(0, 255)


class PGD:
    """steps=1, random_start=False, alpha=eps  ->  FGSM.
    adaptive=True: the attacker also differentiates through the defence
    (straight-through estimate for non-differentiable defences, i.e. BPDA)."""

    def __init__(self, eps, steps=10, alpha=None, random_start=True, adaptive=False, seed=0):
        self.eps, self.steps = float(eps), int(steps)
        self.alpha = float(alpha) if alpha else (self.eps if steps == 1 else 2.5 * self.eps / steps)
        self.random_start, self.adaptive = random_start and steps > 1, adaptive
        self.gen = torch.Generator().manual_seed(seed)
        self.name = "fgsm" if steps == 1 else f"pgd{steps}"
        self._w = None
        self.last_drop = []

    def objective(self, s):
        if self._w is None or self._w.shape[-1] != s.shape[-1]:
            self._w = centre_weights(s.shape[-1], s.device)
        return (self._w * s).sum()

    def __call__(self, x, scorer, defense=None):
        x0 = x.detach()
        through = None
        if self.adaptive and defense is not None:
            through = defense.straight_through
        delta = torch.zeros_like(x0)
        if self.random_start:
            delta = (torch.rand(x0.shape, generator=self.gen).to(x0.device) * 2 - 1) * self.eps
            delta = (x0 + delta).clamp(0, 255) - x0
        with torch.enable_grad():
            start = None
            for _ in range(self.steps):
                delta.requires_grad_(True)
                inp = x0 + delta
                if through is not None:
                    inp = through(inp)
                loss = self.objective(scorer(inp))
                if start is None:
                    start = float(loss.detach())
                (grad,) = torch.autograd.grad(loss, delta)
                with torch.no_grad():
                    delta = delta - self.alpha * grad.sign()      # minimise target score
                    delta = delta.clamp(-self.eps, self.eps)
                    delta = (x0 + delta).clamp(0, 255) - x0
        with torch.no_grad():
            final = float(self.objective(scorer(x0 + delta if through is None else through(x0 + delta))))
        self.last_drop.append((start, final))
        return (x0 + delta).detach()


def make_attack(name, eps, steps=10, adaptive=False, seed=0):
    if name in (None, "", "none") or float(eps) == 0:
        return None
    if name == "noise":
        return RandomSignNoise(eps, seed)
    if name == "fgsm":
        return PGD(eps, steps=1, adaptive=adaptive, seed=seed)
    if name == "pgd":
        return PGD(eps, steps=steps, adaptive=adaptive, seed=seed)
    raise SystemExit(f"unknown attack {name}")
