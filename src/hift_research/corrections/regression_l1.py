"""B2-L1 variant: raw-mapping L1 localization term, a recipe change distinct from B2.

Measured B2 failure (B2-DIRECT-001, window 20): every active cell had an inverted,
non-overlapping decoded box, most with a coordinate at or below the -1 decode clamp.
IoU then has exactly zero gradient (no overlap; clamped coordinates are constants),
so the localization head cannot recover. L1 on the RAW mapping, before the clamp,
has a gradient for every active cell. Weight is an unvalidated heuristic.
"""
import torch

from hift_research.training.lifecycle import window_loss


def masked_mapping_l1(loc, target_mapping, weight):
    """Weighted SUM over active cells of mean |raw - target| across the 4 channels.

    Inactive cells are selected away before arithmetic; active nonfinite values fail closed.
    Callers divide by their own (global) localization denominator.
    """
    if loc.ndim != 4 or loc.shape[1] != 4 or target_mapping.shape != loc.shape:
        raise ValueError("expected matching [B,4,H,W] raw and target mappings")
    if weight.device != loc.device or target_mapping.device != loc.device:
        raise ValueError("mappings and weights must share a device")
    if weight.shape[0] != loc.shape[0] or weight.numel() != loc.numel()//4:
        raise ValueError("weights must match batch and spatial cells")
    weight = weight.reshape(loc.shape[0], -1)
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("weights must be finite and nonnegative")
    active = weight > 0
    if not bool(active.any()):
        return loc.reshape(-1)[:0].sum()
    p = loc.flatten(2).permute(0, 2, 1)[active]
    t = target_mapping.flatten(2).permute(0, 2, 1)[active]
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(t).all()):
        raise ValueError("active raw/target mappings must be finite")
    return ((p - t).abs().mean(1) * weight[active]).sum()


def window_loss_with_l1(heads, targets, denominators, *, l1_weight):
    """Unchanged B2 window objective plus l1_weight * globally normalized raw-mapping L1."""
    if not l1_weight >= 0:
        raise ValueError("l1_weight must be nonnegative")
    total = window_loss(heads, targets, denominators)
    global_weight = denominators["localization"]
    l1 = masked_mapping_l1(heads[0], targets["labelxff"], targets["weightxff"])
    if global_weight:
        l1 = l1/(global_weight+1e-6)
    total = total + l1_weight*l1
    if not bool(torch.isfinite(total)):
        raise ValueError("nonfinite B2-L1 window loss")
    return total
