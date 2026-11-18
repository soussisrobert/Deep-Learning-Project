"""B2 loss adapters with selection-before-arithmetic mask semantics.

Inactive NaN/Inf coordinates or logits are ignored, not multiplied by zero after
poisoning the graph. Nonfinite active values/weights and invalid targets fail
closed. These routines do not implement the training schedule or a data loader.
"""


def _empty_zero(tensor):
    return tensor.reshape(-1)[:0].sum()


def masked_iou_loss(pred, target, weight):
    """Weighted pinned-style IoU on [B,N,4] xyxy boxes, active weights only.

    Target extents must be positive. Invalid predicted extents are clamped to
    zero area, so unconstrained initial predictions still have a finite loss.
    Retains upstream denominator epsilon and +1e-6 IoU offset: an exactly correct
    prediction can therefore have a tiny negative loss (a retained heuristic,
    not an exact metric). No change to ordinary positive-extent finite boxes.
    """
    import torch

    if pred.ndim != 3 or pred.shape[-1] != 4 or pred.shape != target.shape:
        raise ValueError("matching [B,N,4] prediction/target boxes required")
    if weight is None or weight.ndim < 1 or weight.shape[0] != pred.shape[0] or weight.numel() != pred.numel() // 4:
        raise ValueError("weight must match batch and box counts")
    if target.device != pred.device or weight.device != pred.device:
        raise ValueError("boxes and weights must share a device")
    weight = weight.reshape(pred.shape[:2])
    if not torch.isfinite(weight).all() or torch.any(weight < 0):
        raise ValueError("weights must be finite and nonnegative")
    active = weight > 0
    if not active.any():
        return _empty_zero(pred)
    p, t, w = pred[active], target[active], weight[active]
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError("active boxes must be finite")
    target_extent = t[:, 2:] - t[:, :2]
    if torch.any(target_extent <= 0) or not torch.isfinite(target_extent).all():
        raise ValueError("active target boxes must have finite positive extents")
    pred_extent = (p[:, 2:] - p[:, :2]).clamp(min=0)
    intersect_extent = (torch.minimum(p[:, 2:], t[:, 2:])
                        - torch.maximum(p[:, :2], t[:, :2])).clamp(min=0)
    target_area = target_extent[:, 0] * target_extent[:, 1]
    pred_area = pred_extent[:, 0] * pred_extent[:, 1]
    intersection = intersect_extent[:, 0] * intersect_extent[:, 1]
    union = target_area + pred_area - intersection
    if not torch.isfinite(union).all() or not torch.isfinite(w.sum()):
        raise ValueError("active geometry/weight reduction overflow")
    iou = (intersection / (union + 1e-6)).clamp(min=0) + 1e-6
    losses = 1 - iou
    return (losses * w).sum() / (w.sum() + 1e-6)


def masked_bce_with_logits(logits, target, weight=None):
    """Dense mean BCE by default, or explicit nonnegative weighted mean.

    Equal shapes are required; this deliberately rejects implicit broadcasting.
    An all-zero mask returns connected zero without evaluating invalid logits.
    """
    import torch
    import torch.nn.functional as functional

    if logits.shape != target.shape or logits.device != target.device:
        raise ValueError("matching logits/target shapes and devices required")
    if weight is None:
        weight = torch.ones_like(logits)
    if weight.shape != logits.shape or weight.device != logits.device:
        raise ValueError("weight must match logits shape and device")
    if not torch.isfinite(weight).all() or torch.any(weight < 0):
        raise ValueError("weights must be finite and nonnegative")
    active = weight > 0
    if not active.any():
        return _empty_zero(logits)
    x, y, w = logits[active], target[active], weight[active]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or torch.any((y < 0) | (y > 1)):
        raise ValueError("active logits must be finite and targets within [0,1]")
    if not torch.isfinite(w.sum()):
        raise ValueError("weight reduction overflow")
    loss = (functional.binary_cross_entropy_with_logits(x, y, reduction="none") * w).sum() / w.sum()
    if not torch.isfinite(loss):
        raise ValueError("active BCE reduction overflow")
    return loss
