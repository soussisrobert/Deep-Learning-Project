"""Opt-in B2 feasibility graph, not released-checkpoint inference or a training recipe."""
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hift_research.corrections.classification_loss import balanced_classification_loss
from hift_research.corrections.tensor_layout import encoder_forward_corrected
from .upstream import classes


def decode_regression(mapping, *, stride=16, offset=63, search_size=287):
    """Released inverse-tanh/xyxy math with actual B/H/W/device, no in-place edits.

    Upstream substitutes +/-0.99 ONLY outside the open interval (-1,1), not
    throughout [0.99,1). Negative/invalid predicted widths are not silently fixed.
    """
    if mapping.ndim != 4 or mapping.shape[1] != 4 or not mapping.is_floating_point():
        raise ValueError("expected floating [B,4,H,W] regression map")
    if min(mapping.shape[0], mapping.shape[2], mapping.shape[3]) < 1:
        raise ValueError("batch/spatial dimensions must be nonempty")
    bounded = torch.where(mapping <= -1, -0.99, torch.where(mapping >= 1, 0.99, mapping))
    distances = (torch.log1p(bounded) - torch.log1p(-bounded)) * (search_size // 2) / 2
    h, w = mapping.shape[-2:]
    x = torch.arange(w, device=mapping.device, dtype=mapping.dtype) * stride + offset
    y = torch.arange(h, device=mapping.device, dtype=mapping.dtype) * stride + offset
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx-distances[:, 0], yy-distances[:, 2],
                        xx+distances[:, 1], yy+distances[:, 3]), dim=-1).flatten(1, 2)


def masked_decode_regression(mapping, weight):
    """Select inactive raw coordinates before nonlinear decode, including backward.

    Multiplying NaN-derived boxes by zero after inverse-tanh is not sufficient:
    its derivative can produce 0*NaN. Invalid active mappings fail closed.
    """
    if mapping.ndim != 4 or mapping.shape[1] != 4:
        raise ValueError("expected [B,4,H,W] mapping")
    if weight.device != mapping.device or weight.shape[0] != mapping.shape[0] or weight.numel() != mapping.numel()//4:
        raise ValueError("weights must match mapping batch/spatial dimensions/device")
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        raise ValueError("weights must be finite and nonnegative")
    active = weight.reshape(mapping.shape[0], 1, *mapping.shape[-2:]).gt(0).expand_as(mapping)
    if not bool(torch.isfinite(mapping[active]).all()):
        raise ValueError("active regression mappings must be finite")
    return decode_regression(torch.where(active, mapping, 0.0))


def masked_classification_log_probs(logits, labels):
    """Ignore cells before log-softmax; invalid active logits fail closed."""
    if logits.ndim != 4 or logits.shape[1] != 2 or labels.shape != (logits.shape[0], 1, *logits.shape[-2:]):
        raise ValueError("expected [B,2,H,W] logits and [B,1,H,W] labels")
    if labels.device != logits.device or not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must be finite and share logits device")
    allowed = labels.eq(-2) | labels.eq(-1) | labels.eq(0) | labels.eq(1)
    if not bool(allowed.all()):
        raise ValueError("cls1 labels must be -2/-1/0/1")
    active = (labels.eq(0) | labels.eq(1)).expand_as(logits)
    if not bool(torch.isfinite(logits[active]).all()):
        raise ValueError("active classification logits must be finite")
    return F.log_softmax(torch.where(active, logits, 0.0).permute(0, 2, 3, 1), dim=-1)


class B2TrainingGraph(nn.Module):
    """Actual pinned AlexNet, correlation, transformer, heads; explicit B2 layout.

    Random initialization is intentional. Train layers3-5 plus grader; layers1-2
    frozen and their BatchNorm statistics frozen, a disclosed late-phase proxy.
    No official accuracy, checkpoint reproduction, or full-training claim.
    """
    def __init__(self):
        super().__init__()
        alex, hift, _ = classes()
        self.backbone = alex()
        self.grader = hift(SimpleNamespace(TRAIN=SimpleNamespace(groupchannel=32)))
        for layer in self.grader.transformer.encoder.layers:
            layer.forward = MethodType(encoder_forward_corrected, layer)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "backbone"):
            self.backbone.layer1.eval()
            self.backbone.layer2.eval()
        return self

    def features(self, template, search):
        return self.backbone(template), self.backbone(search)

    def losses(self, heads, targets):
        from hift_research.corrections.masked_losses import masked_bce_with_logits, masked_iou_loss
        loc, cls1, cls2 = heads
        logp = masked_classification_log_probs(cls1, targets["label_cls1"])
        cls_loss1 = balanced_classification_loss(logp, targets["label_cls1"])
        cls_loss2 = masked_bce_with_logits(cls2, targets["labelcls2"])
        weights = targets["weightxff"]
        loc_loss = masked_iou_loss(masked_decode_regression(loc, weights),
                                   masked_decode_regression(targets["labelxff"], weights), weights)
        return {"total_loss": 2.4*loc_loss + cls_loss1 + cls_loss2,
                "loc_loss": loc_loss, "cls_loss1": cls_loss1, "cls_loss2": cls_loss2}

    def forward(self, template, search, targets):
        z, x = self.features(template, search)
        return self.losses(self.grader(x, z), targets)


def synthetic_batch(batch_size, device="cpu", *, negative=None):
    """Resident synthetic FP32 crops, normal upstream positive targets, dense negatives.

    Pixels uniform [0,255] mimic upstream unnormalized image scale, not image
    statistics. B1 alternation is controlled by caller; larger B mixes both types.
    """
    from hift_research.corrections.positive_targets import CorrectedAnchorTarget, robust_pair_targets
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    negative = ([False if batch_size == 1 else bool(i % 2) for i in range(batch_size)]
                if negative is None else list(negative))
    if len(negative) != batch_size:
        raise ValueError("negative flags must match batch")
    bbox = SimpleNamespace(x1=95, y1=95, x2=223, y2=223)
    anchor = CorrectedAnchorTarget(SimpleNamespace(TRAIN=SimpleNamespace(
        SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11, LABEL_RANGE=4)))
    rows = [robust_pair_targets(anchor, bbox, 11, is_negative=flag,
                               template_identity=("synthetic", "template", str(i)),
                               search_identity=("synthetic", "unrelated" if flag else "template", str(i)))
            for i, flag in enumerate(negative)]
    keys = ("label_cls1", "labelxff", "labelcls2", "weightxff")
    targets = {key: torch.as_tensor(np.stack([row[i] for row in rows]),
                                   dtype=torch.float32, device=device) for i, key in enumerate(keys)}
    template = torch.rand(batch_size, 3, 127, 127, device=device)*255
    search = torch.rand(batch_size, 3, 287, 287, device=device)*255
    return template, search, targets


def gradient_audit(model):
    """Per-parameter finite/nonzero audit; declare unused upstream eles and gamma gate."""
    dormant, frozen, zero_gated = [], [], []
    missing, zero, nonfinite, active = [], [], [], []
    gamma_zero = {f"grader.transformer.encoder.layers.{i}.cross_attn.":
                  bool(layer.cross_attn.gamma.detach().eq(0).all())
                  for i, layer in enumerate(model.grader.transformer.encoder.layers)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(name)
            continue
        if ".eles." in name:
            dormant.append(name)
            if parameter.grad is not None:
                raise AssertionError("upstream dormant eles unexpectedly receives gradient")
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif not bool(parameter.grad.ne(0).any()):
            if any(name.startswith(prefix) and gated and not name.endswith("gamma")
                   for prefix, gated in gamma_zero.items()):
                zero_gated.append(name)
            else:
                zero.append(name)
        else:
            active.append(name)
    return {"active_nonzero": active, "expected_dormant": dormant, "frozen": frozen,
            "expected_initial_gamma_zero": zero_gated, "unexpected_missing": missing,
            "unexpected_zero": zero, "nonfinite": nonfinite}


def optimizer_for(model):
    return torch.optim.SGD([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 0.0005},
        {"params": model.grader.parameters(), "lr": 0.005},
    ], momentum=0.9, weight_decay=0.0001)
