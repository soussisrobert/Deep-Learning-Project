import math

import pytest
import torch

from hift_research.corrections.masked_losses import masked_iou_loss
from hift_research.corrections.regression_l1 import masked_mapping_l1, window_loss_with_l1
from hift_research.training.graph import masked_decode_regression
from hift_research.training.lifecycle import window_loss


def _single_cell(raw, target=(0.2, 0.2, 0.2, 0.2)):
    loc = torch.zeros(1, 4, 2, 2)
    loc[0, :, 0, 0] = torch.tensor(raw)
    labels = torch.zeros(1, 4, 2, 2)
    labels[0, :, 0, 0] = torch.tensor(target)
    weight = torch.zeros(1, 2, 2)
    weight[0, 0, 0] = 1.0
    return loc.requires_grad_(), labels, weight


@pytest.mark.parametrize("raw", [(-1.5, -1.5, 0.1, 0.1), (-0.3, -0.3, -0.3, -0.3)])
def test_iou_has_zero_gradient_where_l1_does_not(raw):
    # Clamped (-1.5) and unclamped-but-inverted (-0.3) boxes: the measured B2 dead zone.
    loc, labels, weight = _single_cell(raw)
    iou = masked_iou_loss(masked_decode_regression(loc, weight), masked_decode_regression(labels, weight), weight)
    (grad,) = torch.autograd.grad(iou, loc)
    assert torch.count_nonzero(grad) == 0
    (grad,) = torch.autograd.grad(masked_mapping_l1(loc, labels, weight), loc)
    assert torch.count_nonzero(grad[0, :, 0, 0]) == 4
    assert torch.count_nonzero(grad) == 4


def test_exact_value_and_inactive_nan_ignored():
    loc, labels, weight = _single_cell((0.5, 0.2, -0.2, 1.2))
    with torch.no_grad():
        loc[0, :, 1, 1] = float("nan")  # inactive cell must not poison value or gradient
    weight[0, 0, 0] = 2.0
    value = masked_mapping_l1(loc, labels, weight)
    assert math.isclose(float(value), 2.0*(0.3+0.0+0.4+1.0)/4, rel_tol=1e-6)
    (grad,) = torch.autograd.grad(value, loc)
    assert bool(torch.isfinite(grad).all())


def test_empty_mask_is_connected_zero_and_active_nonfinite_fails():
    loc, labels, weight = _single_cell((0.1, 0.1, 0.1, 0.1))
    value = masked_mapping_l1(loc, labels, torch.zeros_like(weight))
    assert float(value) == 0.0 and value.requires_grad
    with torch.no_grad():
        loc[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        masked_mapping_l1(loc, labels, weight)
    with pytest.raises(ValueError):
        masked_mapping_l1(loc.detach(), labels, -weight)
    with pytest.raises(ValueError):
        masked_mapping_l1(loc.detach(), labels[:, :, :1], weight)


def test_window_variant_adds_globally_normalized_term_and_zero_weight_is_b2():
    torch.manual_seed(0)
    b, h = 2, 11
    loc = (torch.rand(b, 4, h, h)*0.4).requires_grad_()
    cls1, cls2 = torch.randn(b, 2, h, h), torch.randn(b, 1, h, h)
    weight = torch.zeros(b, h, h); weight[:, 4:7, 4:7] = 1.0
    labels = torch.full((b, 1, h, h), -1.0); labels[:, 0, 5, 5] = 1; labels[:, 0, 0, 0] = 0
    targets = {"label_cls1": labels, "labelcls2": torch.rand(b, 1, h, h), "weightxff": weight,
               "labelxff": torch.rand(b, 4, h, h)*0.4+0.1}
    denominators = {"positive": 2, "negative": 2, "bce": b*h*h, "localization": 40.0}
    heads = (loc, cls1, cls2)
    base = window_loss(heads, targets, denominators)
    assert torch.equal(window_loss_with_l1(heads, targets, denominators, l1_weight=0.0), base)
    expected = masked_mapping_l1(loc, targets["labelxff"], weight)/(40.0+1e-6)
    variant = window_loss_with_l1(heads, targets, denominators, l1_weight=1.5)
    assert math.isclose(float(variant), float(base+1.5*expected), rel_tol=1e-6)
    with pytest.raises(ValueError):
        window_loss_with_l1(heads, targets, denominators, l1_weight=float("nan"))
