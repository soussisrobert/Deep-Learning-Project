"""Wave2 B2 target/loss regression checks; CPU only, no training."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hift_research.corrections.classification_loss import balanced_classification_loss
from hift_research.corrections.masked_losses import masked_bce_with_logits, masked_iou_loss
from hift_research.corrections.positive_targets import (
    CorrectedAnchorTarget, NegativeIdentityError, robust_pair_targets, validate_negative_identity,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/HiFT"


def _cfg(size=11):
    return SimpleNamespace(TRAIN=SimpleNamespace(
        OUTPUTFEATURE_SIZE=size, SEARCH_SIZE=287, LABEL_RANGE=4,
    ))


def _box(values):
    return SimpleNamespace(**dict(zip(("x1", "y1", "x2", "y2"), values)))


def _source_symbol(path, name, namespace):
    tree = ast.parse((UPSTREAM / path).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(UPSTREAM / path), "exec"), namespace)
    return namespace[name]


@pytest.fixture(autouse=True)
def _numpy_seed():
    old = np.random.get_state()
    np.random.seed(20261023)
    yield
    np.random.set_state(old)


@pytest.mark.parametrize("bbox", [
    (95, 95, 223, 223), (80, 110, 180, 200), (0, 0, 40, 80),
    (-5, 30, 60, 120), (230, 210, 285, 286), (20, 70, 250, 180),
])
@pytest.mark.parametrize("seed", [0, 20261023, 31])
def test_ordinary_finite_positive_is_bit_identical_including_rng(bbox, seed):
    cfg = _cfg()
    upstream_class = _source_symbol("pysot/datasets/anchortarget.py", "AnchorTarget", {"np": np, "cfg": cfg})
    np.random.seed(seed)
    expected = upstream_class().get(_box(bbox), 11)
    expected_next_random = np.random.random(5)
    np.random.seed(seed)
    actual = CorrectedAnchorTarget(cfg).get(_box(bbox), 11)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(right).all()
    np.testing.assert_array_equal(expected_next_random, np.random.random(5))


def test_retained_radial_boundary_heuristic_is_explicit_not_silently_repaired():
    cfg = _cfg()
    original_class = _source_symbol("pysot/datasets/anchortarget.py", "AnchorTarget", {"np": np, "cfg": cfg})
    bbox = _box((63, 63, 95, 127))
    np.random.seed(9)
    original = original_class().get(bbox, 11)
    np.random.seed(9)
    corrected = CorrectedAnchorTarget(cfg).get(bbox, 11)
    for x, y in zip(original, corrected):
        np.testing.assert_array_equal(x, y)
    # The strict > radial cutoff retains raw distance=64 on the boundary.
    # This finite but uncalibrated source heuristic is deliberately not retuned.
    assert corrected[3].max() == 64


@pytest.mark.parametrize("bbox", [
    (143, 143, 144, 144), (144, 144, 145, 145),
    (1, 1, 2, 2), (-0.5, 0, 0.5, 1), (285, 285, 288, 288),
    (230, 230, 285, 286),  # Non-tiny physical box, but clamped to one grid cell.
])
def test_tiny_positive_has_minimal_support_and_decodes_actual_box(bbox):
    upstream_class = _source_symbol("pysot/datasets/anchortarget.py", "AnchorTarget", {"np": np, "cfg": _cfg()})
    with pytest.raises(ValueError, match="zero-size array"):
        upstream_class().get(_box(bbox), 11)
    cls1, encoded, cls2, weight = CorrectedAnchorTarget(_cfg()).get(_box(bbox), 11)
    assert all(np.isfinite(x).all() for x in (cls1, encoded, cls2, weight))
    assert np.count_nonzero(cls1 == 1) == 1
    assert np.count_nonzero(cls2) == np.count_nonzero(weight) == 1
    assert cls2.max() == weight.max() == 1
    row, col = np.argwhere(weight[0] > 0)[0]
    # Grid point may lie OUTSIDE a tiny bbox. Signed side offsets are valid;
    # decoding, not positivity of all four distances, is the consistency test.
    offset = np.arctanh(encoded[:, row, col].astype(np.float64)) * 143
    gx, gy = 63 + 16 * col, 63 + 16 * row
    decoded = np.array([gx - offset[0], gy - offset[2], gx + offset[1], gy + offset[3]])
    np.testing.assert_allclose(decoded, bbox, rtol=0, atol=6e-5)
    if bbox == (144, 144, 145, 145):
        assert gx < bbox[0] and offset[0] < 0


def test_size_one_is_supported_without_squeeze_scalar_error():
    cls1, loc, cls2, weight = CorrectedAnchorTarget(_cfg(1)).get(_box((63, 63, 64, 64)), 1)
    assert cls1.item() == cls2.item() == weight.item() == 1
    assert loc.shape == (4, 1, 1)


@pytest.mark.parametrize("bbox", [
    (10, 10, 10, 20), (20, 10, 10, 20), (0, 0, np.nan, 20),
    (-20, 0, -1, 20), (287, 0, 300, 20), (0, 300, 10, 350),
])
def test_invalid_or_fully_outside_positives_fail_closed(bbox):
    with pytest.raises(ValueError):
        CorrectedAnchorTarget(_cfg()).get(_box(bbox), 11)


def test_mismatched_target_grid_fails_before_construction():
    with pytest.raises(ValueError, match="OUTPUTFEATURE_SIZE"):
        CorrectedAnchorTarget(_cfg()).get(_box((95, 95, 223, 223)), 5)


@pytest.mark.parametrize("setting,value", [("LABEL_RANGE", 0), ("LABEL_RANGE", 1e308),
                                           ("SEARCH_SIZE", 287.5), ("SEARCH_SIZE", 1)])
def test_invalid_or_overflowing_configuration_is_rejected(setting, value):
    cfg = _cfg()
    setattr(cfg.TRAIN, setting, value)
    with pytest.raises(ValueError):
        CorrectedAnchorTarget(cfg).get(_box((95, 95, 223, 223)), 11)


def test_extreme_unrepresentable_box_fails_closed():
    with pytest.raises(ValueError, match="representable"):
        CorrectedAnchorTarget(_cfg()).get(_box((-1e40, -1e40, 1e40, 1e40)), 11)


def test_large_partial_crop_positive_uses_finite_encoding_fallback():
    with np.errstate(over="raise", invalid="raise"):
        labels = CorrectedAnchorTarget(_cfg()).get(_box((-1e5, -1e5, 1e5, 1e5)), 11)
    assert all(np.isfinite(value).all() for value in labels)
    assert np.all(np.abs(labels[1]) <= 1)


@pytest.mark.parametrize("template,search", [
    (None, None), (("data", "video", "track"), ("data", "video", "track")),
    (("data", "video", ""), ("other", "video", "track")),
    (("data", "video", "track", "frame1"), ("data", "video", "track", "frame2")),
])
def test_negative_identity_missing_collision_or_frame_keys_are_rejected(template, search):
    with pytest.raises(NegativeIdentityError):
        robust_pair_targets(None, None, 11, is_negative=True,
                            template_identity=template, search_identity=search)


def test_identity_checked_negative_is_all_background_without_reading_bbox():
    labels = robust_pair_targets(None, None, 11, is_negative=True,
                                 template_identity=("dataset", "video", "track1"),
                                 search_identity=("dataset", "video", "track2"))
    assert all(np.count_nonzero(value) == 0 for value in labels)
    validate_negative_identity(("dataset", "video1", "track"), ("dataset", "video2", "track"))


def test_actual_upstream_random_target_can_guarantee_same_object_collision():
    subset_class = _source_symbol("pysot/datasets/dataset.py", "SubDataset", {"np": np, "os": os})
    subset = object.__new__(subset_class)
    subset.num, subset.videos = 1, ["video"]
    subset.labels = {"video": {"track": {"frames": [1], "000001": [10, 10]}}}
    subset.root, subset.path_format = "fake-root", "{}.{}.{}.jpg"
    template, search = subset.get_random_target(0), subset.get_random_target()
    assert template == search
    # Metadata identity excludes frame. Even different frames would collide.
    with pytest.raises(NegativeIdentityError):
        robust_pair_targets(None, None, 11, is_negative=True,
                            template_identity=("dataset", "video", "track"),
                            search_identity=("dataset", "video", "track"))


def test_identity_wrapper_does_not_relabel_or_consume_random_draws():
    before = np.random.get_state()
    with pytest.raises(NegativeIdentityError):
        validate_negative_identity(("d", "v", "t"), ("d", "v", "t"))
    after = np.random.get_state()
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old, new)


def test_positive_wrapper_uses_corrected_anchor_without_identity_requirement():
    result = robust_pair_targets(CorrectedAnchorTarget(_cfg()), _box((144, 144, 145, 145)),
                                 11, is_negative=False)
    assert result[2].max() == result[3].max() == 1


def _patch_builder(filename):
    path = ROOT / "patches" / filename
    spec = importlib.util.spec_from_file_location("wave2_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bbox", [(95, 95, 223, 223), (144, 144, 145, 145), (-0.5, 0, 0.5, 1)])
def test_unapplied_positive_patch_matches_adapter_on_ordinary_and_fallback_paths(bbox):
    source = (UPSTREAM / "pysot/datasets/anchortarget.py").read_text(encoding="utf-8")
    patched = _patch_builder("0005_build_positive_patch.py").corrected_source(source)
    node = next(item for item in ast.parse(patched).body if getattr(item, "name", None) == "AnchorTarget")
    namespace = {"np": np, "cfg": _cfg()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<unapplied-patch-0005>", "exec"), namespace)
    np.random.seed(17)
    expected = namespace["AnchorTarget"]().get(_box(bbox), 11)
    np.random.seed(17)
    actual = CorrectedAnchorTarget(_cfg()).get(_box(bbox), 11)
    for x, y in zip(expected, actual):
        np.testing.assert_array_equal(x, y)


def _torch():
    return pytest.importorskip("torch")


def test_empty_class_loss_ignores_nonfinite_ignored_values_and_has_zero_gradient():
    torch = _torch()
    pred = torch.tensor([[float("nan"), -float("inf")]], requires_grad=True)
    loss = balanced_classification_loss(pred, torch.tensor([-2]))
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


@pytest.mark.parametrize("label", [0, 1])
def test_singleton_true_class_is_supervised_and_ignored_nonfinites_do_not_poison(label):
    torch = _torch()
    pred = torch.tensor([[-1., -2.], [float("nan"), float("inf")]], requires_grad=True)
    loss = balanced_classification_loss(pred, torch.tensor([label, -1]))
    torch.testing.assert_close(loss, -0.5 * pred[0, label])
    loss.backward()
    assert pred.grad[0, label].item() == -0.5
    assert torch.count_nonzero(pred.grad[1]) == 0


@pytest.mark.parametrize("label", [float("nan"), 2, -3])
def test_classification_rejects_invalid_labels(label):
    torch = _torch()
    with pytest.raises(ValueError):
        balanced_classification_loss(torch.zeros(1, 2), torch.tensor([label]))


def test_classification_rejects_nonfinite_active_target_probability():
    torch = _torch()
    with pytest.raises(ValueError):
        balanced_classification_loss(torch.tensor([[0., -float("inf")]]), torch.tensor([1]))


def test_empty_tensor_classification_is_connected_zero():
    torch = _torch()
    pred = torch.empty(0, 2, requires_grad=True)
    loss = balanced_classification_loss(pred, torch.empty(0, dtype=torch.long))
    assert loss.item() == 0
    loss.backward()
    assert pred.grad.shape == pred.shape


def test_masked_iou_selects_before_nonfinite_arithmetic_and_zeroes_inactive_gradients():
    torch = _torch()
    pred = torch.tensor([[[0., 0., 10., 10.], [float("nan")] * 4]], requires_grad=True)
    target = torch.tensor([[[1., 1., 11., 11.], [float("inf")] * 4]])
    weight = torch.tensor([[[[1., 0.]]]])
    loss = masked_iou_loss(pred, target, weight)
    expected = masked_iou_loss(pred[:, :1], target[:, :1], torch.ones(1, 1))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and torch.count_nonzero(pred.grad[:, 1]) == 0


def test_actual_upstream_iou_zero_mask_is_not_nan_safe_but_adapter_is():
    torch = _torch()
    original_class = _source_symbol("pysot/models/loss.py", "IOULoss", {"torch": torch, "nn": torch.nn})
    pred = torch.full((1, 1, 4), float("nan"), requires_grad=True)
    target, weight = torch.zeros_like(pred), torch.zeros(1, 1)
    assert torch.isnan(original_class()(pred, target, weight))
    corrected = masked_iou_loss(pred, target, weight)
    assert corrected.item() == 0
    corrected.backward()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


def test_masked_iou_preserves_ordinary_positive_extent_loss_and_gradient():
    torch = _torch()
    original_class = _source_symbol("pysot/models/loss.py", "IOULoss", {"torch": torch, "nn": torch.nn})
    pred = torch.tensor([[[0., 0., 10., 10.], [3., 3., 8., 8.]]], requires_grad=True)
    target = torch.tensor([[[1., 1., 11., 11.], [4., 4., 9., 9.]]])
    weight = torch.tensor([[0.7, 1.5]])
    expected = original_class()(pred, target, weight)
    actual = masked_iou_loss(pred, target, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, pred, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("case", ["active_nan", "negative_weight", "nan_weight", "zero_target_extent"])
def test_iou_active_invalid_inputs_fail_closed(case):
    torch = _torch()
    pred = torch.tensor([[[0., 0., 10., 10.]]])
    target, weight = pred.clone(), torch.ones(1, 1)
    if case == "active_nan":
        pred[0, 0, 0] = float("nan")
    elif case == "negative_weight":
        weight.fill_(-1)
    elif case == "nan_weight":
        weight.fill_(float("nan"))
    else:
        target[0, 0, 2] = 0
    with pytest.raises(ValueError):
        masked_iou_loss(pred, target, weight)


def test_unapplied_mask_patch_executes_and_matches_adapter():
    torch = _torch()
    source = (UPSTREAM / "pysot/models/loss.py").read_text(encoding="utf-8")
    patched = _patch_builder("0006_build_mask_patch.py").corrected_source(source)
    node = next(item for item in ast.parse(patched).body if getattr(item, "name", None) == "IOULoss")
    namespace = {"nn": torch.nn, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<unapplied-patch-0006>", "exec"), namespace)
    pred = torch.tensor([[[0., 0., 10., 10.], [float("nan")] * 4]], requires_grad=True)
    target = torch.tensor([[[1., 1., 11., 11.], [float("inf")] * 4]])
    weight = torch.tensor([[1., 0.]])
    torch.testing.assert_close(namespace["IOULoss"]()(pred, target, weight),
                               masked_iou_loss(pred, target, weight), rtol=0, atol=0)


def test_empty_box_and_bce_tensors_return_connected_zero():
    torch = _torch()
    pred = torch.empty(0, 0, 4, requires_grad=True)
    loss = masked_iou_loss(pred, pred.detach(), torch.empty(0, 0))
    assert loss.item() == 0
    loss.backward()
    assert pred.grad.shape == pred.shape
    logits = torch.empty(2, 0, requires_grad=True)
    bce = masked_bce_with_logits(logits, logits.detach())
    assert bce.item() == 0
    bce.backward()
    assert logits.grad.shape == logits.shape


def test_active_loss_overflows_fail_closed():
    torch = _torch()
    with pytest.raises(ValueError, match="overflow"):
        masked_iou_loss(torch.tensor([[[0., 0., 1e30, 1e30]]]),
                        torch.tensor([[[0., 0., 1e30, 1e30]]]), torch.ones(1, 1))
    with pytest.raises(ValueError, match="overflow"):
        masked_bce_with_logits(torch.tensor([3e38, 3e38]), torch.zeros(2))
    with pytest.raises(ValueError, match="overflow"):
        balanced_classification_loss(torch.full((2, 2), -3e38), torch.tensor([1, 1]))


def test_iou_nonpositive_predicted_extent_is_finite_without_assertion_substitution():
    torch = _torch()
    pred = torch.tensor([[[10., 10., 0., 0.]]], requires_grad=True)
    target = torch.tensor([[[0., 0., 10., 10.]]])
    loss = masked_iou_loss(pred, target, torch.ones(1, 1))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_bce_dense_is_upstream_mean_and_masked_nonfinites_are_ignored():
    torch = _torch()
    logits = torch.tensor([-1000., 0., 1000.], requires_grad=True)
    target = torch.tensor([0., 0.5, 1.])
    torch.testing.assert_close(masked_bce_with_logits(logits, target),
                               torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
    x = torch.tensor([0., float("nan")], requires_grad=True)
    y, weight = torch.tensor([1., float("nan")]), torch.tensor([1., 0.])
    masked_bce_with_logits(x, y, weight).backward()
    assert x.grad[0].item() == -0.5 and x.grad[1].item() == 0


def test_bce_empty_zero_mask_is_connected_zero_and_bad_active_target_rejected():
    torch = _torch()
    logits = torch.tensor([float("inf")], requires_grad=True)
    zero = masked_bce_with_logits(logits, torch.tensor([float("nan")]), torch.zeros(1))
    assert zero.item() == 0
    zero.backward()
    assert logits.grad.item() == 0
    with pytest.raises(ValueError):
        masked_bce_with_logits(torch.zeros(1), torch.tensor([1.1]))
