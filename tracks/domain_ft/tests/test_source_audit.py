"""Deterministic source proofs; no training, GPU, data, or checkpoint required.

NumPy tests execute selected AST nodes from the pinned source unchanged, without
importing its CUDA/torch-dependent modules. Optional torch tests execute the real
transformer on CPU. They are synthetic correctness tests, not benchmark results.
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from hift_research.corrections.distributed_guard import require_single_process
from hift_research.corrections.pair_targets import pair_targets

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "HiFT"


def _source_node(relative, name, parent=None):
    path = UPSTREAM / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if parent:
        tree = next(node for node in tree.body if getattr(node, "name", None) == parent)
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    return node, str(path)


def _execute_source(relative, name, namespace, parent=None):
    node, filename = _source_node(relative, name, parent)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def _anchor():
    cfg = SimpleNamespace(TRAIN=SimpleNamespace(
        SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11, LABEL_RANGE=4,
    ))
    anchor_class = _execute_source(
        "pysot/datasets/anchortarget.py", "AnchorTarget", {"np": np, "cfg": cfg},
    )
    return anchor_class()


@pytest.fixture(autouse=True)
def _deterministic_numpy():
    state = np.random.get_state()
    np.random.seed(20261023)
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_outer_transformer_map_sequence_roundtrip(batch):
    # utile.py:108-112: proper map->SBC->map conversion preserves every index.
    x = np.arange(batch * 192 * 121).reshape(batch, 192, 11, 11)
    sequence = x.reshape(batch, 192, -1).transpose(2, 0, 1)
    restored = sequence.transpose(1, 2, 0).reshape(x.shape)
    np.testing.assert_array_equal(restored, x)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_encoder_complete_layout_mapping_gamma_zero(batch):
    # A contiguous result after layer norm has source logical order S,B,C.
    s, c = 121, 192
    src = np.arange(s * batch * c).reshape(s, batch, c)
    # tran.py:250-251, with actual initialized Cattention gamma=0.
    wrong_map = src.reshape(batch, c, 11, 11)
    upstream_output = wrong_map.reshape(batch, c, s).transpose(2, 0, 1)
    source_flat_indices = np.arange(src.size).reshape(batch, c, s).transpose(2, 0, 1)
    np.testing.assert_array_equal(upstream_output, src.ravel()[source_flat_indices])
    assert not np.array_equal(upstream_output, src)
    corrected_map = src.transpose(1, 2, 0).reshape(batch, c, 11, 11)
    corrected_output = corrected_map.reshape(batch, c, s).transpose(2, 0, 1)
    np.testing.assert_array_equal(corrected_output, src)


@pytest.mark.parametrize("batch", [2, 4])
def test_encoder_upstream_mapping_crosses_sample_boundaries(batch):
    src = np.broadcast_to(np.arange(batch)[None, :, None], (121, batch, 192)).copy()
    upstream = src.reshape(batch, 192, 121).transpose(2, 0, 1)
    for b in range(batch):
        assert set(np.unique(upstream[:, b, :])) == set(range(batch))
    perturbed = src.copy()
    perturbed[:, -1, :] += 100
    wrong_perturbed = perturbed.reshape(batch, 192, 121).transpose(2, 0, 1)
    assert np.any(wrong_perturbed[:, 0, :] != upstream[:, 0, :])
    correct = src.transpose(1, 2, 0).reshape(batch, 192, 11, 11)
    correct_perturbed = perturbed.transpose(1, 2, 0).reshape(batch, 192, 11, 11)
    np.testing.assert_array_equal(correct_perturbed[0], correct[0])


def test_actual_negative_sampler_drops_flag_and_supervises_positive_object():
    calls = []
    bbox = SimpleNamespace(x1=95, y1=95, x2=223, y2=223)

    class Subset:
        def get_random_target(self, index=-1):
            calls.append(index)
            return ("template" if index == 0 else "unrelated-search", (128, 128))

        def get_positive_pair(self, index):
            raise AssertionError("negative branch must not sample a positive pair")

    cfg = SimpleNamespace(DATASET=SimpleNamespace(GRAY=0.0, NEG=1.0),
                          TRAIN=SimpleNamespace(EXEMPLAR_SIZE=127, SEARCH_SIZE=287, OUTPUT_SIZE=11))
    function = _execute_source(
        "pysot/datasets/dataset.py", "__getitem__",
        {"np": np, "cfg": cfg, "cv2": SimpleNamespace(imread=lambda path: np.zeros((287, 287, 3)))},
        parent="TrkDataset",
    )
    subset = Subset()
    dataset = SimpleNamespace(
        pick=[0], all_dataset=[subset], _find_dataset=lambda index: (subset, 0),
        _get_bbox=lambda image, shape: bbox, anchor_target=_anchor(),
        template_aug=lambda image, box, size, gray: (image, box),
        search_aug=lambda image, box, size, gray: (image, box),
    )
    result = function(dataset, 0)
    assert calls == [0, -1]
    assert np.count_nonzero(result["label_cls1"] == 1) > 0
    assert np.max(result["labelcls2"]) == 1
    assert np.sum(result["weightxff"]) > 0
    # At zero logits BCE gradient is .5 - target: a negative pair is attracted
    # to the unrelated search object at target=1 instead of repelled (target=0).
    peak = np.argmax(result["labelcls2"])
    assert (0.5 - result["labelcls2"].ravel()[peak]) < 0


def test_actual_tiny_positive_bbox_has_empty_radial_support():
    # Proven valid-input edge failure; frequency in real augmentation is unknown.
    bbox = SimpleNamespace(x1=143, y1=143, x2=144, y2=144)
    with pytest.raises(ValueError, match="zero-size array"):
        _anchor().get(bbox, 11)


def test_isolated_negative_policy_zeroes_both_classes_and_regression_mask():
    class MustNotCall:
        def get(self, bbox, size):
            raise AssertionError("known negatives must not generate positive-object labels")

    cls1, loc, cls2, weight = pair_targets(MustNotCall(), None, 11, is_negative=True)
    assert cls1.shape == cls2.shape == weight.shape == (1, 11, 11)
    assert loc.shape == (4, 11, 11)
    assert np.all(cls1 == 0) and np.all(cls2 == 0) and np.all(weight == 0)
    assert loc.dtype == np.float32 and np.isfinite(loc).all()
    arbitrary_iou_loss = np.full((1, 121), 0.9)
    assert np.sum(arbitrary_iou_loss * weight.reshape(1, 121)) == 0
    assert np.all(0.5 - cls2 > 0)


def test_isolated_positive_policy_delegates_without_changing_targets():
    sentinel = tuple(object() for _ in range(4))
    anchor = SimpleNamespace(get=lambda bbox, size: sentinel)
    assert pair_targets(anchor, None, 11, is_negative=False) is sentinel


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_pair_policy_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        pair_targets(None, None, size, is_negative=True)


def test_actual_public_dist_init_ignores_launcher_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    namespace = {"os": os, "inited": False}
    dist_init = _execute_source("pysot/utils/distributed.py", "dist_init", namespace)
    assert dist_init() == (0, 1)
    assert namespace["rank"] == 0 and namespace["world_size"] == 1
    assert namespace["inited"] is True


@pytest.mark.parametrize("environ", [{}, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}])
def test_single_process_guard_allows_only_documented_mode(environ):
    assert require_single_process(environ) == (0, 1)


@pytest.mark.parametrize("environ", [
    {"WORLD_SIZE": "2"}, {"RANK": "1"}, {"LOCAL_RANK": "1"}, {"WORLD_SIZE": "0"},
])
def test_single_process_guard_fails_closed(environ):
    with pytest.raises(RuntimeError, match="DataParallel"):
        require_single_process(environ)


def _torch_module():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("_audit_pinned_tran", UPSTREAM / "pysot/models/utile/tran.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, module


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_torch_actual_backbone_correlation_and_head_shapes_cpu(batch):
    torch, module = _torch_module()
    torch.manual_seed(20261023)
    torch.set_num_threads(1)
    alexnet_class = _execute_source(
        "pysot/models/backbone/newalexnet.py", "AlexNet", {"nn": torch.nn},
    )
    # Explicit compatibility shim ONLY for two positional arange(...).cuda()
    # calls; no model mathematics or parameters are rewritten.
    cpu_torch = SimpleNamespace(
        nn=torch.nn, cat=torch.cat,
        arange=lambda n: SimpleNamespace(cuda=lambda: torch.arange(n)),
    )
    hift_class = _execute_source(
        "pysot/models/utile/utile.py", "HiFT",
        {"nn": torch.nn, "F": torch.nn.functional, "t": cpu_torch, "Transformer": module.Transformer},
    )
    backbone = alexnet_class().eval()
    head = hift_class(SimpleNamespace(TRAIN=SimpleNamespace(groupchannel=32))).eval()
    with torch.no_grad():
        template = backbone(torch.randn(batch, 3, 127, 127))
        search = backbone(torch.randn(batch, 3, 287, 287))
        assert [tuple(x.shape) for x in template] == [
            (batch, 384, 10, 10), (batch, 384, 8, 8), (batch, 256, 6, 6),
        ]
        assert [tuple(x.shape) for x in search] == [
            (batch, 384, 30, 30), (batch, 384, 28, 28), (batch, 256, 26, 26),
        ]
        for x, z in zip(search, template):
            assert head.xcorr_depthwise(x, z).shape[-2:] == (21, 21)
        outputs = head(search, template)
    assert [tuple(x.shape) for x in outputs] == [
        (batch, 4, 11, 11), (batch, 2, 11, 11), (batch, 1, 11, 11),
    ]
    assert all(torch.isfinite(x).all() for x in outputs)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_torch_actual_transformer_shapes_and_layout_repair(batch):
    torch, module = _torch_module()
    from hift_research.corrections.tensor_layout import encoder_forward_corrected
    torch.manual_seed(20261023)
    torch.set_num_threads(1)
    # Full configured channel count, one encoder/two decoders, full 11x11 map.
    upstream = module.Transformer(192, 6, 1, 2).eval()
    corrected = copy.deepcopy(upstream)
    for layer in corrected.encoder.layers:
        layer.forward = MethodType(encoder_forward_corrected, layer)
    src, srcc, tgt = (torch.randn(121, batch, 192) for _ in range(3))
    with torch.no_grad():
        original_output = upstream(src, srcc, tgt)
        corrected_output = corrected(src, srcc, tgt)
    assert original_output.shape == corrected_output.shape == (121, batch, 192)
    assert torch.isfinite(original_output).all() and torch.isfinite(corrected_output).all()
    assert not torch.allclose(original_output, corrected_output, rtol=1e-5, atol=1e-6)
    # Repair changes no checkpoint keys/shapes; computational equivalence is NOT asserted.
    assert list(upstream.state_dict()) == list(corrected.state_dict())
    for key in upstream.state_dict():
        torch.testing.assert_close(upstream.state_dict()[key], corrected.state_dict()[key])


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("gamma", [0.0, 0.3])
def test_torch_actual_full_transformer_sample_isolation(batch, gamma):
    torch, module = _torch_module()
    from hift_research.corrections.tensor_layout import encoder_forward_corrected
    torch.manual_seed(20261023)
    torch.set_num_threads(1)
    upstream = module.Transformer(192, 6, 1, 2).eval()
    with torch.no_grad():
        upstream.encoder.layers[0].cross_attn.gamma.fill_(gamma)
    corrected = copy.deepcopy(upstream)
    for layer in corrected.encoder.layers:
        layer.forward = MethodType(encoder_forward_corrected, layer)
    src, srcc, tgt = (torch.randn(121, batch, 192) for _ in range(3))
    changed_src, changed_srcc, changed_tgt = (x.clone() for x in (src, srcc, tgt))
    changed_src[:, -1] = torch.randn_like(src[:, -1]) * 3
    changed_srcc[:, -1] = torch.randn_like(srcc[:, -1]) * 3
    changed_tgt[:, -1] = torch.randn_like(tgt[:, -1]) * 3
    with torch.no_grad():
        a = upstream(src, srcc, tgt)
        b = upstream(changed_src, changed_srcc, changed_tgt)
        ca = corrected(src, srcc, tgt)
        cb = corrected(changed_src, changed_srcc, changed_tgt)
    assert (a[:, 0] - b[:, 0]).abs().max().item() > 1e-5
    torch.testing.assert_close(ca[:, 0], cb[:, 0], rtol=0, atol=0)
    # Alone-vs-batched comparison also covers B=1 isolation for the repaired path.
    with torch.no_grad():
        alone = corrected(src[:, :1], srcc[:, :1], tgt[:, :1])
    torch.testing.assert_close(ca[:, :1], alone, rtol=1e-5, atol=2e-6)


def test_torch_actual_losses_negative_labels_and_zero_mask():
    torch = pytest.importorskip("torch")
    cfg = SimpleNamespace(TRAIN=SimpleNamespace(SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11, LABEL_RANGE=4))
    bbox = SimpleNamespace(x1=95, y1=95, x2=223, y2=223)
    original = _anchor().get(bbox, 11)
    corrected = pair_targets(_anchor(), bbox, 11, is_negative=True)
    iou_class = _execute_source("pysot/models/loss.py", "IOULoss", {"nn": torch.nn, "torch": torch})
    pred = torch.tensor([0., 0., 10., 10.]).repeat(1, 121, 1).requires_grad_()
    target = torch.tensor([20., 20., 30., 30.]).repeat(1, 121, 1)
    criterion = iou_class()
    assert criterion(pred, target, torch.tensor(original[3])).item() > 0.9
    zero_loss = criterion(pred, target, torch.tensor(corrected[3]))
    assert zero_loss.item() == 0
    zero_loss.backward()
    assert torch.count_nonzero(pred.grad) == 0
    logits = torch.zeros(1, 1, 11, 11, requires_grad=True)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor(corrected[2])[None])
    loss.backward()
    assert torch.all(logits.grad > 0)


def test_torch_actual_singleton_class_loss_is_dropped_and_repaired():
    torch = pytest.importorskip("torch")
    from hift_research.corrections.classification_loss import balanced_classification_loss
    actual_loss = _execute_source(
        "pysot/models/loss.py", "get_cls_loss", {"torch": torch, "F": torch.nn.functional},
    )
    pred = torch.log_softmax(torch.tensor([[2., -1.]], requires_grad=True), dim=1)
    label = torch.tensor([1])
    scalar_selection = label.eq(1).nonzero(as_tuple=False).squeeze()
    assert scalar_selection.ndim == 0
    assert actual_loss(pred, label, scalar_selection) == 0
    repaired = balanced_classification_loss(pred, label)
    torch.testing.assert_close(repaired, -0.5 * pred[0, 1])
    empty = balanced_classification_loss(pred, torch.tensor([-2]))
    assert empty.item() == 0 and empty.requires_grad
    empty.backward()


def test_torch_layout_helpers_roundtrip_and_invalid_dimensions():
    torch = pytest.importorskip("torch")
    from hift_research.corrections.tensor_layout import (
        encoder_forward_corrected, feature_to_sequence, sequence_to_feature,
    )
    for batch in (1, 2, 4):
        # Non-square map is supported by helpers, not by pinned encoder wrapper.
        x = torch.arange(batch * 192 * 6).view(batch, 192, 2, 3)
        torch.testing.assert_close(sequence_to_feature(feature_to_sequence(x), 2, 3), x)
    with pytest.raises(ValueError):
        sequence_to_feature(torch.zeros(5, 1, 192), 2, 3)
    with pytest.raises(ValueError):
        feature_to_sequence(torch.zeros(5, 1, 192))
    with pytest.raises(ValueError):
        encoder_forward_corrected(None, torch.zeros(6, 1, 192), torch.zeros(6, 1, 192))
