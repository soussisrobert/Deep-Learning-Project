"""CPU-first actual graph and decode invariants; no GPU or full training."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hift_research.training.graph import (
    B2TrainingGraph, decode_regression, masked_decode_regression, masked_classification_log_probs,
    gradient_audit, optimizer_for, synthetic_batch,
)
from hift_research.training.profiling import run_profile
from hift_research.training.upstream import source_class


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    np.random.seed(20261023)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_dynamic_decode_matches_released_normal_math(batch):
    # Execute exact upstream method with solely .cuda() redirected to CPU; cfg
    # matches batch ONLY for reference. Adapter itself has no batch config.
    root = Path(__file__).resolve().parents[1]
    path = root / "third_party/HiFT/pysot/models/model_builder.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == "ModelBuilder")
    node = next(n for n in node.body if getattr(n, "name", None) == "getcentercuda")
    class CPU(ast.NodeTransformer):
        def visit_Call(self, n):
            if isinstance(n.func, ast.Attribute) and n.func.attr == "cuda":
                return self.visit(n.func.value)
            return self.generic_visit(n)
    module = ast.fix_missing_locations(ast.Module(body=[CPU().visit(node)], type_ignores=[]))
    namespace = {"t": torch, "np": np, "cfg": SimpleNamespace(TRAIN=SimpleNamespace(
        SEARCH_SIZE=287, BATCH_SIZE=batch, NUM_GPU=1))}
    exec(compile(module, str(path), "exec"), namespace)
    mapping = torch.rand(batch, 4, 11, 11)*1.8-0.9
    expected = namespace["getcentercuda"](None, mapping.clone())
    torch.testing.assert_close(decode_regression(mapping), expected, rtol=1e-5, atol=4e-5)


def test_decode_rectangular_grid_boundaries_gradients_no_mutation():
    values = torch.tensor([0.999, 1., -1., -0.999]).view(1, 4, 1, 1).requires_grad_()
    before = values.detach().clone()
    decoded = decode_regression(values)
    assert decoded.shape == (1, 1, 4) and decoded.isfinite().all()
    decoded.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 0].abs().sum() > 0  # .999 NOT blanket clamped to .99
    assert values.grad[0, 1].abs().sum() == values.grad[0, 2].abs().sum() == 0
    torch.testing.assert_close(values.detach(), before)
    zero = decode_regression(torch.zeros(4, 4, 2, 3))
    torch.testing.assert_close(zero[0, :, 0], torch.tensor([63., 79., 95., 63., 79., 95.]))
    torch.testing.assert_close(zero[0, :, 1], torch.tensor([63., 63., 63., 79., 79., 79.]))


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_actual_train_graph_positive_backward_and_sgd(batch):
    model = B2TrainingGraph().train()
    assert not model.backbone.layer1[1].training and not model.backbone.layer2[1].training
    assert model.backbone.layer3[1].training
    data = synthetic_batch(batch, negative=[False]*batch)
    optimizer = optimizer_for(model)
    before = model.grader.cls1.weight.detach().clone()
    losses = model(*data)
    assert all(bool(torch.isfinite(loss)) for loss in losses.values())
    losses["total_loss"].backward()
    audit = gradient_audit(model)
    assert not audit["unexpected_missing"] and not audit["nonfinite"] and not audit["unexpected_zero"]
    assert len(audit["expected_dormant"]) == 4
    assert len(audit["expected_initial_gamma_zero"]) == 4
    assert audit["active_nonzero"] and audit["frozen"]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()
    assert not torch.equal(before, model.grader.cls1.weight)


def test_known_negative_has_zero_localization_gradient_and_finite_classifier_gradient():
    model = B2TrainingGraph()
    data = synthetic_batch(1, negative=[True])
    assert data[2]["weightxff"].sum() == 0
    losses = model(*data)
    assert losses["loc_loss"].item() == 0
    losses["total_loss"].backward()
    assert all(p.grad is None or not p.grad.ne(0).any() for p in model.grader.convloc.parameters())
    assert model.grader.cls1.weight.grad.ne(0).any()
    assert model.grader.cls2.weight.grad.ne(0).any()
    assert not gradient_audit(model)["nonfinite"]


def test_real_heads_representative_shapes_and_strict_key_identity():
    model = B2TrainingGraph().eval()
    with torch.no_grad():
        template, search, _ = synthetic_batch(1)
        z, x = model.features(template, search)
        heads = model.grader(x, z)
    assert [tuple(h.shape) for h in heads] == [(1, 4, 11, 11), (1, 2, 11, 11), (1, 1, 11, 11)]
    clone = B2TrainingGraph()
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone.state_dict().keys() == model.state_dict().keys()


def test_raw_regression_mask_precedes_nonlinear_decode_with_finite_backward():
    mapping = torch.full((1, 4, 2, 2), float("nan"), requires_grad=True)
    target = torch.full((1, 4, 2, 2), float("inf"))
    weight = torch.zeros(1, 1, 2, 2)
    decoded = masked_decode_regression(mapping, weight)
    assert decoded.isfinite().all()
    decoded.sum().backward()
    assert torch.isfinite(mapping.grad).all() and not mapping.grad.ne(0).any()
    model = B2TrainingGraph()
    loc = torch.full((1, 4, 11, 11), float("nan"), requires_grad=True)
    cls1 = torch.zeros(1, 2, 11, 11, requires_grad=True)
    cls2 = torch.zeros(1, 1, 11, 11, requires_grad=True)
    _, _, targets = synthetic_batch(1, negative=[True])
    targets["labelxff"].fill_(float("nan"))
    losses = model.losses((loc, cls1, cls2), targets)
    losses["total_loss"].backward()
    assert losses["loc_loss"].item() == 0
    assert torch.isfinite(loc.grad).all() and not loc.grad.ne(0).any()
    weight.fill_(1)
    with pytest.raises(ValueError, match="active regression"):
        masked_decode_regression(mapping, weight)


def test_ignored_cls1_nan_is_selected_before_logsoftmax_at_composition_level():
    model = B2TrainingGraph()
    _, _, targets = synthetic_batch(1, negative=[True])
    targets["label_cls1"].fill_(-1)
    cls1 = torch.full((1, 2, 11, 11), float("nan"), requires_grad=True)
    loc = torch.zeros(1, 4, 11, 11, requires_grad=True)
    cls2 = torch.zeros(1, 1, 11, 11, requires_grad=True)
    losses = model.losses((loc, cls1, cls2), targets)
    assert losses["cls_loss1"].item() == 0
    losses["total_loss"].backward()
    assert torch.isfinite(cls1.grad).all() and not cls1.grad.ne(0).any()
    targets["label_cls1"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="active classification"):
        model.losses((loc, cls1, cls2), targets)
    cls1_finite = torch.randn(1, 2, 11, 11)
    torch.testing.assert_close(masked_classification_log_probs(cls1_finite, targets["label_cls1"])[0, 0, 0],
                              torch.log_softmax(cls1_finite.permute(0, 2, 3, 1), -1)[0, 0, 0])
    targets["labelcls2"].fill_(2)
    with pytest.raises(ValueError, match="targets within"):
        model.losses((loc, torch.zeros_like(cls1), cls2), targets)


def test_profile_validation_and_no_accidental_cuda(tmp_path):
    with pytest.raises(ValueError, match="token"):
        run_profile(tmp_path/"cuda", device="cuda")
    assert not (tmp_path/"cuda").exists()
    with pytest.raises(ValueError, match="repeats"):
        run_profile(tmp_path/"few", repeats=4)
    output = tmp_path/"exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_profile(output)


@pytest.mark.parametrize("bad", [torch.zeros(1, 3, 2, 2), torch.zeros(0, 4, 2, 2), torch.ones(1, 4, 2, 2, dtype=torch.int64)])
def test_malformed_decode(bad):
    with pytest.raises(ValueError):
        decode_regression(bad)
