"""Modern CPU kernels versus historical formula; no old runtime/GPU is used."""
import io
from pathlib import Path
import runpy

import pytest

torch = pytest.importorskip("torch")
from hift_research.compatibility import (
    HISTORICAL_SOURCES, group_norm_formula, historical_mha_formula,
    layer_norm_formula, run_formula_checks,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_all_hift_shape_formula_cases_and_reproducibility():
    rng_before = torch.random.get_rng_state().clone()
    rows = run_formula_checks()
    assert rows == run_formula_checks()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(rows) == 168
    assert all(row["passed"] and row["finite"] for row in rows)
    mha = [row for row in rows if row["operation"] == "mha"]
    assert {row["shape"][1] for row in mha} == {1, 2, 4}
    assert {row["dtype"] for row in mha} == {"torch.float32", "torch.float64"}
    assert {row["mode"] for row in mha} == {
        "self", "cross_k_equals_v", "hift_encoder_q_equals_k", "distinct",
    }


@pytest.mark.parametrize("bias", [False, True])
def test_attention_uniform_logits_equal_mean_value(bias):
    module = torch.nn.MultiheadAttention(4, 2, bias=bias, dtype=torch.float64).eval()
    with torch.no_grad():
        module.in_proj_weight.copy_(torch.eye(4, dtype=torch.float64).repeat(3, 1))
        module.out_proj.weight.copy_(torch.eye(4, dtype=torch.float64))
        if bias:
            module.in_proj_bias.zero_()
            module.out_proj.bias.zero_()
    q = torch.zeros(2, 1, 4, dtype=torch.float64)
    k = torch.arange(12, dtype=torch.float64).reshape(3, 1, 4)
    v = k + 7
    result, weights = historical_mha_formula(module, q, k, v)
    torch.testing.assert_close(result, v.mean(0, keepdim=True).expand_as(q))
    torch.testing.assert_close(weights, torch.full((1, 2, 3), 1 / 3, dtype=torch.float64))


def test_saturated_logits_remain_finite_and_match_modern_default():
    generator = torch.Generator().manual_seed(56)
    module = torch.nn.MultiheadAttention(192, 6, dtype=torch.float64).eval()
    q = torch.randn(121, 2, 192, generator=generator, dtype=torch.float64) * 1e3
    with torch.no_grad():
        actual, weights = module(q, q, q)
        reference, reference_weights = historical_mha_formula(module, q, q, q)
    assert torch.isfinite(reference).all() and torch.isfinite(reference_weights).all()
    torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(weights, reference_weights, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 121, dtype=torch.float64))


@pytest.mark.parametrize("kind", ["layer", "group"])
def test_norm_biased_variance_and_epsilon_hand_calculation(kind):
    x = torch.tensor([1., 3.], dtype=torch.float64)
    expected = torch.tensor([-1., 1.], dtype=torch.float64) / (1 + 1e-5) ** 0.5
    actual = (layer_norm_formula(x, (2,)) if kind == "layer"
              else group_norm_formula(x.reshape(1, 2, 1, 1), 1).flatten())
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("kind", ["layer", "group"])
def test_norm_constant_input_with_affine(kind):
    x = torch.full((2, 4, 3, 3), 4., dtype=torch.float64)
    weight = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    bias = weight + 10
    if kind == "layer":
        actual = layer_norm_formula(x.permute(0, 2, 3, 1), (4,), weight, bias)
        expected = bias.expand_as(actual)
    else:
        actual = group_norm_formula(x, 2, weight, bias)
        expected = bias.view(1, 4, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["mha", "layer", "group"])
def test_nonfinite_inputs_rejected(operation):
    x = torch.ones(3, 2, 4)
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        if operation == "mha":
            historical_mha_formula(torch.nn.MultiheadAttention(4, 2).eval(), x, x, x)
        elif operation == "layer":
            layer_norm_formula(x, (4,))
        else:
            group_norm_formula(x, 2)


@pytest.mark.parametrize("setting", ["training", "batch_first", "add_zero_attn", "add_bias_kv", "kdim"])
def test_unsupported_attention_settings_fail_closed(setting):
    kwargs = {} if setting == "training" else {setting: 2 if setting == "kdim" else True}
    module = torch.nn.MultiheadAttention(4, 2, **kwargs)
    if setting != "training":
        module.eval()
    x = torch.ones(3, 2, 4)
    with pytest.raises(ValueError):
        historical_mha_formula(module, x, x, x)


def test_bad_attention_shape_and_dtype_rejected():
    module = torch.nn.MultiheadAttention(4, 2).eval()
    x = torch.ones(3, 2, 4)
    with pytest.raises(ValueError, match="rank-3"):
        historical_mha_formula(module, x[0], x, x)
    with pytest.raises(ValueError, match="shapes"):
        historical_mha_formula(module, x, x[:, :1], x[:, :1])
    with pytest.raises(ValueError, match="dtype"):
        historical_mha_formula(module, x.double(), x.double(), x.double())
    with pytest.raises(ValueError, match="float32/float64"):
        historical_mha_formula(module.half(), x.half(), x.half(), x.half())


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf")])
def test_bad_normalization_epsilon_rejected(eps):
    x = torch.ones(2, 4)
    with pytest.raises(ValueError):
        layer_norm_formula(x, (4,), eps=eps)
    with pytest.raises(ValueError):
        group_norm_formula(x, 2, eps=eps)


def test_bad_normalization_shapes_and_parameters_rejected():
    x = torch.ones(2, 4, 3, 3)
    for shape in ((), (4,), (2, 4, 3, 3, 1)):
        with pytest.raises(ValueError):
            layer_norm_formula(x, shape)
    with pytest.raises(ValueError):
        layer_norm_formula(x, (3, 3), torch.ones(3))
    with pytest.raises(ValueError):
        group_norm_formula(x, 3)
    with pytest.raises(ValueError):
        group_norm_formula(x, 2, torch.ones(3))


def test_official_source_pins_are_explicit():
    assert set(HISTORICAL_SOURCES) == {"functional.py", "normalization.py"}
    for specification in HISTORICAL_SOURCES.values():
        assert specification["url"].startswith("https://raw.githubusercontent.com/pytorch/pytorch/v1.6.0/")
        assert len(specification["sha256"]) == 64


def test_source_fetch_rejects_changed_bytes_without_importing(tmp_path, monkeypatch):
    cli = runpy.run_path(str(ROOT / "scripts/check_compatibility.py"))
    monkeypatch.setattr(cli["urllib"].request, "urlopen", lambda *a, **kw: io.BytesIO(b"not trusted"))
    with pytest.raises(ValueError, match="identity mismatch"):
        cli["fetch_historical_sources"](tmp_path / "source")
    assert list((tmp_path / "source").iterdir()) == []


def test_cli_refuses_existing_run_and_invalid_threads(tmp_path):
    cli = runpy.run_path(str(ROOT / "scripts/check_compatibility.py"))
    with pytest.raises(FileExistsError):
        cli["main"](["--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli["main"](["--output", str(tmp_path / "never-created"), "--threads", "0"])
    assert not (tmp_path / "never-created").exists()
