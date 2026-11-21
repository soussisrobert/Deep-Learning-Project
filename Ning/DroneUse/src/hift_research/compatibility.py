"""CPU formula probes, NOT execution of a historical PyTorch runtime.

Attention follows the unmasked, equal-embedding-dimension evaluation path in
PyTorch v1.6.0 torch/nn/functional.py. Normalization follows the biased-variance
formula documented in v1.6.0 torch/nn/modules/normalization.py. All arithmetic
runs on the *installed* torch CPU kernels. No upstream modules are modified.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HISTORICAL_SOURCES = {
    "functional.py": {
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v1.6.0/torch/nn/functional.py",
        "sha256": "08220ea822750dff9aff442441b5f72fef564af7dcd948f6a56e98294a83ee45",
    },
    "normalization.py": {
        "url": "https://raw.githubusercontent.com/pytorch/pytorch/v1.6.0/torch/nn/modules/normalization.py",
        "sha256": "36d803e3842d76c2c2a7f66ee2309efd51cbc5853d5e595c49c283466fc490b5",
    },
}


def _cpu_float(*values: Tensor) -> None:
    for value in values:
        if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
            raise ValueError("Compatibility references require CPU float32/float64 tensors")
        if not torch.isfinite(value).all():
            raise ValueError("Compatibility references require finite tensors")


def historical_mha_formula(
    module: nn.MultiheadAttention, query: Tensor, key: Tensor, value: Tensor,
) -> tuple[Tensor, Tensor]:
    """Restricted v1.6 forward formula with averaged weights and dropout disabled.

    Supports the four equality branches relevant to HiFT: Q=K=V, K=V,
    Q=K!=V, and all-distinct Q/K/V. Equality tests deliberately use torch.equal
    as the historical source did (modern torch may branch on tensor identity).
    Unsupported options fail rather than silently changing the reference.
    """
    if module.training:
        raise ValueError("Reference is evaluation-only; call module.eval()")
    if (module.batch_first or not module._qkv_same_embed_dim or module.add_zero_attn
            or module.bias_k is not None or module.bias_v is not None):
        raise ValueError("Reference requires sequence-first equal-dimension unaugmented MHA")
    if any(x.ndim != 3 for x in (query, key, value)):
        raise ValueError("Expected sequence-first rank-3 Q/K/V")
    length, batch, dim = query.shape
    if (dim != module.embed_dim or key.shape != value.shape
            or key.shape[1:] != (batch, dim) or min(length, batch, key.shape[0]) < 1):
        raise ValueError("Invalid attention shapes")
    tensors = [query, key, value, *module.parameters()]
    _cpu_float(*tensors)
    if any(x.dtype != query.dtype for x in tensors):
        raise ValueError("Q/K/V and module parameters must have the same dtype")
    weight, bias = module.in_proj_weight, module.in_proj_bias
    if torch.equal(query, key) and torch.equal(key, value):
        q, k, v = F.linear(query, weight, bias).chunk(3, dim=-1)
    elif torch.equal(key, value):
        q = F.linear(query, weight[:dim], None if bias is None else bias[:dim])
        k, v = F.linear(key, weight[dim:], None if bias is None else bias[dim:]).chunk(2, -1)
    else:
        chunks = weight.chunk(3, dim=0)
        biases = (None, None, None) if bias is None else bias.chunk(3, dim=0)
        q, k, v = [F.linear(x, w, b) for x, w, b in zip((query, key, value), chunks, biases)]
    head_dim = dim // module.num_heads
    q = q * float(head_dim) ** -0.5
    q = q.contiguous().view(length, batch * module.num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, batch * module.num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, batch * module.num_heads, head_dim).transpose(0, 1)
    weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
    result = torch.bmm(weights, v)
    result = result.transpose(0, 1).contiguous().view(length, batch, dim)
    result = F.linear(result, module.out_proj.weight, module.out_proj.bias)
    averaged = weights.view(batch, module.num_heads, length, key.shape[0]).sum(1)
    return result, averaged / module.num_heads


def layer_norm_formula(
    x: Tensor, normalized_shape: tuple[int, ...], weight: Tensor | None = None,
    bias: Tensor | None = None, eps: float = 1e-5,
) -> Tensor:
    """Explicit mean + biased variance, not torch.layer_norm."""
    _cpu_float(x, *[v for v in (weight, bias) if v is not None])
    shape = tuple(normalized_shape)
    if (not shape or len(shape) > x.ndim or tuple(x.shape[-len(shape):]) != shape
            or any(n < 1 for n in shape) or not math.isfinite(eps) or eps <= 0):
        raise ValueError("Invalid normalized shape or epsilon")
    for parameter in (weight, bias):
        if parameter is not None and (tuple(parameter.shape) != shape or parameter.dtype != x.dtype):
            raise ValueError("Invalid LayerNorm affine parameters")
    dims = tuple(range(x.ndim - len(shape), x.ndim))
    centered = x - x.mean(dim=dims, keepdim=True)
    result = centered * torch.rsqrt(centered.square().mean(dim=dims, keepdim=True) + eps)
    if weight is not None:
        result = result * weight
    return result if bias is None else result + bias


def group_norm_formula(
    x: Tensor, groups: int, weight: Tensor | None = None,
    bias: Tensor | None = None, eps: float = 1e-5,
) -> Tensor:
    """Normalize each sample/group over channels and spatial dimensions."""
    _cpu_float(x, *[v for v in (weight, bias) if v is not None])
    if (x.ndim < 2 or groups < 1 or x.shape[1] % groups or x.numel() < 1
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError("Invalid GroupNorm shape, groups or epsilon")
    for parameter in (weight, bias):
        if parameter is not None and (parameter.shape != (x.shape[1],) or parameter.dtype != x.dtype):
            raise ValueError("Invalid GroupNorm affine parameters")
    grouped = x.reshape(x.shape[0], groups, -1)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    result = (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)).reshape_as(x)
    affine_shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
    if weight is not None:
        result = result * weight.view(affine_shape)
    return result if bias is None else result + bias.view(affine_shape)


def _errors(actual: Tensor, reference: Tensor, atol: float, rtol: float) -> dict:
    difference = (actual - reference).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {
        "max_abs_error": difference.max().item(),
        "rms_error": difference.square().mean().sqrt().item(),
        "reference_max_abs": reference.abs().max().item(),
        "atol": atol, "rtol": rtol, "finite": finite,
        "passed": finite and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
    }


def run_formula_checks(seed: int = 20261023) -> list[dict]:
    """Deterministic CPU-only HiFT-shape evaluation probes; no global RNG changes."""
    rows = []
    for dtype in (torch.float32, torch.float64):
        atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (1e-12, 1e-11)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for batch in (1, 2, 4):
            # Constructor initialization is overwritten, but still isolate its RNG consumption.
            with torch.random.fork_rng(devices=[]):
                module = nn.MultiheadAttention(192, 6, dropout=0.1, dtype=dtype, device="cpu")
            with torch.no_grad():
                for parameter in module.parameters():
                    scale = 0.05 if parameter.ndim > 1 else 0.02
                    parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=dtype, device="cpu") * scale)
            module.eval()
            q, k, v = [torch.randn(121, batch, 192, generator=generator, dtype=dtype, device="cpu") for _ in range(3)]
            modes = {"self": (q, q, q), "cross_k_equals_v": (q, k, k),
                     "hift_encoder_q_equals_k": (q, q.clone(), v), "distinct": (q, k, v)}
            with torch.no_grad():
                for mode, inputs in modes.items():
                    reference, reference_weights = historical_mha_formula(module, *inputs)
                    actual, weights = module(*inputs, need_weights=True)
                    for name, modern, expected in (("output", actual, reference),
                                                    ("averaged_weights", weights, reference_weights)):
                        rows.append({"operation": "mha", "mode": mode, "quantity": name,
                                     "shape": [121, batch, 192], "dtype": str(dtype),
                                     **_errors(modern, expected, atol, rtol)})
                    # Supplementary modern SDPA path; upstream defaults to need_weights=True.
                    sdpa, _ = module(*inputs, need_weights=False)
                    rows.append({"operation": "mha_sdpa_supplementary", "mode": mode,
                                 "quantity": "output", "shape": [121, batch, 192],
                                 "dtype": str(dtype), **_errors(sdpa, reference, atol, rtol)})
                for kind, shape in (("layer_norm", (121, batch, 192)),
                                    ("group_norm", (batch, 192, 11, 11))):
                    noise = torch.randn(shape, generator=generator, dtype=dtype, device="cpu")
                    for regime, x in (("normal", noise), ("zero", noise * 0),
                                      ("near_constant", noise * 1e-6 + 1e-3),
                                      ("large_amplitude", noise * 1e4)):
                        for affine in (False, True):
                            w = torch.randn(192, generator=generator, dtype=dtype, device="cpu") if affine else None
                            b = torch.randn(192, generator=generator, dtype=dtype, device="cpu") if affine else None
                            if kind == "layer_norm":
                                reference = layer_norm_formula(x, (192,), w, b)
                                actual = F.layer_norm(x, (192,), w, b, 1e-5)
                            else:
                                reference = group_norm_formula(x, 32, w, b)
                                actual = F.group_norm(x, 32, w, b, 1e-5)
                            rows.append({"operation": kind, "regime": regime, "affine": affine,
                                         "shape": list(shape), "dtype": str(dtype),
                                         **_errors(actual, reference, atol, rtol)})
    return rows
