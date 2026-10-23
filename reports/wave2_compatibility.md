# Wave 2: CPU historical-formula compatibility probe

**Measured 2026-10-23; formula agreement, not historical-runtime parity.** No GPU computation, dataset, checkpoint, training, upstream modification or baseline-math change. This independent diagnostic does not block separately validated real tracking and does not establish benchmark/checkpoint reproduction.

## What was compared

[Reference module](../src/hift_research/compatibility.py), [CLI](../scripts/check_compatibility.py) and [tests](../tests/test_compatibility.py) compare installed **PyTorch 2.10.0+cu128 CPU** operations against explicitly written formulas:

- Official [v1.6.0 functional source](https://raw.githubusercontent.com/pytorch/pytorch/v1.6.0/torch/nn/functional.py), `multi_head_attention_forward`, lines 3854–4142: packed/separate QKV projection, query scaling by `head_dim**-0.5`, head reshape, `bmm`, softmax, `bmm`, output projection and head-averaged attention weights. Equality branches follow historical `torch.equal`, not modern identity shortcuts. Only unmasked, equal-dimensional, sequence-first evaluation is supported; unsupported options fail closed.
- Official [v1.6.0 normalization source](https://raw.githubusercontent.com/pytorch/pytorch/v1.6.0/torch/nn/modules/normalization.py), LayerNorm/GroupNorm: explicit mean subtraction, **biased variance**, epsilon inside the square root and optional affine parameters. The historical functional wrappers call native torch normalization; they are **not independent historical kernels**.

Both official files were byte-SHA256 verified and cached as inert text in the [final run's source evidence](../artifacts/compatibility/wave2-formula-cpu-20261023b/historical_source/functional.py.txt); no downloaded source was imported, evaluated or executed. Full-source pins:

| Source | Bytes | SHA256 |
|---|---:|---|
| functional.py | 191913 | `08220ea822750dff9aff442441b5f72fef564af7dcd948f6a56e98294a83ee45` |
| normalization.py | 9539 | `36d803e3842d76c2c2a7f66ee2309efd51cbc5853d5e595c49c283466fc490b5` |

## Coverage and measured errors

Seed `20261023`; CPU-only tensors; one thread; deterministic algorithms enabled; Windows 10.0.19044, Python 3.13.0, Intel64 Family 6 Model 183 Stepping 1, AVX2/MKL. CUDA is a **build capability only** here and remained uninitialized. [Final immutable manifest](../artifacts/compatibility/wave2-formula-cpu-20261023b/manifest.json) records full environment, git revision/dirty diff hash/source-file hashes, pinned upstream SHA and every numeric comparison; the [initial run](../artifacts/compatibility/wave2-formula-cpu-20261023a/manifest.json) is retained rather than overwritten.

- MHA: `[121, B, 192]`, `B ∈ {1,2,4}`, six heads, FP32/FP64; self attention, cross `K=V`, HiFT encoder `Q=K≠V`, and distinct Q/K/V. These cover the [encoder call](../third_party/HiFT/pysot/models/utile/tran.py#L242-L246) and [decoder calls](../third_party/HiFT/pysot/models/utile/tran.py#L319-L324). Configured dropout is 0.1 but **evaluation disables it**. Output and head-averaged weights are checked. Modern `need_weights=False`/SDPA is supplemental; pinned HiFT uses the default `need_weights=True`.
- LayerNorm: `[121,B,192]`, last-dimension normalization. GroupNorm: `[B,192,11,11]`, 32 groups. Both use `eps=1e-5`, affine on/off, normal random, exact zero, near-constant (`1e-3 + 1e-6*noise`) and large-amplitude (`1e4*noise`) inputs.

| Operation | Cases per dtype | FP32 maximum absolute error | FP64 maximum absolute error |
|---|---:|---:|---:|
| MHA default output/weights | 24 | 0 | 0 |
| MHA supplementary SDPA output | 12 | 1.4901161e-7 | 2.7755576e-16 |
| LayerNorm | 24 | 1.9073486e-6 | 3.5527137e-15 |
| GroupNorm | 24 | 1.9073486e-6 | 3.5527137e-15 |

**All 168 comparisons passed.** Fixed elementwise tolerances: FP32 `atol=3e-6, rtol=3e-5`; FP64 `atol=1e-12, rtol=1e-11`. Maximum errors above are absolute, not near-zero unstable relative errors. Zero MHA differences are unsurprising: the compared evaluation implementations largely share modern primitive operations. They are not evidence of bitwise historical-kernel agreement.

**59 tests passed** in the targeted combined run. Additional compatibility tests cover hand-calculated biased variance/epsilon, zero variance plus affine, saturated attention logits, repeatability/global CPU RNG preservation, malformed shapes/dtypes/nonfinite inputs, unsupported MHA settings, source-hash mismatch rejection and refusal to overwrite a run. Existing [crop geometry tests](../tests/test_geometry_scores.py) and [actual upstream/OpenCV crop pixel fixtures](../tests/test_inference_contract.py#L91-L117) were reused unchanged, including padding and fractional centers. This validates current crop behavior, **not parity with a historical OpenCV binary**; no duplicate crop implementation was added.

## Reproduce and interpret

```bash
.venv/Scripts/python.exe scripts/check_compatibility.py --output artifacts/compatibility/NEW_UNIQUE_RUN --fetch-historical-source
.venv/Scripts/python.exe -m pytest tests/test_compatibility.py tests/test_geometry_scores.py tests/test_inference_contract.py -q
```

The output directory must be new. Network retrieval is optional on later runs; without it, expected source pins are recorded but the manifest explicitly marks them unverified for that run. The stored full sources in the measured runs provide offline audit evidence.

**Conclusion:** the selected HiFT-shaped operations agree with reviewed v1.6 formulas under current CPU kernels at the stated tolerances. Actual torch 1.6 execution was **not attempted**: the present Python 3.13 environment is not a supported historical runtime. A genuine cross-runtime test would require a separate compatible historical Python/torch environment and common serialized inputs/weights; no global downgrade or old-runtime installation was made. CUDA/cuDNN/MKL version drift, gradients, mixed precision, training dropout, arbitrary masks, catastrophic high-offset cancellation, full-model accumulation and real-sequence behavior remain outside this evidence. No inference/training correction is justified solely by these passing synthetic checks.
