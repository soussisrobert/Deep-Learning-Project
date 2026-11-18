"""Explicit W4-only CPU NCHW storage normalization for the immutable source's view().

Real HWC->CHW NumPy transposes can stack into Torch channels-last-strided NCHW.
Only storage layout changes: no channel permutation, resize, normalization,
interpolation, dtype/range conversion, or pixel-value alteration.
"""
import torch


def canonicalize_model_inputs(inputs):
    if inputs is None:
        return None  # honest all-rejected batch
    if not isinstance(inputs, dict):
        raise ValueError("model inputs must be a mapping or None")
    result = dict(inputs)
    sizes = {"template": 127, "search": 287}
    batches = []
    for name, size in sizes.items():
        value = inputs.get(name)
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type != "cpu"
                or value.dtype != torch.float32 or value.ndim != 4 or value.shape[1:] != (3,size,size)):
            raise ValueError(f"{name} must be CPU FP32 strided NCHW [N,3,{size},{size}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite {name} pixels")
        batches.append(value.shape[0])
        canonical = value.contiguous(memory_format=torch.contiguous_format)
        if not canonical.is_contiguous(memory_format=torch.contiguous_format) or not torch.equal(canonical, value):
            raise ValueError("layout canonicalization changed logical pixels")
        result[name] = canonical
    if batches[0] != batches[1]:
        raise ValueError("template/search batch sizes disagree")
    return result
