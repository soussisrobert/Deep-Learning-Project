"""Opt-in B2 layout repair; never apply silently to an official-checkpoint B0/B1.

Parameters and state_dict keys are unchanged, but the learned computation changes,
including B=1. These functions deliberately do not import torch at module load.
"""
from math import isqrt


def sequence_to_feature(sequence, height, width):
    """Convert torch [S,B,C] to [B,C,H,W] without reinterpreting axes."""
    if sequence.ndim != 3 or height < 1 or width < 1:
        raise ValueError("expected [S,B,C] and positive spatial dimensions")
    if sequence.shape[0] != height * width:
        raise ValueError("sequence length must equal height * width")
    return sequence.permute(1, 2, 0).contiguous().view(
        sequence.shape[1], sequence.shape[2], height, width
    )


def feature_to_sequence(feature):
    """Convert torch [B,C,H,W] to [H*W,B,C]."""
    if feature.ndim != 4:
        raise ValueError("expected [B,C,H,W]")
    return feature.flatten(2).permute(2, 0, 1)


def encoder_forward_corrected(layer, src, srcc, src_mask=None, src_key_padding_mask=None):
    """Unbound forward replacement for the pinned TransformerEncoderLayer.

    Caller must explicitly bind this to a *research copy* of an upstream layer.
    Only the proven spatial conversion error is repaired. Dropout, parameter
    names, normalizations and attention arguments retain upstream semantics.
    """
    if src.ndim != 3 or tuple(src.shape) != tuple(srcc.shape):
        raise ValueError("src and srcc must have equal [S,B,C] shapes")
    side = isqrt(src.shape[0])
    if side * side != src.shape[0]:
        raise ValueError("pinned encoder requires a square spatial feature map")
    query = layer.norm0(src + srcc)
    src2 = layer.self_attn(
        query, query, src, attn_mask=src_mask,
        key_padding_mask=src_key_padding_mask,
    )[0]
    src = layer.norm1(src + layer.dropout1(src2))
    src = feature_to_sequence(layer.cross_attn(
        sequence_to_feature(src, side, side),
        sequence_to_feature(srcc, side, side),
    ))
    src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
    return layer.norm2(src + layer.dropout2(src2))
