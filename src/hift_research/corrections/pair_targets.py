"""Low-level Wave1 target construction, not an identity-verified sampler.

New B2 data paths must use positive_targets.robust_pair_targets with explicit
object identities. This compatibility primitive cannot infer identities from a
bbox and does not certify that a requested negative is semantically valid.
"""
import numpy as np


def pair_targets(anchor_target, bbox, size, *, is_negative):
    """Keep upstream positives; mark known negative pairs all-background.

    The published source drops its sampled negative flag. This policy repairs
    that lost supervision with dense zero class targets and zero localization
    weights. Dense background instead of NEG_NUM sampling is an explicit new
    training policy, not a claim about author intent. The existing CE criterion
    retains its 0.5 negative-class factor when a batch has no positives.
    """
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    if not is_negative:
        return anchor_target.get(bbox, size)
    shape = (1, size, size)
    return (
        np.zeros(shape, dtype=np.float64),
        np.zeros((4, size, size), dtype=np.float32),
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    )
