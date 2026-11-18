"""Stage-4 S4-GAP search-frame sampler for positive pairs (template frame tf, search frame from the same track).

Default (long_gap_probability=0): a valid frame != tf within max_gap of tf, one np.random.choice draw, i.e.
exactly the sampler used through stage 3 (same random-number stream). With long_gap_probability=p > 0, one extra
np.random.random() draw first; with probability p the search frame is drawn uniformly over ALL valid frames != tf
of the track (no gap limit). Uses the caller's NumPy global RNG (TrackPairDataset seeds it per index)."""
from __future__ import annotations

import numpy as np


def check_gap_policy(max_gap, long_gap_probability):
    if int(max_gap) != max_gap or max_gap < 1:
        raise ValueError("max_gap must be a positive integer")
    if not 0 <= long_gap_probability <= 1:
        raise ValueError("long_gap_probability must be in [0, 1]")


def search_frame(valid, tf, *, max_gap=100, long_gap_probability=0.0):
    """Index of the search frame, or None when the track has no valid partner frame in the drawn window."""
    valid = np.asarray(valid)
    unlimited = bool(long_gap_probability and np.random.random() < long_gap_probability)
    keep = valid != tf
    if not unlimited:
        keep &= np.abs(valid-tf) <= max_gap
    near = valid[keep]
    if not len(near):
        return None
    return int(np.random.choice(near))
