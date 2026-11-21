"""Shared statistics used for every interval in the paper.

All intervals are 95% percentile bootstrap intervals of a mean paired change.
- video_bootstrap: resample videos with replacement (night, CPU).
- cluster_bootstrap: resample source clusters with replacement, keeping every
  member sequence, then take the sequence-macro mean (domain).
- two_level_bootstrap: resample videos AND calibration draws (CPU TEST-40).

Each function takes its own seed so every paper number can be regenerated exactly.
"""
from __future__ import annotations

import numpy as np


def video_bootstrap(diffs, seed, n_boot=10_000, vectorised=True):
    """Mean of per-video paired differences and its 95% percentile interval.

    vectorised=True draws all index rows at once (rng.integers(0, n, (B, n))), the
    form used by the night and CPU analysis scripts.
    """
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def cluster_bootstrap(diffs, clusters, seed, n_boot=10_000):
    """Paired cluster bootstrap: resample clusters, keep all member sequences."""
    d = np.asarray(diffs, float)
    clusters = list(clusters)
    ids = sorted(set(clusters))
    members = {c: [i for i, x in enumerate(clusters) if x == c] for c in ids}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids))
        rows = [i for c in pick for i in members[c]]
        means.append(d[rows].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def two_level_bootstrap(draws, base, seed=1, n_boot=10_000):
    """draws: (n_draws, n_videos) int8 scores; base: (n_videos,) fp32 scores.

    Each replicate resamples videos and calibration draws with replacement.
    """
    draws = np.asarray(draws, float)
    base = np.asarray(base, float)
    k, n = draws.shape
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        v = rng.integers(0, n, n)
        dr = rng.integers(0, k, k)
        reps.append(draws[dr][:, v].mean() - base[v].mean())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(draws.mean(0).mean() - base.mean()), float(lo), float(hi)
