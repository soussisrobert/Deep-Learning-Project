import numpy as np
import pytest

from hift_research.data_contract.pipeline import numpy_seed
from hift_research.lasot.dataset import TrackPairDataset
from hift_research.lasot.gap_policy import check_gap_policy, search_frame
from test_distractor_negatives import _mot_video, _rows


def _stage3_sampler(valid, tf, max_gap):  # the inline sampler of dataset.py through stage 3
    near = valid[(np.abs(valid-tf) <= max_gap) & (valid != tf)]
    return None if not len(near) else int(np.random.choice(near))


def test_default_is_the_stage3_sampler_and_rng_stream():
    rng = np.random.default_rng(0)
    for seed in range(300):
        valid = np.sort(rng.choice(2000, size=int(rng.integers(1, 400)), replace=False))
        tf = int(rng.choice(valid))
        gap = int(rng.integers(1, 300))
        with numpy_seed(seed):
            expected, after_expected = _stage3_sampler(valid, tf, gap), np.random.random()
        with numpy_seed(seed):
            got, after_got = search_frame(valid, tf, max_gap=gap), np.random.random()
        assert got == expected and after_got == after_expected


def test_bounds_and_validity():
    valid = np.array([0, 3, 50, 120, 400, 401, 999])
    for seed in range(200):
        with numpy_seed(seed):
            short = search_frame(valid, 120, max_gap=100)
            long = search_frame(valid, 120, max_gap=100, long_gap_probability=1.0)
        assert short in {50} and long in set(valid.tolist())-{120}
    with numpy_seed(1):
        seen = {search_frame(valid, 120, max_gap=100, long_gap_probability=1.0) for _ in range(300)}
    assert seen == set(valid.tolist())-{120}  # uniform over the whole track, never tf itself


def test_mixture_share_and_single_frame_tracks():
    valid = np.arange(0, 3000)
    with numpy_seed(5):
        far = [abs(search_frame(valid, 1500, max_gap=100, long_gap_probability=0.5)-1500) > 100 for _ in range(4000)]
    # P(far | long draw) = 2799/2999 -> expected share 0.5*0.933
    assert abs(np.mean(far)-0.5*2799/2999) < 0.03
    for p in (0.0, 0.5, 1.0):
        with numpy_seed(2):
            assert search_frame(np.array([7]), 7, max_gap=100, long_gap_probability=p) is None
    with numpy_seed(3):
        assert search_frame(np.array([0, 500]), 0, max_gap=100) is None
        assert search_frame(np.array([0, 500]), 0, max_gap=100, long_gap_probability=1.0) == 500


@pytest.mark.parametrize("max_gap,p", [(0, 0.0), (-3, 0.0), (2.5, 0.0), (100, -0.1), (100, 1.5), (100, float("nan"))])
def test_invalid_policy_rejected(max_gap, p):
    with pytest.raises(ValueError):
        check_gap_policy(max_gap, p)


def test_dataset_default_pairs_unchanged_and_long_gap_reaches_far_frames(tmp_path):
    tracks = (_mot_video(tmp_path, "uav01", _rows(1, 4, 100, frames=range(1, 61)), frames=60)
              + _mot_video(tmp_path, "uav02", _rows(1, 4, 300, frames=range(1, 61)), frames=60))
    a = TrackPairDataset(tracks, length=40, seed=7, negative_probability=0.2, max_gap=5)
    b = TrackPairDataset(tracks, length=40, seed=7, negative_probability=0.2, max_gap=5, long_gap_probability=0.0)
    for i in range(40):
        (ra, xa), (rb, xb) = a[i], b[i]
        assert ra == rb and (xa is None) == (xb is None)
        if xa is not None:
            assert all(np.array_equal(xa[k], xb[k]) for k in xa)
    far = TrackPairDataset(tracks, length=60, seed=7, negative_probability=0.0, max_gap=5, long_gap_probability=1.0)
    gaps = [abs(r["search"][1]-r["template"][1]) for r, _ in (far[i] for i in range(60)) if "search" in r]
    assert gaps and max(gaps) > 5
    with pytest.raises(ValueError):
        TrackPairDataset(tracks, length=5, seed=0, long_gap_probability=2.0)
