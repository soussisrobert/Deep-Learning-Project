from pathlib import Path
import numpy as np
import pytest
from hift_research.geometry import CropGeometry, blend_scores, stable_sigmoid


@pytest.mark.parametrize('source_size', [1, 2, 7, 64, 287, 1024])
@pytest.mark.parametrize('center', [(0., 0.), (15.5, 9.25), (-2., 105.)])
def test_half_pixel_geometry_round_trip(source_size, center):
    geom = CropGeometry.from_tracker(center, source_size, 287)
    points = np.array([[0., 0.], center, [100., 200.]])
    np.testing.assert_allclose(geom.model_to_source(geom.source_to_model(points)), points, atol=1e-12)
    assert geom.origin_x == np.floor(center[0] - (source_size + 1) / 2 + .5)


def test_resizing_does_not_create_source_support():
    small = CropGeometry.from_tracker((50, 50), 20, 287).support((48, 48, 4, 4), (200, 200))
    large = CropGeometry.from_tracker((50, 50), 100, 287).support((40, 40, 20, 20), (200, 200))
    assert small['resized_width_px'] == pytest.approx(large['resized_width_px'])
    assert small['source_area_px2'] == 16
    assert large['source_area_px2'] == 400
    assert small['containment_fraction'] == large['containment_fraction'] == 1


def test_padding_is_not_genuine_support():
    geom = CropGeometry.from_tracker((0., 0.), 20, 287)
    support = geom.support((-2, -2, 4, 4), (100, 100))
    assert support['genuine_visible_support_px2'] == 4
    assert support['containment_fraction'] == .25


def test_sigmoid_and_blending_extreme_logits():
    z = np.array([-1000., -10., 0., 10., 1000.])
    p = np.full_like(z, .5)
    with np.errstate(over='raise'):
        expected = stable_sigmoid(z)
    assert np.all((expected >= 0) & (expected <= 1))
    assert expected[2] == .5
    original = blend_scores(p, z)
    repaired = blend_scores(p, z, corrected=True)
    assert original.min() < 0 and original.max() > 1
    assert repaired.min() >= 0 and repaired.max() <= 1
    assert original[1] < 0  # negative interpolation LR is possible in original tracker


def test_blend_correction_not_calibration():
    np.testing.assert_allclose(blend_scores([.2, .8], [0, 0], corrected=True), [.35, .65])
    with pytest.raises(ValueError):
        blend_scores([1.1], [0])
    with pytest.raises(ValueError):
        stable_sigmoid([np.nan])


def test_original_minimum_size_and_corner_overflow_are_heuristics():
    # Execute original method with minimal imports when optional dependencies exist.
    pytest.importorskip('torch')
    pytest.importorskip('cv2')
    pytest.importorskip('yacs')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/HiFT'))
    from pysot.tracker.hift_tracker import HiFTTracker
    clipped = HiFTTracker._bbox_clip(None, 0, 0, 2, 20, (100, 100))
    assert clipped == (0, 0, 10, 20)
    assert clipped[0] - clipped[2] / 2 == -5
    # On images smaller than 10 the chosen minimum exceeds image size; record, don't silently fix.
    assert HiFTTracker._bbox_clip(None, 0, 0, 2, 2, (5, 5))[2:] == (10, 10)


@pytest.mark.parametrize('source_size', [0, -1, 1.2])
def test_invalid_crop_size_rejected(source_size):
    with pytest.raises(ValueError):
        CropGeometry.from_tracker((0, 0), source_size, 287)
