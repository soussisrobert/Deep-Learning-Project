"""Geometric invariants behind the half-pixel centre-offset question (reports/B1_CENTER_OFFSET.md).

CPU only, deterministic, no network weights. Coordinates: pixel index i has its centre at i and covers the
continuous interval [i, i+1); a GT box (x, y, w, h) has index centre x+(w-1)/2 and continuous centre x+w/2.
Characterisation tests pin the current conventions; the strict-xfail tests state the ideal convention and fail
because of a convention mismatch kept by the protected upstream files (the report explains why no B1 correction is
proposed).
"""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _probe():
    if "b1_center_offset_probe" not in sys.modules:
        spec = importlib.util.spec_from_file_location("b1_center_offset_probe", ROOT/"scripts/b1_center_offset_probe.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
        spec.loader.exec_module(module)
    return sys.modules["b1_center_offset_probe"]


def _openblas_threads(count):
    """Best effort (speed only): set numpy's bundled OpenBLAS thread count, return the previous count or None.
    Multi-threaded BLAS spin-waits badly on an oversubscribed machine; these tests only do small products."""
    import ctypes
    libs = sorted((Path(np.__file__).resolve().parent.parent/"numpy.libs").glob("libscipy_openblas*"))
    try:
        lib = ctypes.CDLL(str(libs[0]))
        previous = lib.scipy_openblas_get_num_threads64_()
        lib.scipy_openblas_set_num_threads64_(count)
        return previous
    except (IndexError, OSError, AttributeError):
        return None


@pytest.fixture(scope="module")
def upstream():
    import torch
    from hift_research.inference import configure_upstream
    cfg = configure_upstream(ROOT)
    previous, cfg.CUDA = cfg.CUDA, False  # CPU tensors from get_subwindow; no model is built here
    threads, blas = torch.get_num_threads(), _openblas_threads(1)
    torch.set_num_threads(1)  # tiny tensors only; avoids OpenMP barrier stalls on a loaded machine
    yield cfg
    cfg.CUDA = previous
    torch.set_num_threads(threads)
    if blas:
        _openblas_threads(blas)


@pytest.fixture(scope="module")
def oracle_tracker(upstream):
    from hift_research.inference import ScoreProbabilityAdapter
    from hift_research.tracking_nan_safe import make_nan_safe_tracker
    probe = _probe()
    tracker = make_nan_safe_tracker(ScoreProbabilityAdapter(probe.OracleLocator()))
    return tracker, probe.exact_variant(tracker)


def _index_image(size=600):
    """uint8 image whose pixel stores its own (x, y) index: ch0 = x % 256, ch1 = y % 256, ch2 = 16*(x//256)+y//256."""
    y, x = np.mgrid[:size, :size]
    return np.stack([x % 256, y % 256, 16*(x//256)+y//256], -1).astype(np.uint8)


def _crop_origin(tracker, image, pos, source_size):
    patch = tracker.get_subwindow(image, np.asarray(pos, float), source_size, source_size, np.zeros(3)).numpy()[0]
    hi = int(patch[2, 0, 0])
    return int(patch[0, 0, 0])+256*(hi//16), int(patch[1, 0, 0])+256*(hi % 16)


PHASES = (np.arange(16)+0.5)/16  # midpoints: their mean fractional part is exactly 1/2


def _crop_centre_errors(tracker, source_size):
    image = _index_image()
    errors = []
    for phase in PHASES:
        pos = (300+phase, 280.25+phase)
        ox, oy = _crop_origin(tracker, image, pos, source_size)
        errors.append((ox+(source_size-1)/2-pos[0], oy+(source_size-1)/2-pos[1]))
    return np.asarray(errors)


# ------------------------------------------------------------------- G2: crop origin (base_tracker.py:51-56)
@pytest.mark.parametrize("source_size", [7, 8, 54, 55, 287, 438])
def test_crop_origin_is_floor_of_pos_minus_half_size(upstream, oracle_tracker, source_size):
    from hift_research.geometry import CropGeometry
    tracker, _ = oracle_tracker
    image = _index_image()
    for phase in PHASES:
        pos = (300+phase, 280.25+phase)
        origin = _crop_origin(tracker, image, pos, source_size)
        assert origin == (math.floor(pos[0]-source_size/2), math.floor(pos[1]-source_size/2))
        geometry = CropGeometry.from_tracker(pos, source_size, 287)
        assert origin == (geometry.origin_x, geometry.origin_y)


@pytest.mark.parametrize("source_size", [7, 8, 54, 55, 287, 438])
def test_crop_centre_sits_one_pixel_before_the_state_on_average(upstream, oracle_tracker, source_size):
    errors = _crop_centre_errors(oracle_tracker[0], source_size)
    assert np.all(errors > -1.5) and np.all(errors <= -0.5)  # per phase: (-1.5, -0.5]
    np.testing.assert_allclose(errors.mean(0), [-1.0, -1.0], atol=1e-12)


@pytest.mark.xfail(strict=True, reason="convention mismatch G2: get_subwindow origin floor(pos-(sz+1)/2+.5) "
                   "centres the crop at pos-1 on average while generate_anchor/track decode relative to pos")
@pytest.mark.parametrize("source_size", [8, 55])
def test_ideal_crop_is_centred_on_the_state(upstream, oracle_tracker, source_size):
    errors = _crop_centre_errors(oracle_tracker[0], source_size)
    np.testing.assert_allclose(errors.mean(0), [0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------- resize keeps the crop centre (base_tracker.py:88)
@pytest.mark.parametrize("source_size", [50, 51, 300, 301])
@pytest.mark.parametrize("model_size", [127, 287])
def test_resize_maps_crop_centre_to_model_centre(upstream, oracle_tracker, source_size, model_size):
    tracker, _ = oracle_tracker
    c = (source_size-1)/2
    y, x = np.mgrid[:source_size, :source_size]
    blob = 255*np.exp(-((x-c)**2+(y-c)**2)/(2*(source_size/6)**2))
    image = np.repeat(np.rint(blob)[..., None], 3, -1).astype(np.uint8)
    pos = (source_size/2+0.25, source_size/2+0.25)  # origin floor(pos - sz/2) = 0: the crop is the whole image
    patch = tracker.get_subwindow(image, pos, model_size, source_size, np.zeros(3)).numpy()[0, 0]
    rows, cols = np.mgrid[:model_size, :model_size]
    centroid = ((patch*cols).sum()/patch.sum(), (patch*rows).sum()/patch.sum())
    np.testing.assert_allclose(centroid, [(model_size-1)/2]*2, atol=1e-3)  # OpenCV fixed-point weights


# ------------------------------------------- decoder inverts the training label encoding (grid 16*i+63)
def test_decode_inverts_training_encoding(upstream, oracle_tracker):
    import torch
    from hift_research.corrections.positive_targets import CorrectedAnchorTarget
    tracker, _ = oracle_tracker
    target = CorrectedAnchorTarget(SimpleNamespace(TRAIN=SimpleNamespace(SEARCH_SIZE=287, OUTPUTFEATURE_SIZE=11,
                                                                         LABEL_RANGE=4)))
    rng = np.random.default_rng(7)
    np.random.seed(7)
    for _ in range(25):
        cx, cy = rng.uniform(60, 227, 2)
        w, h = rng.uniform(6, 120, 2)
        box = SimpleNamespace(x1=cx-w/2, y1=cy-h/2, x2=cx+w/2, y2=cy+h/2)
        _, encoded, _, _ = target.get(box, 11)
        decoded = tracker.generate_anchor(torch.from_numpy(np.asarray(encoded, np.float64)[None]))
        expected = np.tile([cx-143, cy-143, w, h], (121, 1))  # crop index 143 = (287-1)/2 is the decode origin
        np.testing.assert_allclose(decoded, expected, atol=1e-3)  # float32 label storage


# ---------------------------- G3: training label point = continuous centre x+w/2 used as an index coordinate
def test_training_crop_places_continuous_centre_at_crop_centre():
    from hift_research.data_contract.source_ops import source_operations
    operations = source_operations()
    x0, y0, w, h = 200.3, 180.6, 40.0, 40.0
    edges = np.arange(400)
    ax = np.clip(np.minimum(edges+1, x0+w)-np.maximum(edges, x0), 0, None)
    ay = np.clip(np.minimum(edges+1, y0+h)-np.maximum(edges, y0), 0, None)
    image = np.repeat((255*np.outer(ay, ax))[..., None], 3, -1).astype(np.float32)
    _, crop = operations.crop(image, [x0, y0, x0+w, y0+h], instanc_size=511, padding=(0, 0, 0))
    side = math.sqrt((w+(w+h)/2)*(h+(w+h)/2))*511/127  # crop_like_SiamFC s_x for context 0.5
    scale = 510/side
    mass = crop[..., 0]
    rows, cols = np.mgrid[:511, :511]
    centroid = np.array([(mass*cols).sum(), (mass*rows).sum()])/mass.sum()
    # object index centre x0+(w-1)/2 lands half a source pixel (scaled) before crop index 255
    np.testing.assert_allclose(centroid, [255-0.5*scale]*2, atol=0.02)
    label = operations.get_bbox(None, crop, [x0, y0, x0+w, y0+h])
    assert ((label.x1+label.x2)/2, (label.y1+label.y2)/2) == (255, 255)


# -------------------------------------------- G1: init (w-1)/2 vs output cx-w/2 (hift_tracker.py:81 vs 182)
def _oracle_errors(tracker, exact, case="static", sizes=(13, 24), seeds=3, frames=4, shift=0.0):
    probe = _probe()
    out = []
    for size in sizes:
        for seed in range(seeds):
            scenes, _ = probe.sequence(case, size, seed, frames, marker=True, shift=shift)
            predictions, padded = probe.track(tracker, scenes, exact)
            assert padded == 0
            e = probe.errors(predictions, scenes)
            out.append(np.stack([e["dx"], e["dy"]], 1))
    return np.concatenate(out)


def test_exact_locator_on_exact_crops_is_offset_by_minus_half_pixel(upstream, oracle_tracker):
    errors = _oracle_errors(oracle_tracker[1], exact=True)
    np.testing.assert_allclose(errors, -0.5, atol=0.01)


@pytest.mark.xfail(strict=True, reason="convention mismatch G1: init stores the index centre x+(w-1)/2, track() "
                   "emits x = cx - w/2, so an exact state is reported 0.5 px left/up")
def test_ideal_exact_locator_reproduces_the_gt_box(upstream, oracle_tracker):
    errors = _oracle_errors(oracle_tracker[1], exact=True, sizes=(24,), seeds=1, frames=2)
    np.testing.assert_allclose(errors, 0.0, atol=0.05)


# ------------------------------------------------- G1 + G2 end to end with the real crop and an exact locator
def test_exact_locator_through_the_real_pipeline_follows_the_floor_formula(upstream, oracle_tracker):
    """Static target: state_t = c - d(state_{t-1}), d = floor(p - S/2) + (S-1)/2 - p (S = round(s_x)), so the
    reported continuous centre is off by -d - 1/2 in [0, 1) per frame; mean +0.5 over phases."""
    from pysot.core.config import cfg
    probe = _probe()
    tracker = oracle_tracker[0]
    residuals, offsets = [], []
    for size in (13, 24, 48):
        for seed in range(4):
            scenes, _ = probe.sequence("static", size, seed, 4, marker=True)
            x0, y0, w, h = scenes[0].box
            centre = np.array([x0+(w-1)/2, y0+(h-1)/2])
            base = probe.render_background(scenes[0], 0.0, 0.0, 1.0, probe.FRAME, probe.FRAME)
            image = probe.to_uint8(probe.render(scenes[0], 0.0, 0.0, 1.0, probe.FRAME, probe.FRAME, base=base))
            tracker.init(image, list(scenes[0].box))
            for _ in range(4):
                p, size_now = tracker.center_pos.copy(), tracker.size.copy()
                pad = cfg.TRACK.CONTEXT_AMOUNT*size_now.sum()
                source = round(math.sqrt((size_now[0]+pad)*(size_now[1]+pad))*287/127)
                d = np.floor(p-source/2)+(source-1)/2-p
                tracker.track(image)
                residuals.append(tracker.center_pos-(centre-d))
                offsets.append(-d-0.5)
    assert np.abs(residuals).max() < 0.06  # crop resampling of the coverage marker
    offsets = np.asarray(offsets)
    assert np.all(offsets >= 0) and np.all(offsets < 1)
    errors = _oracle_errors(tracker, exact=False, sizes=(13, 24, 48, 97), seeds=6, frames=4)
    assert abs(errors.mean()-0.5) < 0.1


def test_whole_scene_shift_equivariance(upstream, oracle_tracker):
    """Integer translations are exactly equivariant; a half-pixel translation moves the pipeline's first
    prediction by 0 or 1 px (crop-origin quantisation) but the exact-crop arm by exactly 0.5 px."""
    pipeline, exact = oracle_tracker
    kwargs = dict(case="shift", sizes=(13, 24, 49), seeds=2, frames=1)
    for tracker, is_exact in ((pipeline, False), (exact, True)):
        base = _oracle_errors(tracker, is_exact, shift=0.0, **kwargs)
        integer = _oracle_errors(tracker, is_exact, shift=1.0, **kwargs)
        half = _oracle_errors(tracker, is_exact, shift=0.5, **kwargs)
        np.testing.assert_allclose(integer, base, atol=1e-6)  # errors are GT-relative: shift already removed
        if is_exact:
            np.testing.assert_allclose(half, base, atol=0.01)
        else:
            np.testing.assert_allclose(np.abs(half-base), 0.5, atol=0.06)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_crop_geometry_rejects_non_finite_state(bad):
    from hift_research.geometry import CropGeometry
    with pytest.raises(ValueError):
        CropGeometry.from_tracker((bad, 3.0), 55, 287)
