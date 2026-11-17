"""CPU contracts for GT-free inputs, adapter semantics, and original crop extraction."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
read_sequence = runpy.run_path(str(ROOT / 'scripts/run_sequence.py'))['read_sequence']


def test_importer_export_is_accepted_but_no_ground_truth_allowed(tmp_path):
    importer = runpy.run_path(str(ROOT / 'scripts/import_assets.py'))['main']
    images = tmp_path / 'DTB70' / 'Toy' / 'img'
    images.mkdir(parents=True)
    (images / '00001.jpg').write_bytes(b'fixture-for-existence-not-decoding')
    (images.parent / 'groundtruth_rect.txt').write_text('1,1,10,20\n')
    manifest = tmp_path / 'dataset.json'
    exports = tmp_path / 'inference'
    assert importer(['dataset', '--root', str(images.parents[1]), '--dataset', 'DTB70', '--allow-partial',
                     '--output', str(manifest), '--coordinate-origin', '1', '--inference-dir', str(exports)]) == 0
    sequence, frames = read_sequence(exports / 'Toy.json')
    assert sequence['initial_bbox'] == [.5, .5, 10, 20]
    assert frames == [images / '00001.jpg']
    sequence['ground_truth'] = [[1, 1, 10, 20]]
    path = tmp_path / 'forbidden.json'
    path.write_text(json.dumps(sequence))
    with pytest.raises(ValueError, match='forbidden'):
        read_sequence(path)
    sequence.pop('ground_truth')
    sequence['evaluation_ground_truth_included'] = True
    path.write_text(json.dumps(sequence))
    with pytest.raises(ValueError, match='cannot include'):
        read_sequence(path)


@pytest.mark.parametrize('origin', [None, 1])
def test_runner_never_guesses_coordinate_origin(tmp_path, origin):
    frame = tmp_path / 'one.jpg'
    frame.write_bytes(b'fixture')
    sequence = {'name': 'Toy', 'dataset': 'DTB70', 'frames': ['one.jpg'], 'initial_bbox': [0, 0, 1, 1], 'coordinate_origin': origin}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(sequence))
    with pytest.raises(ValueError, match='zero-based'):
        read_sequence(path)


def test_native_passthrough_metadata_cannot_smuggle_evaluation_gt(tmp_path):
    frame = tmp_path / 'one.jpg'
    frame.write_bytes(b'fixture')
    sequence = {'name': 'Native', 'dataset': 'UAV123@10fps', 'frames': ['one.jpg'],
                'initial_bbox': [1.5, 1.5, 9, 8], 'coordinate_origin': 0,
                'annotation_coordinate_origin': None,
                'annotation_to_tracker_transform': 'upstream_numeric_passthrough_plus_half_pixel_no_origin_shift'}
    path = tmp_path / 'native.json'
    path.write_text(json.dumps(sequence))
    assert read_sequence(path)[0]['initial_bbox'] == [1.5, 1.5, 9, 8]
    sequence['annotation_to_tracker_transform'] = {'GT': [[1, 1, 9, 8]]}
    path.write_text(json.dumps(sequence))
    with pytest.raises(ValueError, match='declared literal'):
        read_sequence(path)


def test_score_adapter_changes_only_bce_head():
    torch = pytest.importorskip('torch')
    from hift_research.inference import ScoreProbabilityAdapter
    outputs = {'loc': torch.ones(1, 4, 11, 11), 'cls1': torch.ones(1, 2, 11, 11),
               'cls2': torch.tensor([-1000., 0., 1000.])}
    class Fake:
        def eval(self):
            return self
        def template(self, image):
            self.input = image
        def track(self, image):
            return outputs
    model = Fake()
    adapter = ScoreProbabilityAdapter(model).eval()
    adapter.template('sentinel')
    assert model.input == 'sentinel'
    actual = adapter.track(None)
    assert actual['loc'] is outputs['loc'] and actual['cls1'] is outputs['cls1']
    torch.testing.assert_close(actual['cls2'], torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(outputs['cls2'], torch.tensor([-1000., 0., 1000.]))


def test_restricted_checkpoint_numpy_metadata_and_prefix_safety(tmp_path):
    torch = pytest.importorskip('torch')
    from hift_research.inference import read_checkpoint_state
    path = tmp_path / 'safe.pth'
    torch.save({'state_dict': {'module.weight': torch.ones(2)}, 'optimizer': {'lr': np.float64(.01)}}, path)
    before = set(torch.serialization.get_safe_globals())
    state = read_checkpoint_state(path)
    assert list(state) == ['weight']
    torch.testing.assert_close(state['weight'], torch.ones(2))
    assert set(torch.serialization.get_safe_globals()) == before  # API returns an unordered set as a list
    torch.save({'weight': torch.ones(2), 'module.weight': torch.zeros(2)}, path)
    with pytest.raises(ValueError, match='Duplicate'):
        read_checkpoint_state(path)
    torch.save({'state_dict': {'weight': 'not a tensor'}}, path)
    with pytest.raises(ValueError, match='tensor weights'):
        read_checkpoint_state(path)


@pytest.mark.parametrize('center,source_size', [((0., 0.), 7), ((15.5, 10.25), 8), ((10., 10.), 20)])
def test_crop_geometry_matches_actual_upstream_pixels(center, source_size):
    torch = pytest.importorskip('torch')
    cv2 = pytest.importorskip('cv2')
    pytest.importorskip('yacs')
    sys.path.insert(0, str(ROOT / 'third_party/HiFT'))
    from pysot.core.config import cfg
    from pysot.tracker.base_tracker import SiameseTracker
    from hift_research.geometry import CropGeometry
    image = np.arange(40 * 50 * 3, dtype=np.uint8).reshape(40, 50, 3)
    average = image.mean(axis=(0, 1))
    geometry = CropGeometry.from_tracker(center, source_size, 13)
    expected = np.empty((source_size, source_size, 3), dtype=np.uint8)
    expected[:] = average
    for y in range(source_size):
        for x in range(source_size):
            sy, sx = geometry.origin_y + y, geometry.origin_x + x
            if 0 <= sy < image.shape[0] and 0 <= sx < image.shape[1]:
                expected[y, x] = image[sy, sx]
    expected = cv2.resize(expected, (13, 13)).transpose(2, 0, 1)[None].astype(np.float32)
    was_cuda = cfg.CUDA
    cfg.CUDA = False
    try:
        actual = SiameseTracker().get_subwindow(image, np.array(center), 13, source_size, average)
    finally:
        cfg.CUDA = was_cuda
    np.testing.assert_array_equal(actual.numpy(), expected)
