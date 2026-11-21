"""Actual original tracker decisions vs passive observer, CPU and deterministic outputs."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('baseline', ['B0', 'B1-score'])
def test_observer_reconstructs_actual_original_tracker_without_mutating_decisions(baseline):
    torch = pytest.importorskip('torch')
    pytest.importorskip('cv2')
    pytest.importorskip('yacs')
    sys.path.insert(0, str(ROOT / 'third_party/HiFT'))
    from pysot.core.config import cfg
    from hift_research.inference import make_tracker
    from hift_research.diagnostics.policy import RawOutputRecorder, observe_decision
    previous = cfg.clone()
    try:
        cfg.defrost()
        cfg.merge_from_file(str(ROOT / 'third_party/HiFT/experiments/config.yaml'))
        cfg.CUDA = False
        grid = torch.linspace(-4., 4., 121).reshape(1, 1, 11, 11)
        outputs = {'loc': torch.full((1, 4, 11, 11), .15), 'cls1': torch.cat([-grid, grid], dim=1), 'cls2': grid}
        class FakeModel:
            def eval(self):
                return self
            def template(self, image):
                pass
            def track(self, image):
                return outputs
        recorder = RawOutputRecorder(FakeModel())
        tracker = make_tracker(recorder, baseline)
        image = np.zeros((200, 240, 3), dtype=np.uint8)
        tracker.init(image, [60, 60, 20, 20])
        center, size = tracker.center_pos.copy(), tracker.size.copy()
        wz = size[0] + cfg.TRACK.CONTEXT_AMOUNT * size.sum()
        hz = size[1] + cfg.TRACK.CONTEXT_AMOUNT * size.sum()
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / np.sqrt(wz * hz)
        result = tracker.track(image)
        summary, grids = observe_decision(tracker, recorder, size, scale_z, cfg)
        selected = summary[baseline]
        assert result['best_score'] == pytest.approx(selected['selected_score'], abs=1e-7)
        box = grids['predicted_crop_boxes'][:, selected['selected_index']] / scale_z
        lr = selected['interpolation_lr']
        cx, cy, width, height = tracker._bbox_clip(box[0] + center[0], box[1] + center[1],
            size[0] * (1 - lr) + box[2] * lr, size[1] * (1 - lr) + box[3] * lr, image.shape[:2])
        np.testing.assert_allclose(result['bbox'], [cx - width/2, cy - height/2, width, height], rtol=0, atol=1e-10)
        torch.testing.assert_close(outputs['cls2'], grid)
    finally:
        cfg.clear()
        cfg.update(previous)
