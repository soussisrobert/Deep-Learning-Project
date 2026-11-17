import json
import numpy as np
import pytest
from hift_research.diagnostics.policy import RawOutputRecorder, decision_maps
from hift_research.diagnostics.source_support import retrospective_rows, summarize_rows


def test_recorder_is_transparent_and_preserves_original_outputs():
    outputs = {'loc': object(), 'cls1': object(), 'cls2': object()}
    class Fake:
        def eval(self):
            return self
        def template(self, image):
            self.input = image
        def track(self, image):
            return outputs
    original = Fake()
    recorder = RawOutputRecorder(original).eval()
    recorder.template('initial')
    assert original.input == 'initial'
    assert recorder.track(None) is outputs
    assert recorder.latest is outputs


def test_raw_logit_vs_probability_can_reverse_ranking_on_same_state():
    boxes = np.array([[0., 0.], [0., 0.], [10., 10.], [10., 10.]])
    p, logits = np.array([1., .2], dtype=np.float32), np.array([-1., 0.], dtype=np.float32)
    kwargs = dict(penalty_k=0., window_influence=0.)
    raw = decision_maps(boxes, p, logits, [10, 10], 1., np.zeros(2), **kwargs)
    probability = decision_maps(boxes, p, 1 / (1 + np.exp(-logits)), [10, 10], 1., np.zeros(2), **kwargs)
    assert raw['selected_index'] == 1 and probability['selected_index'] == 0
    assert np.allclose(raw['score'], (p + logits) / 2)


def test_same_selected_cell_different_interpolation_not_ranking():
    boxes = np.array([[0.], [0.], [10.], [10.]])
    raw = decision_maps(boxes, np.array([.8]), np.array([-2.]), [10, 10], 1., np.zeros(1))
    bounded = decision_maps(boxes, np.array([.8]), np.array([.12]), [10, 10], 1., np.zeros(1))
    assert raw['selected_index'] == bounded['selected_index'] == 0
    assert raw['interpolation_lr'] < 0 < bounded['interpolation_lr'] < 1


def test_retrospective_gt_join_is_explicit_and_keeps_support_distinct():
    gt = [[49, 49, 4, 4], [49, 49, 4, 4]]  # annotation1-origin -> [48,48,4,4]
    pred = [[48, 48, 4, 4], [48, 48, 4, 4]]
    logs = [{'frame': i, 'bbox': p, 'image_shape': [100, 100, 3],
             'crop_geometry': None if i == 0 else {'origin_x': 40, 'origin_y': 40, 'source_size': 20, 'model_size': 287}}
            for i, p in enumerate(pred)]
    rows = retrospective_rows(gt, pred, logs, annotation_origin=1)
    assert gt[0][0] == 49  # input annotations not modified
    assert rows[1]['source_area_px2'] == 16
    assert rows[1]['resized_width_px'] == pytest.approx(57.4)
    assert rows[1]['crop_containment_fraction'] == 1
    assert rows[1]['iou'] == 1
    assert rows[1]['support_source'] == 'retrospective_GT_not_used_in_inference'
    assert rows[1]['occlusion'] is None and rows[1]['optical_detail_measured'] is False
    summary = summarize_rows(rows)
    assert summary['tracked_frames_excluding_init'] == 1
    assert summary['source_area_bins'][0]['frames'] == 1
    with pytest.raises(ValueError):
        retrospective_rows(gt, pred, logs, annotation_origin=None)
    with pytest.raises(ValueError):
        retrospective_rows(gt, pred[:1], logs, annotation_origin=1)


def test_absent_gt_is_not_dropped_or_imputed():
    gt = [[1, 1, 4, 4], [None, None, None, None]]
    pred = [[1, 1, 4, 4]] * 2
    logs = [{'frame': i, 'bbox': p, 'image_shape': [100, 100, 3], 'crop_geometry': None} for i,p in enumerate(pred)]
    rows = retrospective_rows(gt, pred, logs, annotation_origin=1, invalid_policy='count_as_failure')
    assert len(rows) == 2 and rows[1]['source_area_px2'] is None
    assert rows[1]['ground_truth_unavailable'] and rows[1]['iou'] == -1
    summary = summarize_rows(rows)
    assert summary['GT_unavailable_frames_counted_as_failure'] == 1
    assert summary['measured_zero_overlap_frames_valid_GT_only'] == 0
    assert summary['source_area_bins'] == []
    json.dumps(rows, allow_nan=False)


def test_crop_miss_and_image_boundary_are_not_conflated():
    pred, gt = [[0, 0, 4, 4]] * 2, [[-2, -2, 4, 4]] * 2
    logs = [{'frame': i, 'bbox': p, 'image_shape': [100, 100, 3],
             'crop_geometry': None if i == 0 else {'origin_x': -10, 'origin_y': -10, 'source_size': 20, 'model_size': 287}}
            for i, p in enumerate(pred)]
    row = retrospective_rows(gt, pred, logs, annotation_origin=0)[1]
    assert row['crop_containment_fraction'] == 1
    assert row['source_image_containment_fraction'] == .25
    assert row['joint_crop_image_containment_fraction'] == .25
