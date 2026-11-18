"""Retrospective annotation join; GT is never used by the inference process."""
import numpy as np
from hift_research.geometry import CropGeometry


def _intersection_fraction(box, bounds):
    x, y, w, h = box
    x0, y0, x1, y1 = bounds
    return float(max(0., min(x + w, x1) - max(x, x0)) *
                 max(0., min(y + h, y1) - max(y, y0)) / (w * h))


def frame_errors(gt, pred, *, invalid_policy='error'):
    gt, pred = np.asarray(gt, dtype=float), np.asarray(pred, dtype=float)
    if invalid_policy not in ('error', 'count_as_failure'):
        raise ValueError('invalid GT policy')
    if gt.shape != (4,) or pred.shape != (4,) or not np.isfinite(pred).all() or min(pred[2:]) <= 0:
        raise ValueError('finite positive prediction xywh and four GT coordinates required')
    if not np.isfinite(gt).all() or min(gt[2:]) <= 0:
        if invalid_policy == 'error':
            raise ValueError('finite positive GT required')
        return {'iou': -1., 'center_error_px': None, 'normalized_center_error': None,
                'zero_overlap_failure': True, 'iou_below_0_1': True, 'center_error_over_20px': True,
                'ground_truth_unavailable': True, 'failure_interpretation': 'protocol_count_as_failure_not_observed_error'}
    extent = np.maximum(0., np.minimum(gt[:2] + gt[2:], pred[:2] + pred[2:]) - np.maximum(gt[:2], pred[:2]))
    intersection = np.prod(extent)
    iou = intersection / (np.prod(gt[2:]) + np.prod(pred[2:]) - intersection)
    center_delta = pred[:2] + (pred[2:] - 1) / 2 - gt[:2] - (gt[2:] - 1) / 2
    return {'iou': float(iou), 'center_error_px': float(np.linalg.norm(center_delta)),
            'normalized_center_error': float(np.linalg.norm(center_delta / gt[2:])),
            'zero_overlap_failure': bool(iou <= 0), 'iou_below_0_1': bool(iou < .1),
            'center_error_over_20px': bool(np.linalg.norm(center_delta) > 20),
            'ground_truth_unavailable': False, 'failure_interpretation': 'measured_valid_GT'}


def retrospective_rows(ground_truth, predictions, logs, *, annotation_origin, invalid_policy='error'):
    """Boxes are annotation-native GT and zero-origin predictions; require explicit origin."""
    if annotation_origin not in (0, 1):
        raise ValueError('explicit annotation origin 0 or 1 required')
    gt, pred = np.array(ground_truth, dtype=float), np.asarray(predictions, dtype=float)
    if gt.ndim != 2 or gt.shape[1] != 4 or gt.shape != pred.shape or len(gt) != len(logs):
        raise ValueError('frame/GT/prediction count mismatch')
    gt[:, :2] -= annotation_origin
    result = []
    previous_gt_center, previous_gt_area = None, None
    for index, (box, prediction, log) in enumerate(zip(gt, pred, logs)):
        if log['frame'] != index:
            raise ValueError('log frames must be complete, ordered and zero-based')
        if not np.allclose(log['bbox'], prediction, rtol=1e-8, atol=1e-7):
            raise ValueError('prediction/log mismatch')
        errors = frame_errors(box, prediction, invalid_policy=invalid_policy)
        if errors['ground_truth_unavailable']:
            result.append({'frame': index, 'initialization': index == 0, **errors,
                           'source_width_px': None, 'source_height_px': None, 'source_area_px2': None,
                           'aspect_ratio': None, 'motion_px': None, 'absolute_log_area_change': None,
                           'support_source': 'unavailable_GT_no_support_imputed', 'occlusion': None,
                           'motion_blur': None, 'distractor_confusion': None})
            previous_gt_center, previous_gt_area = None, None
            continue
        width, height = box[2:]
        center = box[:2] + (box[2:] - 1) / 2
        row = {'frame': index, 'initialization': index == 0, 'source_width_px': float(width),
               'source_height_px': float(height), 'source_area_px2': float(width * height),
               'aspect_ratio': float(width / height),
               'motion_px': float(np.linalg.norm(center - previous_gt_center)) if previous_gt_center is not None else None,
               'absolute_log_area_change': float(abs(np.log(width * height / previous_gt_area))) if previous_gt_area is not None else None,
               'occlusion': None, 'motion_blur': None, 'distractor_confusion': None,
               'attributes_status': 'not_inferred_from_boxes_or_failures', **errors}
        previous_gt_center, previous_gt_area = center, width * height
        geometry = log.get('crop_geometry')
        if geometry is not None:
            crop = CropGeometry(**{key: geometry[key] for key in ('origin_x', 'origin_y', 'source_size', 'model_size')})
            support = crop.support(box, log['image_shape'])
            ih, iw = log['image_shape'][:2]
            row.update({key: support[key] for key in ('resized_width_px', 'resized_height_px', 'crop_source_size_px',
                                                    'resize_scale', 'source_pixels_per_model_pixel')})
            row.update({'crop_containment_fraction': _intersection_fraction(box, (crop.origin_x, crop.origin_y,
                              crop.origin_x + crop.source_size, crop.origin_y + crop.source_size)),
                        'source_image_containment_fraction': _intersection_fraction(box, (0, 0, iw, ih)),
                        'joint_crop_image_containment_fraction': support['containment_fraction'],
                        'geometric_target_in_crop_area_px2': support['genuine_visible_support_px2'],
                        'resized_contained_target_area_px2': support['genuine_visible_support_px2'] * crop.scale**2,
                        'support_source': 'retrospective_GT_not_used_in_inference',
                        'optical_detail_measured': False})
        result.append(row)
    return result


def _episode_count(flags):
    return sum(value and (index == 0 or not flags[index - 1]) for index, value in enumerate(flags))


def summarize_rows(rows):
    """Prespecified descriptive bins; no frame-iid significance or causal claim."""
    tracked = [row for row in rows if not row['initialization']]
    available = [row for row in tracked if not row['ground_truth_unavailable']]
    bins = []
    for name, low, high in [('area_lt_16_squared', 0., 256.), ('area_16_squared_to_32_squared', 256., 1024.),
                            ('area_ge_32_squared', 1024., float('inf'))]:
        subset = [row for row in available if low <= row['source_area_px2'] < high]
        if subset:
            bins.append({'bin': name, 'frames': len(subset), 'mean_iou': float(np.mean([r['iou'] for r in subset])),
                         'zero_overlap_fraction': float(np.mean([r['zero_overlap_failure'] for r in subset])),
                         'mean_resized_min_side_px': float(np.mean([min(r['resized_width_px'], r['resized_height_px']) for r in subset]))})
    flags = [row['zero_overlap_failure'] for row in tracked]
    return {'tracked_frames_excluding_init': len(tracked),
            'GT_unavailable_frames_counted_as_failure': len(tracked) - len(available),
            'measured_zero_overlap_frames_valid_GT_only': sum(row['zero_overlap_failure'] for row in available),
            'zero_overlap_frames': sum(flags),
            'zero_overlap_episodes': _episode_count(flags),
            'iou_below_0_1_frames': sum(row['iou_below_0_1'] for row in tracked),
            'center_error_over_20px_frames': sum(row['center_error_over_20px'] for row in tracked),
            'crop_not_fully_contained_frames': sum(row['crop_containment_fraction'] < 1 - 1e-9 for row in available),
            'source_area_bins': bins,
            'interpretation': 'descriptive_within_selected_sequences_only; time/sequence/containment confounding remains'}
