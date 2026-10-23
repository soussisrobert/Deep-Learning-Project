"""Compare immutable matched runs; native-numeric metrics vs physical diagnostics explicit."""
import json
from pathlib import Path
import numpy as np
from hift_research.evaluation.assets import read_json, sha256_file
from hift_research.evaluation.metrics import aggregate_sequences, evaluate_sequence
from hift_research import OFFICIAL_CHECKPOINT_SHA256, UPSTREAM_COMMIT
from .source_support import frame_errors, retrospective_rows, summarize_rows


def _logs(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def compare_sequence(dataset_row, b0_directory, b1_directory, *, annotation_origin, invalid_policy='error'):
    directories = [Path(b0_directory), Path(b1_directory)]
    manifests = [read_json(d / 'manifest.json') for d in directories]
    a, b = manifests
    if [a['baseline'], b['baseline']] != ['B0', 'B1-score'] or any(m['status'] != 'completed' for m in manifests):
        raise ValueError('Completed B0 and B1-score runs required, in that order')
    if any(m['checkpoint']['sha256'] != OFFICIAL_CHECKPOINT_SHA256 or m['source']['upstream_commit'] != UPSTREAM_COMMIT for m in manifests):
        raise ValueError('released checkpoint and pinned upstream identity required')
    if a['sequence']['name'] != dataset_row['name']:
        raise ValueError('sequence name mismatch')
    if [f['sha256'] for f in a['sequence']['frames']] != dataset_row.get('frame_sha256'):
        raise ValueError('inference frame order/content differs from hashed dataset')
    for key in ('checkpoint', 'upstream_config_sha256', 'seed', 'sequence'):
        if a[key] != b[key]:
            raise ValueError(f'unmatched run contract: {key}')
    for key in ('torch', 'cuda', 'gpu'):
        if a['environment'][key] != b['environment'][key]:
            raise ValueError(f'unmatched runtime: {key}')
    critical = ['src/hift_research/inference.py', 'scripts/run_sequence.py', 'src/hift_research/geometry.py',
                'src/hift_research/diagnostics/policy.py']
    if any(not a['source']['source_files_sha256'].get(p) or
           a['source']['source_files_sha256'].get(p) != b['source']['source_files_sha256'].get(p) for p in critical):
        raise ValueError('inference source differs')
    logs = [_logs(d / 'diagnostics.jsonl') for d in directories]
    predictions = [np.loadtxt(d / f"{dataset_row['name']}.csv", delimiter=',', ndmin=2) for d in directories]
    gt = np.asarray(dataset_row['ground_truth'], dtype=float)
    n = len(gt)
    if any(len(items) != n for items in logs + predictions):
        raise ValueError('complete equal frame counts required')
    primary = [read_json(d / 'evaluation.json')['sequences'][dataset_row['name']] for d in directories]
    for d, m, pred, score in zip(directories, manifests, predictions, primary):
        if sha256_file(d / f"{dataset_row['name']}.csv") != m['prediction_sha256']:
            raise ValueError('prediction bytes differ from completed inference manifest')
        if score['invalid_policy'] != invalid_policy:
            raise ValueError('evaluator invalid-frame policy differs')
        recalculated = evaluate_sequence(gt, pred, invalid_policy=invalid_policy,
                                        normalized_precision=score['normalized_precision_curve'] is not None)
        if recalculated != score:
            raise ValueError('saved primary evaluation differs from metric recomputation')
    physical = [retrospective_rows(gt, pred, log, annotation_origin=annotation_origin, invalid_policy=invalid_policy)
                for pred, log in zip(predictions, logs)]
    alternate = [retrospective_rows(gt, pred, log, annotation_origin=1-annotation_origin, invalid_policy=invalid_policy)
                 for pred, log in zip(predictions, logs)]
    grids = [np.load(d / 'policy_grids.npz', allow_pickle=False) for d in directories]
    if any(array[key].shape[0] != n-1 for array in grids for key in ('class_probability', 'bce_logits', 'predicted_crop_boxes')):
        for grid in grids:
            grid.close()
        raise ValueError('policy grids must cover every non-initial frame')
    rows, first_difference = [], None
    for index in range(n):
        difference = float(np.max(np.abs(predictions[0][index] - predictions[1][index])))
        changed = difference > 1e-7
        if changed and first_difference is None:
            first_difference = index
        errors = [frame_errors(gt[index], pred[index], invalid_policy=invalid_policy) for pred in predictions]  # primary native numeric convention
        row = {'frame': index, 'bbox_max_abs_difference_px': difference, 'predictions_differ': changed,
               'B0_native_iou': errors[0]['iou'], 'B1_native_iou': errors[1]['iou'],
               'native_iou_B1_minus_B0': errors[1]['iou'] - errors[0]['iou'],
               'GT_source_area_px2': physical[0][index]['source_area_px2']}
        if index:
            same_state = all(np.allclose(logs[0][index]['prior_state'][key], logs[1][index]['prior_state'][key], rtol=0, atol=1e-9)
                             for key in ('center', 'size'))
            response_diff = max(float(np.max(np.abs(grids[0][key][index-1] - grids[1][key][index-1])))
                                for key in ('class_probability', 'bce_logits', 'predicted_crop_boxes'))
            decisions = [logs[0][index]['decision']['B0'], logs[1][index]['decision']['B1-score']]
            ranking_diff = decisions[0]['selected_index'] != decisions[1]['selected_index']
            mechanism = ('same_state_ranking_change' if ranking_diff else 'same_state_smoothing_change') if same_state and changed else (
                         'downstream_state_divergence' if changed else 'same_prediction')
            row.update({'same_prior_state': same_state, 'raw_response_max_abs_difference': response_diff,
                        'observed_selected_indices_differ': ranking_diff, 'difference_context': mechanism,
                        'B0_selected_index': decisions[0]['selected_index'], 'B1_selected_index': decisions[1]['selected_index'],
                        'B0_interpolation_lr': decisions[0]['interpolation_lr'], 'B1_interpolation_lr': decisions[1]['interpolation_lr'],
                        'B0_trajectory_same_state_counterfactual_rankflip': logs[0][index]['decision']['same_state_counterfactual_ranking_differs'],
                        'B1_trajectory_same_state_counterfactual_rankflip': logs[1][index]['decision']['same_state_counterfactual_ranking_differs']})
        rows.append(row)
    for grid in grids:
        grid.close()
    failures = []
    for pred in predictions:
        flags = [frame_errors(g, p, invalid_policy=invalid_policy) for g, p in zip(gt[1:], pred[1:])]
        failures.append({key + '_frames': sum(row[key] for row in flags)
                         for key in ('zero_overlap_failure', 'iou_below_0_1', 'center_error_over_20px')})
    result = {'sequence': dataset_row['name'], 'frames': n, 'primary_metric_coordinates': 'annotation_native_upstream_numeric_no_origin_shift',
              'physical_source_diagnostic_GT_offset': annotation_origin,
              'coordinate_origin_status': 'explicit_MATLAB_consistent_assumption_not_author_origin_guarantee',
              'origin_0_vs_1_sensitivity': {baseline: {
                  'changed_crop_containment_frames': sum(abs(x.get('crop_containment_fraction', 0)-y.get('crop_containment_fraction', 0)) > 1e-9 for x,y in zip(physical[i], alternate[i])),
                  'max_valid_iou_difference': max(abs(x['iou']-y['iou']) for x,y in zip(physical[i], alternate[i]))}
                  for i,baseline in enumerate(('B0', 'B1-score'))},
              'B0': {'metrics': primary[0], 'failures_excluding_init': failures[0], 'runtime': a['metrics'],
                     'source_support': summarize_rows(physical[0])},
              'B1-score': {'metrics': primary[1], 'failures_excluding_init': failures[1], 'runtime': b['metrics'],
                           'source_support': summarize_rows(physical[1])},
              'success_auc_B1_minus_B0': primary[1]['success_auc'] - primary[0]['success_auc'],
              'precision_B1_minus_B0': primary[1]['precision_at_20px'] - primary[0]['precision_at_20px'],
              'differing_prediction_frames': sum(row['predictions_differ'] for row in rows),
              'first_differing_frame_zero_based': first_difference,
              'B1_higher_IoU_frames_excluding_init': sum(row['native_iou_B1_minus_B0'] > 1e-9 for row in rows[1:]),
              'B1_lower_IoU_frames_excluding_init': sum(row['native_iou_B1_minus_B0'] < -1e-9 for row in rows[1:]),
              'run_manifests': [{'path': str(d / 'manifest.json'), 'sha256': sha256_file(d / 'manifest.json')} for d in directories],
              'analysis_input_sha256': {str(d / name): sha256_file(d / name) for d in directories
                                       for name in ('diagnostics.jsonl', 'policy_grids.npz', 'evaluation.json')}}
    return result, rows, physical


def aggregate_matched(sequences):
    return {baseline: aggregate_sequences({row['sequence']: row[baseline]['metrics'] for row in sequences})
            for baseline in ('B0', 'B1-score')}
