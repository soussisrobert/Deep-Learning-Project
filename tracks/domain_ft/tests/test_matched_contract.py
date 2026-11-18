import json
import numpy as np
import pytest
from hift_research import OFFICIAL_CHECKPOINT_SHA256, UPSTREAM_COMMIT
from hift_research.diagnostics.matched import compare_sequence
from hift_research.evaluation.assets import sha256_file
from hift_research.evaluation.metrics import evaluate_sequence


def fixture_runs(root, absent=False):
    gt = [[10, 10, 5, 5], [None]*4 if absent else [10, 10, 5, 5]]
    pred = np.array([[10., 10., 5., 5.]]*2)
    row = {'name': 'sequence', 'ground_truth': gt, 'frame_sha256': ['f'*64, 'e'*64]}
    paths = []
    policy = 'count_as_failure' if absent else 'error'
    for baseline in ('B0', 'B1-score'):
        directory = root / baseline
        directory.mkdir(parents=True)
        paths.append(directory)
        np.savetxt(directory/'sequence.csv', pred, delimiter=',', fmt='%.10g')
        critical = ['src/hift_research/inference.py', 'scripts/run_sequence.py', 'src/hift_research/geometry.py', 'src/hift_research/diagnostics/policy.py']
        manifest = {'baseline': baseline, 'status': 'completed', 'checkpoint': {'sha256': OFFICIAL_CHECKPOINT_SHA256},
                    'source': {'upstream_commit': UPSTREAM_COMMIT, 'source_files_sha256': {p:'a'*64 for p in critical}},
                    'sequence': {'name': 'sequence', 'frames': [{'sha256': value} for value in row['frame_sha256']]},
                    'upstream_config_sha256': 'b'*64, 'seed': 0,
                    'environment': {'torch':'test', 'cuda':None, 'gpu':None}, 'metrics': {},
                    'prediction_sha256': sha256_file(directory/'sequence.csv')}
        (directory/'manifest.json').write_text(json.dumps(manifest))
        score = evaluate_sequence(gt, pred, invalid_policy=policy)
        (directory/'evaluation.json').write_text(json.dumps({'sequences': {'sequence':score}}))
        decision = {'B0': {'selected_index':0, 'interpolation_lr':.2}, 'B1-score': {'selected_index':0, 'interpolation_lr':.21},
                    'same_state_counterfactual_ranking_differs': False}
        logs = [{'frame': i, 'bbox': p.tolist(), 'image_shape': [100,100,3], 'decision': decision,
                 'prior_state': {'center':[12.5,12.5], 'size':[5,5]},
                 'crop_geometry': None if i == 0 else {'origin_x':0, 'origin_y':0,'source_size':30,'model_size':287}}
                for i,p in enumerate(pred)]
        (directory/'diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs))
        np.savez(directory/'policy_grids.npz', class_probability=np.ones((1,2)), bce_logits=np.zeros((1,2)),
                 predicted_crop_boxes=np.ones((1,4,2)))
    return row, paths, policy


@pytest.mark.parametrize('absent', [False, True])
def test_matched_validation_keeps_primary_and_physical_origin_contracts_separate(tmp_path, absent):
    row, paths, policy = fixture_runs(tmp_path, absent)
    result, comparison, physical = compare_sequence(row, *paths, annotation_origin=1, invalid_policy=policy)
    assert result['frames'] == 2 and result['differing_prediction_frames'] == 0
    assert result['B0']['metrics']['invalid_ground_truth_frames'] == int(absent)
    assert physical[0][0]['iou'] < comparison[0]['B0_native_iou'] == 1
    assert 'analysis_input_sha256' in result


@pytest.mark.parametrize('mutation,match', [('checkpoint','checkpoint'), ('frames','frame order'), ('source','source differs'),
                                           ('prediction','prediction bytes'), ('metric','metric recomputation')])
def test_mismatch_or_tampering_rejected(tmp_path, mutation, match):
    row, paths, policy = fixture_runs(tmp_path)
    path = paths[0]/'manifest.json'
    manifest = json.loads(path.read_text())
    if mutation == 'checkpoint':
        manifest['checkpoint']['sha256'] = '0'*64
    elif mutation == 'frames':
        row['frame_sha256'].reverse()
    elif mutation == 'source':
        manifest['source']['source_files_sha256'].pop('src/hift_research/inference.py')
    elif mutation == 'prediction':
        with (paths[0]/'sequence.csv').open('a') as stream:
            stream.write('\n')
    elif mutation == 'metric':
        evaluation = paths[0]/'evaluation.json'
        value = json.loads(evaluation.read_text())
        value['sequences']['sequence']['success_auc'] = .1
        evaluation.write_text(json.dumps(value))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        compare_sequence(row, *paths, annotation_origin=1, invalid_policy=policy)
