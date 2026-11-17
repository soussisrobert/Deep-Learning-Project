"""CPU-only independent recomputation and compact reproducible Wave2 evidence pack."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hift_research.diagnostics.matched import compare_sequence, aggregate_matched
from hift_research.evaluation.assets import read_json, validate_dataset_manifest, sha256_file, write_json_exclusive
from hift_research.provenance import source_identity


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def support_summary(rows):
    available = [r for r in rows if not r['initialization'] and not r['ground_truth_unavailable']]
    def small(r):
        return min(r['source_width_px'], r['source_height_px']) < 16
    def footprint(r):
        return min(r['resized_width_px'], r['resized_height_px']) >= 32
    groups = {}
    for name, select in [('source_min_side_lt_16', small), ('resized_min_side_ge_32', footprint),
                         ('small_source_large_nominal_footprint', lambda r: small(r) and footprint(r)),
                         ('small_source_large_footprint_fully_contained', lambda r: small(r) and footprint(r) and r['crop_containment_fraction'] >= 1-1e-9)]:
        subset = [r for r in available if select(r)]
        groups[name] = {'frames': len(subset), 'mean_physical_diagnostic_iou': float(np.mean([r['iou'] for r in subset])) if subset else None,
                        'measured_zero_overlap_frames': sum(r['zero_overlap_failure'] for r in subset),
                        'not_fully_in_search_crop_frames': sum(r['crop_containment_fraction'] < 1-1e-9 for r in subset)}
    correlations = {}
    for key in ('source_area_px2', 'aspect_ratio', 'motion_px', 'resized_width_px', 'resize_scale', 'crop_containment_fraction'):
        pairs = np.array([(r[key], r['iou']) for r in available if r.get(key) is not None])
        coefficient = (float(np.corrcoef(pairs.T)[0, 1]) if len(pairs) >= 3 and np.std(pairs[:, 0]) > 0 and np.std(pairs[:, 1]) > 0 else None)
        correlations[key] = {'frames': len(pairs), 'pearson_with_physical_diagnostic_iou': coefficient,
                             'inference': 'exploratory_temporally_confounded_no_p_value_no_causality'}
    return {'groups': groups, 'within_sequence_descriptive_correlations': correlations,
            'source_min_side_range_px': [min(min(r['source_width_px'], r['source_height_px']) for r in available),
                                         max(min(r['source_width_px'], r['source_height_px']) for r in available)],
            'optical_detail_entropy_blur_confusion_measured': False}


def policy_summary(logs, baseline):
    logs = logs[1:]
    decisions = [r['decision'][baseline] for r in logs]
    scores = [d['selected_score'] for d in decisions]
    lrs = [d['interpolation_lr'] for d in decisions]
    return {'same_state_counterfactual_rankflip_frames': sum(r['decision']['same_state_counterfactual_ranking_differs'] for r in logs),
            'selected_score_range': [min(scores), max(scores)], 'interpolation_lr_range': [min(lrs), max(lrs)],
            'selected_score_outside_0_1_frames': sum(not 0 <= s <= 1 for s in scores),
            'interpolation_lr_outside_0_1_frames': sum(not 0 <= s <= 1 for s in lrs),
            'raw_logit_range': [min(r['decision']['raw_logit_min'] for r in logs), max(r['decision']['raw_logit_max'] for r in logs)],
            'nonpositive_candidate_size_frames': sum(d['nonpositive_candidate_sizes'] > 0 for d in decisions),
            'nonfinite_candidate_frames': sum(d['nonfinite_candidates'] > 0 for d in decisions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--compatibility', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, help='New compact committed JSON evidence file; never overwrite')
    args = parser.parse_args()
    dataset = validate_dataset_manifest(read_json(args.dataset_manifest))
    acquisition, profile, compatibility = [read_json(p) for p in (args.acquisition, args.profile, args.compatibility)]
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in dataset['sequences']:
        result, comparisons, physical = compare_sequence(row, args.runs / row['name'] / 'B0', args.runs / row['name'] / 'B1-score',
                                                        annotation_origin=1, invalid_policy=dataset['invalid_policy'])
        result['descriptive_support'] = {baseline: support_summary(physical[i]) for i,baseline in enumerate(('B0', 'B1-score'))}
        result['policy_observations'] = {baseline: policy_summary(read_rows(args.runs / row['name'] / baseline / 'diagnostics.jsonl'), baseline)
                                         for baseline in ('B0', 'B1-score')}
        result['observed_selected_index_difference_frames'] = sum(r.get('observed_selected_indices_differ', False) for r in comparisons)
        result['first_difference_observation'] = next((r for r in comparisons if r['predictions_differ']), None)
        for label, records in [('comparison', comparisons), ('B0-source-support', physical[0]), ('B1-score-source-support', physical[1])]:
            with (args.output / f"{row['name']}--{label}.jsonl").open('x', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
        results.append(result)
    aggregate = aggregate_matched(results)
    compact_sequences = []
    for result in results:
        item = {key: value for key,value in result.items() if key not in ('B0', 'B1-score')}
        for baseline in ('B0', 'B1-score'):
            item[baseline] = {**result[baseline], 'metrics': {key: result[baseline]['metrics'][key] for key in
                ('frame_count', 'invalid_ground_truth_frames', 'invalid_prediction_frames', 'success_auc', 'precision_at_20px', 'normalized_precision_at_0_20')}}
        compact_sequences.append(item)
    training_paths = ['src/hift_research/training/graph.py', 'src/hift_research/training/profiling.py',
                      'src/hift_research/corrections/positive_targets.py', 'src/hift_research/corrections/masked_losses.py',
                      'src/hift_research/corrections/classification_loss.py', 'src/hift_research/corrections/tensor_layout.py']
    evidence = {'schema_version': 1, 'kind': 'wave2_verified_pilot_evidence_not_full_benchmark',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'audit_source': source_identity(ROOT),
                'references': {key: {'path': str(p.resolve()), 'sha256': sha256_file(p)} for key,p in
                               [('dataset', args.dataset_manifest), ('acquisition', args.acquisition), ('profile', args.profile),
                                ('compatibility', args.compatibility), ('original_matched_report', args.runs / 'comparison.json')]},
                'dataset': {'name': dataset['dataset'], 'sequences': len(results), 'frames': sum(r['frames'] for r in results),
                            'official_archive': acquisition['source'], 'full_archive_sha256': None,
                            'network_bytes_successful_pilot': acquisition['network_bytes_this_run'],
                            'config_sha256': acquisition['original_config_sha256'], 'readme_sha256': acquisition['original_readme_sha256'],
                            'annotation_sha256': dataset['annotation_sha256'], 'absent_rows_counted_as_failure': 1,
                            'source_annotation_origin': None, 'primary_origin_convention': 'upstream_native_numeric_no_origin_shift',
                            'physical_support_origin': 'assumed_1_with_explicit_0_vs_1_sensitivity',
                            'sampling': 'fixed_short_complete_sequences_before_tracking_not_representative'},
                'aggregate': {baseline: {key: value for key,value in values.items() if key not in ('success_curve', 'precision_curve', 'normalized_precision_curve')}
                              for baseline, values in aggregate.items()},
                'sequences': compact_sequences,
                'training': {'initialization': 'random_not_official_checkpoint', 'settings': profile['settings'],
                             'environment': {key: profile['environment'].get(key) for key in ('torch', 'cuda_build', 'gpu', 'nvidia_smi')},
                             'source_snapshot_sha256': profile['source']['source_tree_sha256'],
                             'current_training_source_matches_measured_snapshot': {p: sha256_file(ROOT / p) == profile['source']['source_files_sha256'].get(p) for p in training_paths},
                             'batches': [{key: r[key] for key in ('status', 'batch_size', 'synthetic_pairs_per_second', 'stage_latency', 'memory', 'total_parameters', 'trainable_parameters')} for r in profile['results']]},
                'compatibility_manifest_summary': {key: value for key,value in compatibility.items() if key in ('summary', 'result', 'comparisons', 'settings')},
                'gates': {'G1': 'PARTIAL', 'G2': 'PASS', 'G3': 'PASS', 'G4': 'PASS', 'G5': 'PARTIAL'},
                'gate_scope': {'G2': 'authentic_complete_selected_sequences_only_not_full_M2', 'G3': 'matched_pilot_comparison_not_improvement',
                               'G4': 'bounded_synthetic_training_feasibility_only_not_production_recipe',
                               'G5': 'geometric_support_measured_but_mechanistic_R1_evidence_insufficient'},
                'limitations': ['3/123 sequences, partial M2', 'historical runtime equivalence not established',
                                'annotation origin unresolved; primary native protocol preserved',
                                'no independent training/pretraining overlap audit', 'no B2 accuracy or real training pipeline',
                                'R1 architecture and full-scale training remain prohibited and unimplemented']}
    write_json_exclusive(args.output / 'evidence.json', evidence)
    if args.summary:
        write_json_exclusive(args.summary, evidence)
    print(json.dumps({'aggregate': evidence['aggregate'],
                      'sequences': [{key: r[key] for key in ('sequence', 'frames', 'success_auc_B1_minus_B0', 'observed_selected_index_difference_frames', 'policy_observations', 'first_difference_observation')} for r in results],
                      'training_source_matches': evidence['training']['current_training_source_matches_measured_snapshot']}, indent=2))


if __name__ == '__main__':
    main()
