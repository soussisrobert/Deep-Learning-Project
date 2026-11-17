"""One GPU at a time: fixed real subset, matched native-numeric B0/B1, retrospective GT join."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hift_research import OFFICIAL_CHECKPOINT_SHA256
from hift_research.evaluation.assets import read_json, sha256_file, validate_dataset_manifest, write_json_exclusive
from hift_research.diagnostics.matched import compare_sequence, aggregate_matched
from hift_research.provenance import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--inference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-annotation-origin', type=int, choices=[0, 1], required=True)
    parser.add_argument('--gpu-token', required=True, help='Lead-exclusive GPU allocation identifier')
    args = parser.parse_args()
    dataset = validate_dataset_manifest(read_json(args.dataset_manifest))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {'schema_version': 1, 'experiment_id': output.name, 'kind': 'matched_real_sequence_pilot',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'source': source_identity(ROOT), 'dataset_manifest': {'path': str(args.dataset_manifest.resolve()), 'sha256': sha256_file(args.dataset_manifest)},
            'checkpoint_sha256': OFFICIAL_CHECKPOINT_SHA256, 'gpu_token': args.gpu_token,
            'sequences': [row['name'] for row in dataset['sequences']],
            'selection': 'fixed complete sequences selected by acquisition before any inference; not accuracy-selected',
            'primary_evaluation': 'native annotation numbers, original +half-pixel initialization, no origin shift',
            'physical_diagnostic_GT_origin': args.physical_annotation_origin,
            'execution_order': {row['name']: (['B0', 'B1-score'] if i % 2 == 0 else ['B1-score', 'B0']) for i, row in enumerate(dataset['sequences'])},
            'limitations': ['small deliberately selected subset, not benchmark-level reproduction',
                            'sequential single-GPU runs with desktop GUI activity; latency is descriptive not a speed comparison',
                            'normalized precision supplemental only', 'historical torch runtime parity unverified',
                            'post-timing raw-response observer adds wall-clock overhead without changing tracker decisions']}
    write_json_exclusive(output / 'subset_plan.json', plan)
    results = []
    for row in dataset['sequences']:
        sequence_path = args.inference_dir / f"{row['name']}.json"
        for baseline in plan['execution_order'][row['name']]:
            run = output / row['name'] / baseline
            subprocess.run([sys.executable, str(ROOT / 'scripts/run_sequence.py'), '--sequence', str(sequence_path),
                            '--config', str(ROOT / f'configs/{baseline}.json'), '--output', str(run), '--record-policy',
                            '--experiment-id', f"{output.name}--{row['name']}--{baseline}"], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate.py'), '--dataset-manifest', str(args.dataset_manifest.resolve()),
                            '--predictions', str(run / f"{row['name']}.csv"), '--sequence', row['name'],
                            '--output', str(run / 'evaluation.json'), '--run-manifest', str(run / 'manifest.json'),
                            '--prediction-coordinate-origin', 'annotation_native', '--normalized-precision',
                            '--invalid-policy', dataset['invalid_policy']], cwd=ROOT, check=True)
        result, comparisons, physical = compare_sequence(row, output / row['name'] / 'B0', output / row['name'] / 'B1-score',
                                                        annotation_origin=args.physical_annotation_origin, invalid_policy=dataset['invalid_policy'])
        results.append(result)
        for name, records in [('comparison', comparisons), ('B0-source-support', physical[0]), ('B1-score-source-support', physical[1])]:
            with (output / row['name'] / f'{name}.jsonl').open('x', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
    report = {**plan, 'status': 'completed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'sequence_results': results, 'aggregate': aggregate_matched(results),
              'source_support_inference': 'descriptive_only; source-size/sequence/crop/motion confounding unadjusted; no R1 mechanism established'}
    write_json_exclusive(output / 'comparison.json', report)
    print(json.dumps({'output': str(output), 'sequences': len(results), 'frames': sum(r['frames'] for r in results),
                      'aggregate': {key: {m: report['aggregate'][key][m] for m in ('success_auc', 'precision_at_20px')}
                                    for key in ('B0', 'B1-score')}}, indent=2))


if __name__ == '__main__':
    main()
