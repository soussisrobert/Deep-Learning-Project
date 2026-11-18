"""Verify the immutable Wave2 handoff by hashes, without rerunning experiments."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hift_research import OFFICIAL_CHECKPOINT_SHA256, UPSTREAM_COMMIT
from hift_research.evaluation.assets import read_json, sha256_file, write_json_exclusive
from hift_research.provenance import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = read_json(ROOT / 'experiments/wave2_validation.json')
    checks = {}
    for name, reference in evidence['references'].items():
        actual = sha256_file(reference['path'])
        if actual != reference['sha256']:
            raise ValueError(f'Wave2 reference changed: {name}')
        checks[name] = {**reference, 'verified': True}
    checkpoint = ROOT / 'artifacts/checkpoints/general_model.pth'
    if sha256_file(checkpoint) != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError('official checkpoint identity changed')
    current = source_identity(ROOT)
    if current['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('upstream mismatch')
    result = {'kind': 'wave3_start_immutable_wave2_verification',
              'initial_verified_head': 'fba88e5b3452bf6e92980f14a52604d5d93eb0b7',
              'initial_tool_verification': 'git status empty; upstream clean;254 tests passed13.64s;pip check passed',
              'current_source_snapshot': current, 'wave2_reference_checks': checks,
              'checkpoint_sha256': OFFICIAL_CHECKPOINT_SHA256,
              'frozen_baseline_config_sha256': {name:sha256_file(ROOT / f'configs/{name}.json') for name in ('B0','B1-score')},
              'no_experiments_rerun': True}
    write_json_exclusive(args.output, result)
    print(f'Verified {len(checks)} immutable Wave2 references, official checkpoint and pinned source')


if __name__ == '__main__':
    main()
