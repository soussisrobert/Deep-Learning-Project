"""CPU-only check that recorded crops equal pinned tracker rounding/origin mathematics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hift_research.geometry import CropGeometry
from hift_research.evaluation.assets import read_json, sha256_file, write_json_exclusive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = ROOT/'third_party/HiFT/experiments/config.yaml'
    import yaml
    released = yaml.safe_load(config.read_text())
    context = released['TRACK']['CONTEXT_AMOUNT']
    model_size, exemplar = released['TRACK']['INSTANCE_SIZE'], released['TRACK']['EXEMPLAR_SIZE']
    records, failures, count = [], [], 0
    for manifest_path in sorted(args.runs.glob('*/*/manifest.json')):
        manifest = read_json(manifest_path)
        if manifest['upstream_config_sha256'] != sha256_file(config):
            raise ValueError('recorded run used a different upstream config')
        with manifest_path.with_name('diagnostics.jsonl').open() as stream:
            logs = [json.loads(line) for line in stream]
        initial_size = np.array(logs[0]['bbox'][2:])
        initial_z = round(np.sqrt(np.prod(initial_size + context*initial_size.sum())))
        for row in logs[1:]:
            size = np.array(row['prior_state']['size'])
            z = np.sqrt(np.prod(size + context*size.sum()))
            if size.prod() > .5*np.prod(row['image_shape'][:2]):
                z = initial_z
            source_size = round(z*(model_size/exemplar))  # literal original association
            geometry = CropGeometry.from_tracker(row['prior_state']['center'], source_size, model_size)
            expected = {'origin_x': geometry.origin_x, 'origin_y': geometry.origin_y,
                        'source_size': source_size, 'model_size': model_size}
            if any(row['crop_geometry'][key] != value for key,value in expected.items()):
                failures.append({'run': str(manifest_path.parent), 'frame':row['frame']})
            count += 1
        records.append({'run_manifest': str(manifest_path), 'sha256':sha256_file(manifest_path)})
    result = {'kind':'recorded_crop_geometry_reverification', 'tracked_rows':count, 'runs':records,
              'failures':failures, 'passed':not failures,
              'note':'Checks observer geometry only, no GT or GPU; final observer uses the literal grouped original ratio.'}
    write_json_exclusive(args.output, result)
    print(json.dumps({'tracked_rows': count, 'failures': failures, 'passed':not failures}))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
