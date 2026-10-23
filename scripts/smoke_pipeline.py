"""Three synthetic frames exercise BOTH complete GPU tracker/CLI paths, never a benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import cv2
    import jsonschema
    import numpy as np
    rng = np.random.default_rng(20261023)
    frames = []
    background = rng.integers(0, 128, (200, 240, 3), dtype=np.uint8)
    for index in range(3):
        image = background.copy()
        cv2.rectangle(image, (60 + index * 2, 50), (83 + index * 2, 69), (210, 240, 220), -1)
        path = output / f'{index:04d}.png'
        assert cv2.imwrite(str(path), image)
        frames.append(path.name)
    sequence = output / 'synthetic.json'
    sequence.write_text(json.dumps({'name': 'synthetic', 'dataset': 'synthetic_NOT_a_benchmark',
                                   'frames': frames, 'initial_bbox': [60, 50, 24, 20],
                                   'coordinate_origin': 0, 'initialization_convention': 'explicit_tracker_ready_bbox'}), encoding='utf-8')
    schema = json.loads((ROOT / 'experiments/schema.json').read_text(encoding='utf-8'))
    for baseline in ('B0', 'B1-score'):
        run = output / baseline
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_sequence.py'), '--sequence', str(sequence),
                        '--config', str(ROOT / f'configs/{baseline}.json'), '--output', str(run)], cwd=ROOT, check=True)
        record = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        jsonschema.Draft202012Validator(schema).validate(record)
        predictions = np.loadtxt(run / 'synthetic.csv', delimiter=',')
        assert predictions.shape == (3, 4) and np.isfinite(predictions).all()
        assert record['status'] == 'completed' and record['baseline'] == baseline
    print('PASS: B0 and B1-score complete synthetic GPU pipelines, 3 frames each, schema validated. NOT a benchmark.')


if __name__ == '__main__':
    main()
