"""Export GT-free inputs preserving literal released-tool numeric initialization.

This does not normalize the annotation origin. The tracker consumes raw x/y +0.5
as the pinned tools/test.py does; primary evaluation must use native GT numbers.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hift_research.evaluation.assets import read_json, validate_dataset_manifest, write_json_exclusive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_dataset_manifest(read_json(args.dataset_manifest))
    args.output.mkdir(parents=True, exist_ok=False)
    for row in manifest['sequences']:
        x, y, w, h = row['init_box']
        value = {'name': row['name'], 'dataset': manifest['dataset'],
                 'frames': [str((Path(manifest['root']) / path).resolve()) for path in row['frame_paths']],
                 'initial_bbox': [x+.5, y+.5, w, h], 'coordinate_origin': 0,
                 'annotation_coordinate_origin': manifest.get('annotation_coordinate_origin'),
                 'annotation_to_tracker_transform': 'upstream_numeric_passthrough_plus_half_pixel_no_origin_shift',
                 'initialization_convention': 'literal_pinned_tools_test_get_axis_aligned_bbox_no_origin_conversion',
                 'evaluation_ground_truth_included': False}
        write_json_exclusive(args.output / f"{row['name']}.json", value)
    print(f"Exported {len(manifest['sequences'])} native-numeric initialization-only inputs: {args.output}")


if __name__ == '__main__':
    main()
