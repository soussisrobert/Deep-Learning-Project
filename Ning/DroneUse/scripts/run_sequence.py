"""Original HiFT one-pass inference; manifest contains initialization only, never later GT."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hift_research import OFFICIAL_CHECKPOINT_SHA256
from hift_research.evaluation.assets import read_json
from hift_research.geometry import CropGeometry
from hift_research.inference import load_original, make_tracker
from hift_research.provenance import new_run, runtime_identity, sha256, source_identity


def read_sequence(path):
    import numpy as np
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError('Sequence input must be a JSON object')
    allowed = {'name', 'dataset', 'frames', 'initial_bbox', 'coordinate_origin', 'initialization_convention',
               'annotation_coordinate_origin', 'evaluation_ground_truth_included', 'annotation_to_tracker_transform'}
    if data.get('evaluation_ground_truth_included', False) is not False:
        raise ValueError('Inference inputs cannot include evaluation ground truth')
    if set(data) - allowed:
        raise ValueError(f'Unknown or forbidden fields (GT must not enter inference): {set(data) - allowed}')
    if ('annotation_to_tracker_transform' in data and data['annotation_to_tracker_transform'] !=
            'upstream_numeric_passthrough_plus_half_pixel_no_origin_shift'):
        raise ValueError('Only the declared literal upstream initialization transform is supported')
    if data.get('coordinate_origin') != 0:
        raise ValueError('Explicit zero-based conversion required; no guessed dataset coordinates')
    if not isinstance(data.get('name'), str) or not data['name'] or any(c in data['name'] for c in '/\\:') or data['name'] in {'.', '..'}:
        raise ValueError('Sequence name must be a safe basename')
    if not isinstance(data.get('dataset'), str) or not data['dataset']:
        raise ValueError('dataset identity required')
    if not isinstance(data.get('frames'), list) or not data['frames'] or not all(isinstance(f, str) and f for f in data['frames']):
        raise ValueError('Explicit ordered nonempty frame list required')
    bbox = np.asarray(data.get('initial_bbox'), dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= 0):
        raise ValueError('finite initial xywh with positive size required')
    frames = [(path.parent / f).resolve() for f in data['frames']]
    if not all(f.is_file() for f in frames):
        raise FileNotFoundError('A listed sequence image is unavailable')
    return data, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/B0.json')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-id', help='Globally unique run label; defaults to the new output directory name')
    parser.add_argument('--record-policy', action='store_true', help='Observe raw response/selection grids after timing; never alter decisions')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    sequence, frames = read_sequence(args.sequence.resolve())
    config = read_json(args.config)
    if config['identity'] not in {'B0', 'B1-score'}:
        raise ValueError('B2/R1 are not validated or runnable')
    checkpoint = args.checkpoint or ROOT / config['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_digest = sha256(checkpoint)
    if config.get('checkpoint_sha256') != OFFICIAL_CHECKPOINT_SHA256 or checkpoint_digest != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError('B0/B1-score require the hash-pinned released checkpoint, not substitute weights')
    output = new_run(args.output)
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_original(ROOT, checkpoint, ROOT / config['upstream_config'])
    from pysot.core.config import cfg
    recorder = None
    if args.record_policy:
        from hift_research.diagnostics.policy import RawOutputRecorder, observe_decision
        recorder = RawOutputRecorder(model)
    tracker = make_tracker(recorder or model, config['identity'])
    manifest = {
        'schema_version': 1, 'kind': 'inference_run', 'experiment_id': args.experiment_id or output.name,
        'baseline': config['identity'], 'status': 'running', 'source': source_identity(ROOT),
        'environment': runtime_identity(), 'configuration': config,
        'config_sha256': sha256(args.config), 'upstream_config_sha256': sha256(ROOT / config['upstream_config']),
        'checkpoint': {'path': str(checkpoint.resolve()), 'sha256': sha256(checkpoint)},
        'sequence': {'name': sequence['name'], 'dataset': sequence['dataset'],
                     'manifest_sha256': sha256(args.sequence),
                     'initialization_convention': sequence.get('initialization_convention', 'explicit_tracker_ready_bbox'),
                     'frames': [{'path': str(f), 'sha256': sha256(f)} for f in frames]},
        'seed': seed, 'parameter_count': sum(p.numel() for p in model.parameters()), 'metrics': {},
        'policy_recording': bool(args.record_policy),
        'timing_contract': 'tracking=crop+model+postprocess+support instrumentation; end_to_end adds image decode; both exclude post-timing decision observer/output writes',
        'limitations': ['modern torch numerical equivalence to 2021 environment unverified',
                        'single local GPU shared with desktop applications',
                        'initialization included in saved predictions; no warmup excluded from sequence runtime',
                        'source_support fields use previous prediction, not ground truth or measured image detail'],
    }
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    times, e2e_times, predictions, policy_grids = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    sequence_wall_start = time.perf_counter()
    try:
        with (output / 'diagnostics.jsonl').open('x', encoding='utf-8') as diagnostic_stream:
            for index, frame in enumerate(frames):
                wall_start = time.perf_counter()
                image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'Cannot decode frame {frame}')
                torch.cuda.synchronize()
                start = time.perf_counter()
                support, crop_record, prior_state, decision = None, None, None, None
                if index == 0:
                    bbox = sequence['initial_bbox']
                    tracker.init(image, bbox)
                    score = None
                else:
                    # Reconstruct actual search geometry from PREVIOUS predicted state only.
                    wz = tracker.size[0] + cfg.TRACK.CONTEXT_AMOUNT * tracker.size.sum()
                    hz = tracker.size[1] + cfg.TRACK.CONTEXT_AMOUNT * tracker.size.sum()
                    sz = np.sqrt(wz * hz)
                    if tracker.size.prod() > .5 * image.shape[0] * image.shape[1]:
                        sz = tracker.scaleaa
                    sx = sz * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
                    geometry = CropGeometry.from_tracker(tracker.center_pos, round(sx), cfg.TRACK.INSTANCE_SIZE)
                    prior_size = tracker.size.copy()
                    prior_state = {'center': tracker.center_pos.tolist(), 'size': prior_size.tolist()}
                    crop_record = {'origin_x': geometry.origin_x, 'origin_y': geometry.origin_y,
                                   'source_size': geometry.source_size, 'model_size': geometry.model_size,
                                   'ideal_decoder_scale_z': float(cfg.TRAIN.EXEMPLAR_SIZE / sz)}
                    previous_bbox = [*(tracker.center_pos - tracker.size / 2), *tracker.size]
                    support = geometry.support(previous_bbox, image.shape)
                    support['estimated_from'] = 'previous_prediction_not_GT'
                    result = tracker.track(image)
                    bbox, score = result['bbox'], float(result['best_score'])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                end_to_end = time.perf_counter() - wall_start
                if not np.isfinite(bbox).all() or np.any(np.asarray(bbox)[2:] <= 0):
                    raise ValueError(f'Invalid prediction at frame {index}; no silent fallback')
                if index and recorder is not None:
                    decision, grids = observe_decision(tracker, recorder, prior_size, cfg.TRAIN.EXEMPLAR_SIZE / sz, cfg)
                    if not np.isclose(decision[config['identity']]['selected_score'], score, rtol=1e-6, atol=1e-7):
                        raise AssertionError('Observer does not reproduce original tracker score')
                    policy_grids.append(grids)
                times.append(elapsed)
                e2e_times.append(end_to_end)
                predictions.append(bbox)
                diagnostic_stream.write(json.dumps({'frame': index, 'bbox': list(map(float, bbox)), 'score': score,
                                                    'tracking_latency_ms': elapsed * 1000, 'end_to_end_latency_ms': end_to_end * 1000,
                                                    'image_shape': list(image.shape), 'crop_geometry': crop_record,
                                                    'prior_state': prior_state, 'decision': decision,
                                                    'prior_prediction_support': support}, allow_nan=False) + '\n')
        prediction_path = output / f"{sequence['name']}.csv"
        np.savetxt(prediction_path, np.asarray(predictions), delimiter=',', fmt='%.10g')
        if policy_grids:
            np.savez_compressed(output / 'policy_grids.npz',
                                **{key: np.stack([g[key] for g in policy_grids]) for key in policy_grids[0]})
        tracking = times[1:]
        e2e_tracking = e2e_times[1:]
        manifest.update({'status': 'completed', 'prediction_sha256': sha256(prediction_path),
                         'frames_completed': len(predictions),
                         'metrics': {'tracking_fps': len(tracking) / sum(tracking) if tracking else None,
                                     'end_to_end_fps': len(e2e_tracking) / sum(e2e_tracking) if e2e_tracking else None,
                                     'tracking_latency_ms_mean': float(np.mean(tracking) * 1000) if tracking else None,
                                     'end_to_end_latency_ms_mean': float(np.mean(e2e_tracking) * 1000) if e2e_tracking else None,
                                     'init_latency_ms': times[0] * 1000,
                                     'sequence_wall_fps_including_init_diagnostics_and_saving': len(predictions) / (time.perf_counter() - sequence_wall_start),
                                     'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                                     'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                                     'flops_macs': None}})
    except Exception as exc:
        manifest.update({'status': 'failed', 'error': repr(exc), 'frames_completed': len(predictions)})
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'run': str(output), 'status': manifest['status'], 'metrics': manifest['metrics']}, indent=2))


if __name__ == '__main__':
    main()
