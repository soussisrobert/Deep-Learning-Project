"""GPU execution correctness only; synthetic inputs are not tracking evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hift_research.inference import load_original
from hift_research.provenance import new_run, runtime_identity, sha256, source_identity


def execute(args, report):
    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_original(ROOT, args.checkpoint)
    report['checkpoint_load'] = 'strict_model_keys_with_restricted_numpy_metadata_allowlist' if args.checkpoint else 'random_weights'
    z = torch.rand(1, 3, 127, 127, device='cuda') * 255
    x = torch.rand(1, 3, 287, 287, device='cuda') * 255
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model.template(z)
        for _ in range(2):
            model.track(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            result = model.track(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    expected = {'loc': [1, 4, 11, 11], 'cls1': [1, 2, 11, 11], 'cls2': [1, 1, 11, 11]}
    actual = {k: list(v.shape) for k, v in result.items()}
    assert actual == expected, actual
    assert all(torch.isfinite(v).all().item() for v in result.values())
    from hift_research import OFFICIAL_CHECKPOINT_SHA256
    report.update({
        'status': 'completed', 'output_shapes': actual, 'all_outputs_finite': True,
        'weights': 'provided_checkpoint_strictly_loaded' if args.checkpoint else 'random_NOT_a_reproduction',
        'official_checkpoint_hash_match': report['checkpoint_sha256'] == OFFICIAL_CHECKPOINT_SHA256,
        'ranges': {k: [v.min().item(), v.max().item()] for k, v in result.items()},
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'synthetic_search_forward_mean_ms': elapsed / 5 * 1000,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    output = new_run(args.output)
    report = {
        'schema_version': 1, 'experiment_id': output.name,
        'kind': 'synthetic_GPU_execution_not_benchmark', 'baseline': 'B0', 'status': 'running',
        'source': source_identity(ROOT), 'environment': runtime_identity(), 'seed': args.seed,
        'weights': 'provided_checkpoint_pending_load' if args.checkpoint else 'random_NOT_a_reproduction',
        'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
        'config_sha256': sha256(ROOT / 'third_party/HiFT/experiments/config.yaml'),
        'timing_scope': 'five warmed search model forwards only; excludes crop/postprocess/I/O; not tracking FPS',
        'warmup': 2, 'repeats': 5, 'flops_macs': None,
        'limitations': ['modern torch numerical equivalence with original environment unverified',
                        'desktop GPU shared with GUI applications', 'synthetic input not benchmark performance'],
    }
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        execute(args, report)
    except Exception as exc:
        report.update({'status': 'failed', 'error': repr(exc)})
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('experiment_id', 'status', 'checkpoint_load',
                     'output_shapes', 'parameter_count', 'peak_cuda_allocated_bytes',
                     'peak_cuda_reserved_bytes', 'synthetic_search_forward_mean_ms')}, indent=2))


if __name__ == '__main__':
    main()
