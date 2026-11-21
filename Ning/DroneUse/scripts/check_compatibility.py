"""Write an immutable CPU formula-parity probe; never install or execute old torch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hift_research.compatibility import HISTORICAL_SOURCES, run_formula_checks
from hift_research.provenance import git_output, source_identity


def fetch_historical_sources(directory: Path) -> dict:
    """Download official text as inert evidence, checking previously reviewed byte hashes."""
    directory.mkdir(exist_ok=False)
    records = {}
    for name, specification in HISTORICAL_SOURCES.items():
        with urllib.request.urlopen(specification["url"], timeout=30) as response:
            content = response.read(2_000_001)
        digest = hashlib.sha256(content).hexdigest()
        if len(content) > 2_000_000 or digest != specification["sha256"]:
            raise ValueError(f"Historical source identity mismatch: {name}")
        # .txt is intentional: cached historical source is evidence, never imported/executed.
        path = directory / (name + ".txt")
        with path.open("xb") as stream:
            stream.write(content)
        records[name] = {**specification, "bytes": len(content), "cached_as": str(path),
                         "sha256_verified": True}
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New run directory; never reused")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fetch-historical-source", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    sources = (fetch_historical_sources(output / "historical_source")
               if args.fetch_historical_source else {
                   name: {**record, "sha256_verified": False}
                   for name, record in HISTORICAL_SOURCES.items()
               })
    identity = source_identity(ROOT)
    rows = run_formula_checks(args.seed)
    summaries = []
    for operation in sorted({row["operation"] for row in rows}):
        for dtype in ("torch.float32", "torch.float64"):
            selected = [r for r in rows if r["operation"] == operation and r["dtype"] == dtype]
            summaries.append({"operation": operation, "dtype": dtype, "cases": len(selected),
                              "max_abs_error": max(r["max_abs_error"] for r in selected),
                              "max_rms_error": max(r["rms_error"] for r in selected),
                              "passed": all(r["passed"] for r in selected)})
    manifest = {
        "experiment": "historical_formula_modern_cpu_NOT_historical_runtime",
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "logical_cpus": os.cpu_count(), "torch": torch.__version__,
                        "torch_cuda_build_only": torch.version.cuda,
                        "execution_device": "cpu", "threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cuda_initialized": torch.cuda.is_initialized(),
                        "torch_build_configuration": torch.__config__.show()},
        "source_identity": identity, "historical_sources": sources,
        "data": "seeded synthetic tensors, no dataset/checkpoint/training",
        "attention_configuration": {"sequence": 121, "embedding": 192, "heads": 6,
                                    "batch_sizes": [1, 2, 4], "training": False,
                                    "configured_dropout": 0.1, "effective_dropout": 0,
                                    "masks": None, "upstream_need_weights_default": True},
        "normalization_configuration": {"eps": 1e-5, "groupnorm_groups": 32,
                                        "biased_variance": True},
        "summary": summaries, "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "References and comparison both use installed torch CPU primitive kernels.",
            "This is not PyTorch 1.6 runtime, CUDA, gradient, whole-model or checkpoint parity.",
            "No masks, attention bias tokens, different Q/K/V embedding sizes or training dropout.",
            "MHA need_weights=False is supplementary, not pinned HiFT's default inference path.",
            "Crop fixtures are reused in the existing CPU tests, not reimplemented here.",
        ],
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Preserve actual tracked diff as well as its hash; local source hashes cover new files.
    with (output / "tracked.diff").open("xb") as stream:
        stream.write(git_output(ROOT, "diff", "HEAD"))
    print(json.dumps({"status": manifest["status"], "summary": summaries,
                      "manifest": str(output / "manifest.json"),
                      "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
