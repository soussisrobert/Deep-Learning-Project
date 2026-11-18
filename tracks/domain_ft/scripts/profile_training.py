"""Bounded representative B2 synthetic profile; CUDA requires an explicit lead token."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from hift_research.training.profiling import run_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    parser.add_argument("--gpu-token", help="explicit lead grant identifier; never acquire a GPU implicitly")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts/profiles") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    result = run_profile(output, device=args.device, batches=args.batches, seed=args.seed,
                         warmup=args.warmup, repeats=args.repeats, threads=args.threads,
                         gpu_token=args.gpu_token)
    print(f"Immutable output: {output}")
    return 0 if all(row["status"] == "measured" for row in result["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
