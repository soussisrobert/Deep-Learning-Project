"""FUTURE command only. Wave4 does NOT authorize real-image SGD or create a grant."""
import argparse
import sys
from pathlib import Path

from hift_research.training.recipe import ROOT
from hift_research.training.wave4_pilot import launch_future_pilot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/B2-Pilot-W4.json")
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--ack-real-sgd", action="store_true", help="future separate explicit human authorization is ALSO required")
    parser.add_argument("--output", type=Path, required=True, help="new exclusive immutable run directory")
    parser.add_argument("--resume", type=Path, help="atomic committed effective-window boundary only")
    parser.add_argument("--resume-sha256", help="exact previously recorded checkpoint SHA256; mandatory with --resume")
    args = parser.parse_args(argv)
    try:
        launch_future_pilot(args.config, authorization_file=args.authorization_file, ack_real_sgd=args.ack_real_sgd,
                            output=args.output, resume=args.resume, resume_sha256=args.resume_sha256)
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
