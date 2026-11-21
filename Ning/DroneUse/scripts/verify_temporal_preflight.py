"""CPU-only independent all-record/all-array verification, never model loading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
from hift_research.temporal_contract.verify import verify_run


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",action="append",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    for run in args.run:
        split=json.loads((Path(run)/"summary.json").read_text(encoding="utf-8"))["split"]
        result=verify_run(run,output/(split+"_verification.json"))
        print(json.dumps({key:result[key] for key in ("split","all_checks_passed","all_candidates_replayed","all_accepted_archives_verified")}),flush=True)


if __name__=="__main__":main()
