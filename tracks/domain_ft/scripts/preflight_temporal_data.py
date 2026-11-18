"""Freeze/run the authentic temporal pilot; CPU arrays only, never SGD."""
from __future__ import annotations
import argparse
import json
import cv2
from hift_research.temporal_contract.freeze import freeze_contract
from hift_research.temporal_contract.preflight import run_preflight


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--freeze-output")
    parser.add_argument("--reservations",default="artifacts/wave3-training/FROZEN-001/reservations.json")
    parser.add_argument("--prereg",default="artifacts/wave4-preflight/ELIGIBILITY-PREREG-001.json")
    parser.add_argument("--selection",default="artifacts/wave4-source/SELECTION-001.json")
    parser.add_argument("--policy")
    parser.add_argument("--split",default="train")
    parser.add_argument("--output")
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    if args.freeze_output:
        policies=freeze_contract(args.manifest,args.freeze_output,args.reservations,args.prereg,args.selection)
        print(json.dumps({"frozen_policies":policies}),flush=True)
        if args.output:
            for split,path in policies.items():run_preflight(args.manifest,path,args.output+"/"+split,split)
    elif args.policy and args.output:
        print(json.dumps(run_preflight(args.manifest,args.policy,args.output,args.split)),flush=True)
    else:parser.error("provide --freeze-output or --policy and --output")


if __name__=="__main__":main()
