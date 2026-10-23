"""Independent source-endpoint/physical-negative/grid audit; no model or SGD."""
import argparse
from pathlib import Path
from hift_research.wave4_integrity import independent_pair_algebra
from hift_research.evaluation.assets import write_json_exclusive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--preflight-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt={split:independent_pair_algebra(args.manifest,args.preflight_root/split/'candidate_ledger.jsonl',split=split)
             for split in ('train','development')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json_exclusive(args.output,receipt)
    print({key:value['counts'] for key,value in receipt.items()})
