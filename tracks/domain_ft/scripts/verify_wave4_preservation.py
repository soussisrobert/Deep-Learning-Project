"""Verify frozen Wave2/3 assets and code without executing their experiments."""
import argparse
from pathlib import Path
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hift_research.wave4_integrity import verify_prior
from hift_research.evaluation.assets import write_json_exclusive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt={'kind':'wave4_prior_delivery_preservation_no_experiments',
             'created_utc':datetime.now(timezone.utc).isoformat(),**verify_prior(ROOT)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json_exclusive(args.output,receipt)
    print(f'Preserved {receipt["protected_prior_file_count"]} Wave3 delivery files,21 Wave3 and5 Wave2 references; checkpoint/upstream unchanged')


if __name__=='__main__':main()
