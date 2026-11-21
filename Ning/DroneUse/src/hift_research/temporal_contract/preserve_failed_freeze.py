"""Reconstruct exact failed001 source bytes and verify precommitted hashes.

Only three scalar-conversion edits occurred before this preservation. Every
reconstructed file must match the immutable FROZEN001 source SHA, or stop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from hift_research.data_contract.source_ops import ROOT
from hift_research.data_contract.acquisition import write_exclusive_json
from hift_research.data_contract.schema import file_hash


def preserve(frozen,output):
    frozen=Path(frozen);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    policy=json.loads((frozen/"train_policy.json").read_text(encoding="utf-8"))
    substitutions={
        "population.py":[("return list(map(float,[source[0]-source_margin,source[1]-source_margin,source[2]+source_margin,source[3]+source_margin]))",
                          "return [source[0]-source_margin,source[1]-source_margin,source[2]+source_margin,source[3]+source_margin]")],
        "pipeline.py":[("negative_overlap=float(intersection(visible_t,guarded)) if negative else None",
                        "negative_overlap=intersection(visible_t,guarded) if negative else None"),
                       ("negative_exclusion_proved=bool(negative_overlap==0) if negative else None,",
                        "negative_exclusion_proved=(negative_overlap==0) if negative else None,")]}
    receipt=[]
    for filename,expected in policy["source_code_hashes"].items():
        path=Path(filename);payload=path.read_bytes();changes=[]
        if "/temporal_contract/" in path.as_posix() and path.name in substitutions:
            for newer,older in substitutions[path.name]:
                if payload.count(newer.encode())!=1:raise ValueError("reconstruction literal not unique")
                payload=payload.replace(newer.encode(),older.encode());changes.append({"failed001":older,"corrected002":newer})
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError("failed-source SHA reconstruction mismatch: "+filename)
        target=output/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as stream:stream.write(payload)
        receipt.append(dict(original_path=filename,snapshot_path=str(target.resolve()),sha256=expected,exact_edits=changes))
    write_exclusive_json(output/"snapshot_receipt.json",dict(schema="hift-failed-freeze-source-snapshot-v1",files=receipt,
                                                           all_precommitted_hashes_match=True,original_train_policy_sha256=file_hash(frozen/"train_policy.json"),
                                                           method="exact three scalar-conversion edits inverted; every byte authenticated by frozen001 SHA"))
    return len(receipt)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--frozen",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();print("verified_source_snapshots",preserve(args.frozen,args.output))
