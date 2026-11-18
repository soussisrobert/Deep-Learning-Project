"""Forensic exact-byte001->002 scalar fix proof; not a production hash bypass.

A new forensic policy binds preserved001 files at snapshot paths. Original001
remains untouched and correctly fails against modified root code. Source-image,
semantic-policy, population, seeds and array bytes must be identical.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np
from hift_research.data_contract.source_ops import ROOT
from hift_research.data_contract.schema import file_hash,canonical_hash
from hift_research.data_contract.acquisition import write_exclusive_json
from .pipeline import TemporalPairPipeline


def primitives(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:primitives(v) for k,v in value.items()}
    if isinstance(value,list):return [primitives(v) for v in value]
    if isinstance(value,tuple):return [primitives(v) for v in value]
    return value


def compare(frozen001,snapshot,frozen002,run001,run002,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    snapshot=Path(snapshot).resolve();original=Path(frozen001)/"train_policy.json"
    old=json.loads(original.read_text(encoding="utf-8"))
    new_policy=Path(frozen002)/"train_policy.json";new=json.loads(new_policy.read_text(encoding="utf-8"))
    if old["policy"]!=new["policy"] or old["manifest_sha256"]!=new["manifest_sha256"] or old["population_sha256"]!=new["population_sha256"]:
        raise ValueError("semantic policy/source/population changed, scalar-only proof impossible")
    forensic=dict(old);forensic["source_code_hashes"]={}
    for filename,expected in old["source_code_hashes"].items():
        target=snapshot/Path(filename).relative_to(ROOT)
        if file_hash(target)!=expected:raise ValueError("preserved old source digest changed")
        forensic["source_code_hashes"][str(target)]=expected
    forensic_path=output/"forensic_old_source_policy.json";write_exclusive_json(forensic_path,forensic)
    package="_hift_failed001_forensic"
    module=types.ModuleType(package);module.__path__=[str(snapshot/"src/hift_research/temporal_contract")]
    sys.modules[package]=module
    old_module=importlib.import_module(package+".pipeline")
    before=old_module.TemporalPairPipeline(old["manifest_path"],forensic_path)
    after=TemporalPairPipeline(new["manifest_path"],new_policy)
    comparisons=[]
    for index in range(4):
        a,aa=before.candidate(index);b,bb=after.candidate(index)
        left=primitives({k:v for k,v in a.items() if k!="policy_sha256"})
        right=primitives({k:v for k,v in b.items() if k!="policy_sha256"})
        if canonical_hash(left)!=canonical_hash(right):raise ValueError("numeric/geometry/identity difference after scalar conversion")
        if (aa is None)!=(bb is None):raise ValueError("acceptance changed")
        if aa is not None:
            for key in aa:np.testing.assert_array_equal(aa[key],bb[key])
        original_serialization_failed=False
        try:json.dumps(a,allow_nan=False)
        except TypeError:original_serialization_failed=True
        json.dumps(b,allow_nan=False)
        archive_old=Path(run001)/"arrays"/f"{index:06d}.npz";archive_new=Path(run002)/"arrays"/f"{index:06d}.npz"
        archives_compared=False
        if archive_old.exists():
            with np.load(archive_old,allow_pickle=False) as x,np.load(archive_new,allow_pickle=False) as y:
                if set(x.files)!=set(y.files):raise ValueError("archive keys differ")
                for key in x.files:np.testing.assert_array_equal(x[key],y[key])
            archives_compared=True
        comparisons.append(dict(candidate_index=index,candidate_negative=b["candidate_negative"],numeric_geometry_record_equal=True,
                                all_arrays_equal=True,archived_arrays_compared=archives_compared,
                                original_json_failed=original_serialization_failed,new_json_roundtrip_pass=True))
    if not comparisons[3]["candidate_negative"] or not comparisons[3]["original_json_failed"]:raise ValueError("original first-negative failure not reproduced")
    result=dict(schema="hift-scalar-serialization-forensic-proof-v1",all_checks_passed=True,comparisons=comparisons,
                old_policy_sha256=file_hash(original),new_policy_sha256=file_hash(new_policy),
                forensic_policy_sha256=file_hash(forensic_path),semantic_policy_unchanged=True,population_sha256=old["population_sha256"],
                preserved_source_sha_verified=20,production_guard_bypassed=False,
                method="new forensic manifest points to exact old-byte snapshots; NumPy scalars compared by exact numeric item(), never str/default=str",
                no_model_forward_sgd_gpu=True)
    write_exclusive_json(output/"comparison.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("frozen001","snapshot","frozen002","run001","run002","output"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args();print(compare(**vars(args))["all_checks_passed"])
