"""Immutable real temporal candidate ledger and CPU arrays, no model invocation."""
from __future__ import annotations
from collections import Counter
import json
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from hift_research.data_contract.acquisition import write_exclusive_json
from hift_research.data_contract.schema import file_hash
from hift_research.data_contract.preflight import wilson
from hift_research.data_contract.source_ops import ROOT
from .pipeline import TemporalPairPipeline


def run_preflight(manifest_path,policy_path,output,split="train"):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    arrays_dir=output/"arrays";arrays_dir.mkdir()
    pipeline=TemporalPairPipeline(manifest_path,policy_path,split)
    records=[];ledger=output/"candidate_ledger.jsonl"
    with ledger.open("x",encoding="utf-8",newline="\n") as stream:
        for index in range(pipeline.length):
            record,arrays=pipeline.candidate(index)
            if arrays is not None:
                path=arrays_dir/f"{index:06d}.npz"
                with path.open("xb") as file:np.savez_compressed(file,**arrays)
                record.update(array_archive_path=str(path.resolve()),array_archive_sha256=file_hash(path))
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n");stream.flush()
            records.append(record)
            if (index+1)%128==0:print(f"{split}: candidates={index+1} accepted={sum(row['status']=='accepted' for row in records)}",flush=True)
    accepted=[row for row in records if row["status"]=="accepted"]
    negatives=sum(row["candidate_negative"] for row in records)
    accepted_negative=sum(row["candidate_negative"] for row in accepted)
    summary={"schema":"hift-temporal-preflight-v1","split":pipeline.split,"candidate_count":len(records),
             "accepted_count":len(accepted),"rejected_count":len(records)-len(accepted),"candidate_negative_count":negatives,
             "accepted_negative_count":accepted_negative,"accepted_positive_count":len(accepted)-accepted_negative,
             "candidate_negative_fraction":negatives/len(records),"accepted_negative_fraction":accepted_negative/max(len(accepted),1),
             "accepted_fraction_wilson95":wilson(len(accepted),len(records)),
             "candidate_negative_wilson95":wilson(negatives,len(records)) if pipeline.split=="train" else None,
             "candidate_negative_interval_interpretation":"Bernoulli conditional-pilot cohort only" if pipeline.split=="train" else "not_applicable_fixed_balanced_control_schedule",
             "accepted_negative_wilson95_descriptive":wilson(accepted_negative,len(accepted)),
             "rejection_reasons":dict(Counter(row["reason"] for row in records if row["status"]!="accepted")),
             "replacement_count":0,"manifest_path":pipeline.manifest_path,"manifest_sha256":pipeline.manifest_sha256,
             "policy_path":pipeline.policy_path,"policy_sha256":pipeline.policy_sha256,"reservations_sha256":pipeline.reservations_sha256,
             "candidate_ledger_sha256":file_hash(ledger),"cache_statistics":pipeline.cache_statistics(),
             "source_population":{"positives":len(pipeline.population["positives"]),"negatives":len(pipeline.population["negatives"]),
                                  "representation":pipeline.population["representation"],"exclusion_counts":pipeline.population["exclusion_counts"]},
             "nonzero_gap_positives":sum(not row["candidate_negative"] and row["native_gap_abs"]>0 for row in accepted),
             "safe_negative_actual_exclusion_proved":sum(row["candidate_negative"] and row["negative_exclusion_proved"] for row in accepted),
             "no_model_forward_sgd_gpu":True,"production_training_authorized":False,
             "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             "tracked_dirty_diff_sha256":hashlib.sha256(subprocess.check_output(["git","diff","--binary"],cwd=ROOT)).hexdigest(),
             "upstream_revision":subprocess.check_output(["git","-C","third_party/HiFT","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             "environment":{"python":sys.version,"numpy":np.__version__,"opencv":__import__('cv2').__version__,
                            "torch_distribution":importlib.metadata.version('torch'),"platform":platform.platform(),
                            "processor":platform.processor(),"logical_cpu_count":os.cpu_count(),"opencv_threads":__import__('cv2').getNumThreads()},
             "checkpoint":None,"optimizer":None,"model":None}
    write_exclusive_json(output/"summary.json",summary)
    return summary
