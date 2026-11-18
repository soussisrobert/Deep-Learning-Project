"""Descriptive source/resize/support/cell-weight audit of immutable1024 cohort."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .schema import file_hash
from .acquisition import write_exclusive_json


def describe(values):
    values=np.asarray(values,dtype=np.float64)
    if values.size==0:
        return {"count":0}
    probabilities=[0,.01,.05,.25,.5,.75,.95,.99,1]
    return {"count":int(values.size),"mean":float(values.mean()),
            "quantiles":dict(zip(("min","p01","p05","p25","p50","p75","p95","p99","max"),
                                 map(float,np.quantile(values,probabilities))))}


def sizes(rows):
    values=np.asarray(rows,dtype=float)
    return {"width_px":describe(values[:,0]),"height_px":describe(values[:,1]),
            "area_px2":describe(values[:,0]*values[:,1]),"aspect_w_over_h":describe(values[:,0]/values[:,1])}


def audit_distributions(run_path, output):
    run=Path(run_path)
    summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    independent_path=run/"independent_verification.json"
    independent=json.loads(independent_path.read_text(encoding="utf-8"))
    if not independent["all_checks_passed"] or not independent["all_1024_candidates_fresh_replay_equal"]:
        raise ValueError("independent artifact/replay verification required")
    ledger_path=run/"candidate_ledger.jsonl"
    if file_hash(ledger_path)!=summary["candidate_ledger_sha256"]:
        raise ValueError("candidate ledger changed")
    manifest_path=Path(summary["manifest_path"])
    if file_hash(manifest_path)!=summary["manifest_sha256"]:
        raise ValueError("training manifest changed")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    records=[json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    accepted=[row for row in records if row["status"]=="accepted"]
    active_weights,all_weights,cls2values=[],[],[]
    counts=Counter()
    pair_counts=[]
    for record in accepted:
        with np.load(record["array_archive_path"],allow_pickle=False) as archive:
            weight,cls1,cls2=archive["weightxff"],archive["label_cls1"],archive["labelcls2"]
            active_weights.extend(weight[weight>0].tolist())
            all_weights.extend(weight.reshape(-1).tolist())
            cls2values.extend(cls2.reshape(-1).tolist())
            row={"candidate_index":record["candidate_index"],"regression_active":int(np.count_nonzero(weight>0)),
                 "cls1_positive":int(np.count_nonzero(cls1==1)),"cls1_background":int(np.count_nonzero(cls1==0)),
                 "cls1_ignored":int(np.count_nonzero(cls1<0)),"cls1_active":int(np.count_nonzero(cls1>=0)),
                 "cls2_positive":int(np.count_nonzero(cls2>0)),"cls2_dense_loss_mask":int(cls2.size),
                 "regression_zero":int(np.count_nonzero(weight==0)),"weight64":int(np.count_nonzero(weight==64))}
            pair_counts.append(row)
            for key,value in row.items():
                if key!="candidate_index":counts[key]+=value
            for code in (-2,-1,0,1):counts[f"cls1_label_{code}"]+=int(np.count_nonzero(cls1==code))
            counts["positive_empty_regression"]+=not np.any(weight>0)
            counts["positive_empty_cls1"]+=not np.any(cls1==1)
            counts["positive_empty_cls2"]+=not np.any(cls2>0)
    instance_counts=Counter(str(row["template_annotation_id"]) for row in records)
    image_counts=Counter(str(row["template_image_id"]) for row in records)
    active=np.asarray(active_weights)
    result={"schema":"hift-real-cohort-distributions-v1","scope":"973acceptedpositives_from1024fixedattempts_not_new_sampling",
            "manifest_sha256":summary["manifest_sha256"],"candidate_ledger_sha256":summary["candidate_ledger_sha256"],
            "independent_verification_sha256":file_hash(independent_path),"accepted_count":len(accepted),
            "source_instance_sizes_candidate_weighted":sizes([row["source_object_size_px"] for row in accepted]),
            "source_instance_sizes_all209_unweighted":sizes([row["bbox_xywh"][2:] for row in manifest["instances"]]),
            "source_visible_object_support_px2":describe([row["genuine_visible_object_support_px2"] for row in accepted]),
            "augmented_search_label_sizes_candidate_weighted":sizes([row["augmented_object_size_px"] for row in accepted]),
            "centered511_label_sizes_candidate_weighted":sizes([row["centered_crop_object_size_px"] for row in accepted]),
            "source_crop511_side_px":describe([row["source_crop511_side_px"] for row in accepted]),
            "cell_counts_total":dict(counts),"per_pair_support_counts":{key:describe([row[key] for row in pair_counts])
                                                                       for key in pair_counts[0] if key!="candidate_index"},
            "regression_weights_all_active_cells":describe(active),"regression_weights_all_cells_including_zero":describe(all_weights),
            "regression_active_cell_value_counts":dict(Counter(str(float(value)) for value in active)),
            "regression_weight64_active_cell_count":int(np.count_nonzero(active==64)),
            "regression_weight64_active_cell_fraction":float(np.mean(active==64)),
            "regression_weights_above1_5_active_cell_count":int(np.count_nonzero(active>1.5)),
            "cls2_soft_target_all_cells":describe(cls2values),
            "mask_shapes":{"cls1":[1,11,11],"cls2":[1,11,11],"regression_weight":[1,11,11],"regression_target":[4,11,11]},
            "model_input_shapes":{"template":[3,127,127],"search":[3,287,287]},
            "positive_empty_support":{"regression":counts["positive_empty_regression"],"cls1":counts["positive_empty_cls1"],"cls2":counts["positive_empty_cls2"]},
            "dataset_representation":{"COCO2017_train2017":len(records),"VID":0,"GOT":0,"YOUTUBEBB":0},
            "eligible_still_images":32,"candidate_represented_images":len(image_counts),"candidate_image_counts":dict(image_counts),
            "eligible_annotation_instances":len(manifest["instances"]),"candidate_represented_instances":len(instance_counts),
            "candidate_annotation_counts":dict(instance_counts),
            "unrepresented_annotation_ids":[row["annotation_id"] for row in manifest["instances"] if str(row["annotation_id"]) not in instance_counts],
            "temporal_positive_interval_frames":{"0":len(accepted)},"positive_motion_px":{"0":len(accepted)},
            "tracking_video_count":0,"no_new_pixels_from_resize_claim":True,"no_model_forward_sgd_gpu":True}
    write_exclusive_json(output,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=audit_distributions(args.run,args.output)
    print(json.dumps({key:result[key] for key in ("accepted_count","regression_weight64_active_cell_count","positive_empty_support")}))
