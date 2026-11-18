"""Extra repeated-pair, crop/augmentation and mask descriptives; no new sampling."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from hift_research.data_contract.acquisition import write_exclusive_json
from hift_research.data_contract.schema import canonical_hash,file_hash
from .verify import describe
from .population import intersection


def audit(root,output):
    root=Path(root);results={}
    for split in ("train","development"):
        folder=root/split;summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"));ledger=folder/"candidate_ledger.jsonl"
        if file_hash(ledger)!=summary["candidate_ledger_sha256"]:raise ValueError("ledger changed")
        records=[json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        accepted=[r for r in records if r["status"]=="accepted"];positive=[r for r in accepted if not r["candidate_negative"]]
        classes={}
        for name,predicate in (("positive",lambda r:not r["candidate_negative"]),("negative",lambda r:r["candidate_negative"])):
            rows=[r for r in accepted if predicate(r)];counts=Counter(canonical_hash(r["selection"]) for r in rows)
            classes[name]=dict(accepted=len(rows),unique_source_pairs=len(counts),repeated_pair_attempts=len(rows)-len(counts),
                               source_pair_max_multiplicity=max(counts.values()) if counts else 0,
                               exact_input_array_hash_sets=len({canonical_hash(r["array_hashes"]) for r in rows}))
        augmentation={}
        for view,size in (("template",127),("search",287)):
            geometries=[r[view+"_geometry"] for r in records if r[view+"_geometry"] is not None]
            rois=[g["augmentation"]["crop_xyxy"] for g in geometries]
            augmentation[view]=dict(realized_shift_x_crop511_px=describe([(r[0]+r[2])/2-255 for r in rois]),
                                    realized_shift_y_crop511_px=describe([(r[1]+r[3])/2-255 for r in rois]),
                                    realized_scale_x=describe([(r[2]-r[0])/(size-1) for r in rois]),
                                    realized_scale_y=describe([(r[3]-r[1])/(size-1) for r in rois]))
        containment={}
        for view,size in (("template",127),("search",287)):
            values=[];native=[]
            for r in positive:
                box=r[view+"_augmented_bbox"];area=(box[2]-box[0])*(box[3]-box[1])
                values.append(intersection(box,[0,0,size-1,size-1])/area)
                source=r[view+"_source_bbox_xywh"]
                native.append(r[view+"_available_source_support_px2"]/(source[2]*source[3]))
            containment[view]=dict(label_fraction_in_output=describe(values),available_original_bbox_support_fraction=describe(native))
        results[split]=dict(ledger_sha256=file_hash(ledger),repeated_source_pairs=classes,
                           cls1_positive_active_per_pair=describe([r["cls1_positive_cells"]+r["cls1_background_cells"] for r in positive]),
                           cls1_positive_active_total=sum(r["cls1_positive_cells"]+r["cls1_background_cells"] for r in positive),
                           cls1_positive_background_total=sum(r["cls1_background_cells"] for r in positive),
                           cls1_positive_ignore_total=sum(r["cls1_ignored_cells"] for r in positive),
                           realized_augmentation_geometry=augmentation,crop_containment=containment,
                           negative_rejected_geometry=[{k:r[k] for k in ("candidate_index","video_lineage","selection","template_identity","search_identity",
                                                                       "template_source_bbox_xywh","search_source_bbox_xywh","negative_guarded_source_footprint_xyxy",
                                                                       "negative_template_A_overlap_px2","reason")} for r in records if r["status"]!="accepted"],
                           template_track_counts=dict(Counter(r["template_identity_key"] for r in records)),
                           template_frame_counts=dict(Counter(r["video_lineage"]+"/"+str(r["template_frame_id"]) for r in records)),
                           frozen_policy=json.loads(Path(summary["policy_path"]).read_text(encoding="utf-8"))["policy"])
    write_exclusive_json(output,dict(schema="hift-temporal-posthoc-distributions-v1",no_new_candidates=True,no_model_forward_sgd_gpu=True,splits=results))
    return {split:value["repeated_source_pairs"] for split,value in results.items()}


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    print(json.dumps(audit(a.root,a.output),sort_keys=True))
