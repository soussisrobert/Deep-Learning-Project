"""Independent all-candidate/source/array and distribution audit; CPU only."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from hift_research.data_contract.acquisition import write_exclusive_json
from hift_research.data_contract.schema import file_hash,canonical_hash
from hift_research.data_contract.pipeline import array_digest
from hift_research.data_contract.screening import phash64,content_screen
from .pipeline import TemporalPairPipeline
from .policy import load_policy
from .population import intersection,xyxy


def describe(values):
    a=np.asarray(values,dtype=float)
    if not a.size:return {"count":0}
    return dict(count=int(a.size),mean=float(a.mean()),quantiles=dict(zip(
        ("min","p01","p05","p25","p50","p75","p95","p99","max"),map(float,np.quantile(a,[0,.01,.05,.25,.5,.75,.95,.99,1])))))


def dimensions(boxes,xywh=True):
    a=np.asarray(boxes,dtype=float)
    if not len(a):return {"count":0}
    widths=a[:,2] if xywh else a[:,2]-a[:,0];heights=a[:,3] if xywh else a[:,3]-a[:,1]
    return dict(width_px=describe(widths),height_px=describe(heights),area_px2=describe(widths*heights),aspect_w_over_h=describe(widths/heights))


def verify_run(run_path,output):
    run=Path(run_path);summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    frozen=load_policy(summary["policy_path"])
    if file_hash(summary["policy_path"])!=summary["policy_sha256"]:raise ValueError("run policy changed")
    ledger=run/"candidate_ledger.jsonl"
    if file_hash(ledger)!=summary["candidate_ledger_sha256"]:raise ValueError("candidate ledger changed")
    records=[json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    pipeline=TemporalPairPipeline(summary["manifest_path"],summary["policy_path"],summary["split"])
    if [row["candidate_index"] for row in records]!=list(range(pipeline.length)):raise ValueError("candidate stream omission/reorder")
    active_weights=[];all_weights=[];mask_counts=Counter();archive_count=0
    accepted=[];rejected=[]
    for index,record in enumerate(records):
        negative=(index%2==1) if pipeline.split=="development" else bool(np.random.RandomState(pipeline.seed+index).random_sample()<.05)
        if negative!=record["candidate_negative"] or record["replacement_count"] or record["replacement"] is not None:
            raise ValueError("class schedule/replacement differs")
        replay,arrays=pipeline.candidate(index)
        comparable={k:v for k,v in record.items() if k not in ("array_archive_path","array_archive_sha256")}
        if canonical_hash(replay)!=canonical_hash(comparable):raise ValueError("fresh record replay mismatch at "+str(index))
        if record["status"]!="accepted":
            if arrays is not None:raise ValueError("rejected candidate unexpectedly model ready")
            rejected.append(record);continue
        accepted.append(record)
        path=Path(record["array_archive_path"])
        if file_hash(path)!=record["array_archive_sha256"]:raise ValueError("NPZ archive changed")
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays):raise ValueError("archive array key mismatch")
            for key in saved.files:
                actual=saved[key]
                if not np.isfinite(actual).all() or array_digest(actual)!=record["array_hashes"][key]:raise ValueError("stored finite/digest mismatch")
                np.testing.assert_array_equal(actual,arrays[key])
            pipeline._validate(arrays,negative)
            weight=saved["weightxff"];a=saved["label_cls1"];c=saved["labelcls2"]
            if not negative:
                active_weights.extend(weight[weight>0].tolist());all_weights.extend(weight.reshape(-1).tolist())
                mask_counts["positive_empty_regression"]+=not np.any(weight>0)
                mask_counts["positive_empty_cls1"]+=not np.any(a==1)
                mask_counts["positive_empty_cls2"]+=not np.any(c>0)
            for code in (-2,-1,0,1):mask_counts[f"cls1_label_{code}"]+=int(np.count_nonzero(a==code))
            mask_counts["cls2_dense_cells"]+=int(c.size);mask_counts["cls2_positive_cells"]+=int(np.count_nonzero(c>0))
            mask_counts["regression_nonzero_cells"]+=int(np.count_nonzero(weight>0))
        if negative:
            if (record["template_frame_id"]!=record["search_frame_id"] or record["template_identity"]==record["search_identity"]
                    or not record["negative_exclusion_proved"] or record["negative_template_A_overlap_px2"]!=0):raise ValueError("negative presence/footprint witness invalid")
        else:
            if record["template_identity"]!=record["search_identity"] or not 1<=record["native_gap_abs"]<=19:raise ValueError("positive fragment/gap invalid")
            if min(record["template_available_source_support_px2"],record["search_available_source_support_px2"])<=0:raise ValueError("no real source support")
        archive_count+=1
    reservations=json.loads(Path(frozen["reservations_path"]).read_text(encoding="utf-8"))
    content=[]
    for lineage,video in pipeline.videos.items():
        for frame in video["frames"].values():
            payload=Path(frame["path"]).read_bytes()
            import hashlib
            if hashlib.sha256(payload).hexdigest()!=frame["sha256"]:raise ValueError("source bytes changed in independent screen")
            image=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape!=(video["height"],video["width"],3):raise ValueError("independent native decode/dimensions mismatch")
            ph=phash64(image)
            reason,_=content_screen(frame["sha256"],ph,reservations)
            if reason:raise ValueError("independent reserved eval collision")
            content.append(dict(video=lineage,split=video["split"],frame=frame["frame_id"],phash64=ph,sha256=frame["sha256"]))
    cross=[]
    for i,a in enumerate(content):
        for b in content[i+1:]:
            if a["split"]!=b["split"]:
                distance=(int(a["phash64"],16)^int(b["phash64"],16)).bit_count()
                if distance<=4 or a["sha256"]==b["sha256"]:cross.append([a,b,distance])
    if cross:raise ValueError("independent train/development content overlap")
    for source in reservations.get("source_manifests",[]):
        if file_hash(source["path"])!=source["sha256"]:raise ValueError("reserved eval manifest changed")
    positives=[r for r in accepted if not r["candidate_negative"]];negatives=[r for r in accepted if r["candidate_negative"]]
    result={"schema":"hift-temporal-independent-verification-v1","all_checks_passed":True,"split":pipeline.split,
            "all_candidates_replayed":len(records),"all_accepted_archives_verified":archive_count,
            "manifest_sha256":pipeline.manifest_sha256,"policy_sha256":pipeline.policy_sha256,
            "ledger_sha256":file_hash(ledger),"summary_sha256":file_hash(run/"summary.json"),
            "verification_code_sha256":file_hash(__file__),"accepted_positive_count":len(positives),"accepted_negative_count":len(negatives),
            "rejection_reasons":dict(Counter(r["reason"] for r in rejected)),"replacement_count":0,
            "source_frame_count":len(content),"reserved_eval_frames":len(reservations["frames"]),"reserved_eval_lineages":reservations["reserved_family_lineages"],
            "cross_split_content_near_collisions":cross,"global_physical_person_disjointness":"unknown_not_claimed",
            "positive_native_sizes":dimensions([r["search_source_bbox_xywh"] for r in positives]),
            "positive_augmented_label_sizes":dimensions([r["search_augmented_bbox"] for r in positives],False),
            "positive_gap_frames":describe([r["native_gap_abs"] for r in positives]),"positive_gap_seconds":describe([r["gap_seconds"] for r in positives]),
            "positive_motion_px":describe([r["native_center_motion_px"] for r in positives]),
            "positive_motion_normalized":describe([r["native_center_motion_normalized"] for r in positives]),
            "positive_scale_area_ratio":describe([r["source_scale_area_ratio"] for r in positives]),
            "positive_aspect_ratio_change":describe([r["source_aspect_ratio_change"] for r in positives]),
            "positive_missing_intermediate_count":describe([r["missing_intermediate_annotation_count"] for r in positives]),
            "positive_available_source_support_px2":describe([r["search_available_source_support_px2"] for r in positives]),
            "positive_visible_fraction":describe([intersection(r["search_visible_source_bbox_xyxy"],xyxy(r["search_source_bbox_xywh"]))/
                                                 (r["search_source_bbox_xywh"][2]*r["search_source_bbox_xywh"][3]) for r in positives]),
            "positive_label_output_containment_fraction":describe([intersection(r["search_augmented_bbox"],[0,0,286,286])/
                                                                    ((r["search_augmented_bbox"][2]-r["search_augmented_bbox"][0])*(r["search_augmented_bbox"][3]-r["search_augmented_bbox"][1])) for r in positives]),
            "regression_all_active_positive_cell_weights":describe(active_weights),"regression_all_positive_cells_including_zero":describe(all_weights),
            "regression_weight64_cells":int(np.count_nonzero(np.asarray(active_weights)==64)),
            "regression_weight64_pairs":sum(r["weight64_cells"]>0 for r in positives),
            "regression_weight64_active_cell_fraction":float(np.mean(np.asarray(active_weights)==64)) if active_weights else None,
            "mask_totals":dict(mask_counts),"per_positive_pair_support":{key:describe([r[key] for r in positives]) for key in
                        ("regression_support_cells","cls1_positive_cells","cls1_background_cells","cls1_ignored_cells","cls2_positive_cells")},
            "representation":{"videos":dict(Counter(r["video_lineage"] for r in records)),
                              "unique_video_track_fragments":len({tuple(r["template_identity"]) for r in records}|{tuple(r["search_identity"]) for r in records}),
                              "unique_video_frame_endpoints":len({(r["video_lineage"],r["template_frame_id"]) for r in records}|{(r["video_lineage"],r["search_frame_id"]) for r in records})},
            "actual_aug_crop_side511":describe([r["search_geometry"]["augmentation"]["crop_xyxy"][2]-r["search_geometry"]["augmentation"]["crop_xyxy"][0] for r in records if r["search_geometry"] is not None]),
            "authentic_edge_examples":{name:next((dict(candidate_index=r["candidate_index"],video=r["video_lineage"],frames=[r["template_frame_id"],r["search_frame_id"]]) for r in positives if predicate(r)),None)
                for name,predicate in {"stationary":lambda r:r["native_center_motion_px"]==0,
                                       "large_motion_gt100px":lambda r:r["native_center_motion_px"]>100,
                                       "rapid_scale_area_ratio_gt2_or_lt_half":lambda r:r["source_scale_area_ratio"]>2 or r["source_scale_area_ratio"]<.5,
                                       "tiny_native_area_lt64":lambda r:r["search_source_bbox_xywh"][2]*r["search_source_bbox_xywh"][3]<64,
                                       "edge_clipped":lambda r:intersection(r["search_visible_source_bbox_xyxy"],xyxy(r["search_source_bbox_xywh"]))<r["search_source_bbox_xywh"][2]*r["search_source_bbox_xywh"][3],
                                       "missing_intermediate":lambda r:r["missing_intermediate_annotation_count"]>0}.items()},
            "fresh_cache_statistics":pipeline.cache_statistics(),"no_model_forward_sgd_gpu":True,"production_training_authorized":False}
    write_exclusive_json(output,result)
    return result
