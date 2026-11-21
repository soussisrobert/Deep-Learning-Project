"""Freeze authenticated endpoints/pools/policies before candidate outcomes."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
from hift_research.data_contract.acquisition import write_exclusive_json
from hift_research.data_contract.schema import file_hash,canonical_hash
from hift_research.data_contract.screening import phash64,content_screen
from hift_research.data_contract.source_ops import ROOT
from .policy import make_policy
from .population import normalize_source,build_population

SOURCE_FILES=["src/hift_research/temporal_contract/"+name+".py" for name in
              ("population","policy","pipeline","torch_adapter","freeze","preflight")]+[
    "scripts/preflight_temporal_data.py","src/hift_research/data_contract/source_ops.py",
    "src/hift_research/data_contract/pipeline.py","src/hift_research/data_contract/schema.py",
    "src/hift_research/data_contract/torch_adapter.py","src/hift_research/data_contract/screening.py",
    "src/hift_research/data_contract/acquisition.py","src/hift_research/corrections/positive_targets.py",
    "src/hift_research/corrections/pair_targets.py","third_party/HiFT/pysot/datasets/augmentation.py",
    "third_party/HiFT/pysot/datasets/dataset.py","third_party/HiFT/pysot/utils/bbox.py",
    "third_party/HiFT/training_dataset/coco/par_crop.py","third_party/HiFT/experiments/config.yaml"]


def freeze_contract(manifest_path,output,reservations_path,prereg_path=None,selection_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest_path=Path(manifest_path).resolve();reservations_path=Path(reservations_path).resolve()
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    videos,quarantine=normalize_source(manifest)
    if prereg_path is not None:
        prereg=json.loads(Path(prereg_path).read_text(encoding="utf-8"))
        if selection_path is None or file_hash(selection_path)!=prereg["source_selection_sha256"]:raise ValueError("source selection changed after eligibility prereg")
        for split in ("train","development"):
            if canonical_hash(prereg[split+"_policy"])!=canonical_hash(make_policy(split)):raise ValueError("eligibility policy changed after prereg")
    reservations=json.loads(reservations_path.read_text(encoding="utf-8"))
    frame_hashes={};content=[];fatal=[]
    for lineage,video in sorted(videos.items()):
        for index,frame in sorted(video["frames"].items()):
            image=cv2.imread(frame["path"],cv2.IMREAD_COLOR)
            if image is None or image.shape!=(video["height"],video["width"],3):raise ValueError("source dimension/decode mismatch")
            ph=phash64(image)
            if frame.get("phash64") is not None and ph!=frame["phash64"]:raise ValueError("declared native pHash differs")
            reason,match=content_screen(frame["sha256"],ph,reservations)
            status={"status":"clear" if reason is None else "quarantined","reason":reason,"match":match}
            row={"video_lineage":lineage,"split":video["split"],"frame_id":index,"sha256":frame["sha256"],"phash64":ph,"reserved_eval_screen":status}
            content.append(row)
            if status["status"]!="clear":fatal.append(dict(row,reason="reserved_eval_content_overlap"))
            old=frame_hashes.get(frame["sha256"])
            if old is not None and old[0]!=lineage:fatal.append(dict(row,reason="duplicate_content_across_video_lineages"))
            frame_hashes[frame["sha256"]]=(lineage,index)
    near_cross_split=[];near_within_video=0
    for i,a in enumerate(content):
        for b in content[i+1:]:
            distance=(int(a["phash64"],16)^int(b["phash64"],16)).bit_count()
            if distance<=4:
                pair={"a":[a["video_lineage"],a["frame_id"]],"b":[b["video_lineage"],b["frame_id"]],"phash_distance":distance}
                if a["split"]!=b["split"]:near_cross_split.append(pair)
                elif a["video_lineage"]==b["video_lineage"]:near_within_video+=1
    write_exclusive_json(output/"content_screen.json",dict(frames=content,quarantine=fatal,cross_split_near_pairs=near_cross_split,
                                                         within_video_near_pair_count=near_within_video,
                                                         within_video_near_policy="record_temporal_similarity_not_global_alias_proof"))
    if fatal or near_cross_split:raise ValueError("source/eval or train/development content quarantine; no training contract frozen")
    code_hashes={str((ROOT/name).resolve()):file_hash(ROOT/name) for name in SOURCE_FILES}
    policies={}
    for split in ("train","development"):
        policy=make_policy(split);population=build_population(videos,policy,split)
        population["identity_quarantine"]=quarantine
        # Pipeline recomputation also includes this source-level quarantine record.
        if not population["positives"] or not population["negatives"]:
            write_exclusive_json(output/(split+"_population_unready.json"),population)
            raise ValueError(split+" lacks positive or guarded known-distinct population; no favorable refill")
        population_path=(output/(split+"_population.json")).resolve()
        write_exclusive_json(population_path,population)
        path=(output/(split+"_policy.json")).resolve()
        frozen={"schema":"hift-temporal-policy-v1","split":split,"policy":policy,"policy_sha256":canonical_hash(policy),
                "manifest_path":str(manifest_path),"manifest_sha256":file_hash(manifest_path),
                "population_path":str(population_path),"population_sha256":file_hash(population_path),
                "reservations_path":str(reservations_path),"reservations_sha256":file_hash(reservations_path),
                "source_code_hashes":code_hashes,"source_content_screen_path":str((output/"content_screen.json").resolve()),
                "source_content_screen_sha256":file_hash(output/"content_screen.json"),"production_training_authorized":False,
                "cache_limits_payload_bytes":{"decoded":256*1024*1024,"source_crop511":256*1024*1024}}
        if prereg_path is not None:frozen.update(eligibility_prereg_path=str(Path(prereg_path).resolve()),eligibility_prereg_sha256=file_hash(prereg_path))
        write_exclusive_json(path,frozen);policies[split]=str(path)
    write_exclusive_json(output/"freeze_receipt.json",dict(manifest_sha256=file_hash(manifest_path),policies=policies,
                                                        source_code_hashes=code_hashes,quarantine=quarantine))
    return policies
