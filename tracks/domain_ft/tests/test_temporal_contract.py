"""Synthetic source-content/GT fixtures, NOT the authentic MOT cohort."""
from __future__ import annotations
import copy
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
import pytest
from hift_research.data_contract.schema import canonical_hash,file_hash
from hift_research.data_contract.screening import phash64
from hift_research.temporal_contract.population import normalize_source,build_population,source_footprint,intersection,xyxy,parse_gt
from hift_research.temporal_contract.policy import make_policy,load_policy
from hift_research.temporal_contract.freeze import freeze_contract
from hift_research.temporal_contract.pipeline import TemporalPairPipeline,PixelLRU


def source_fixture(root,mode="ordinary"):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);videos=[]
    for vi,(name,split) in enumerate((("MOT17-02","train"),("MOT17-05","development"))):
        directory=root/name;directory.mkdir();gtrows=[];frames=[]
        for frame in (1,2,3,4):
            x=20.+frame*5;y=20.;w=20.;h=40.
            if mode=="stationary":x=20.
            if mode=="motion":x=20.+(frame-1)*80
            if mode=="scale":w*=frame;h*=frame
            if mode=="tiny":w=.25;h=.5
            if mode=="clipped":x=-8.
            if mode=="offimage":x=600.
            annotations=[]
            for track,box in ((1,[x+1,y+1,w,h]),(2,[401.,151.,20.,40.]),(3,[311.,151.,20.,40.])):
                if frame==3 and track==1:continue
                mark,cls,vis=1,1,1.
                row=[frame,track,*box,mark,cls,vis];gtrows.append(row)
                annotations.append(dict(track_id=str(track),bbox_xywh_raw=box,mark=mark,class_id=cls,visibility=vis,presence=True))
            if frame==3:continue
            image=np.random.RandomState(10+vi*100+frame).randint(0,256,(256,512,3),dtype=np.uint8)
            for ann in annotations:
                x1,y1,bw,bh=ann["bbox_xywh_raw"]
                cv2.rectangle(image,(int(x1),int(y1)),(int(x1+bw),int(y1+bh)),(40,150,240),1)
            path=directory/f"{frame:06d}.jpg";assert cv2.imwrite(str(path),image)
            decoded=cv2.imread(str(path));frames.append(dict(frame_id=frame,path=str(path.resolve()),sha256=file_hash(path),phash64=phash64(decoded),annotations=annotations))
        gt=directory/"gt.txt";gt.write_text("\n".join(",".join(map(str,row)) for row in gtrows)+"\n",encoding="utf-8")
        videos.append(dict(video_lineage=name,split=split,author_sequence_id=name+"-DPM",seqinfo=dict(w=512,h=256,fps=30,totalframes=4),
                           original_GT=dict(path=str(gt.resolve()),sha256=file_hash(gt)),frames=frames))
    value=dict(schema="hift-temporal-source-v1",coordinate_origin=1,identity_scope="source_trajectory_fragment_not_global_person",videos=videos)
    path=root/"source.json";path.write_text(json.dumps(value),encoding="utf-8")
    reservations=root/"reservations.json";reservations.write_text(json.dumps(dict(frames=[],reserved_family_lineages=[],source_manifests=[])),encoding="utf-8")
    return path,reservations,value


@pytest.fixture
def frozen(tmp_path):
    manifest,reservations,value=source_fixture(tmp_path/"data")
    policies=freeze_contract(manifest,tmp_path/"freeze",reservations)
    return manifest,policies,value


def test_frozen_populations_are_nonzero_gap_and_simultaneous_safe(frozen):
    manifest,policies,value=frozen
    pipeline=TemporalPairPipeline(manifest,policies["train"])
    assert all(1<=abs(p["template_frame"]-p["search_frame"])<=19 for p in pipeline.population["positives"])
    assert all(p["template_track"]!=p["search_track"] for p in pipeline.population["negatives"])
    assert pipeline.population["negatives"]


def test_replay_global_rng_and_masks_for_both_classes(frozen):
    manifest,policies,_=frozen
    pipeline=TemporalPairPipeline(manifest,policies["train"])
    np.random.seed(72);before=np.random.get_state()
    rows=[pipeline.candidate(i) for i in range(128)]
    after=np.random.get_state()
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    assert any(row[0]["candidate_negative"] for row in rows)
    assert all(row[0]["replacement_count"]==0 for row in rows)
    for record,arrays in rows:assert json.loads(json.dumps(record,allow_nan=False))==record
    replay=TemporalPairPipeline(manifest,policies["train"])
    for index in (0,7,65,127):assert canonical_hash(replay.candidate(index)[0])==canonical_hash(rows[index][0])
    for record,arrays in rows:
        if arrays is None:continue
        assert arrays["template"].shape==(3,127,127) and arrays["search"].shape==(3,287,287)
        if record["candidate_negative"]:
            assert record["negative_exclusion_proved"]
            assert not any(np.any(arrays[key]) for key in ("label_cls1","labelcls2","weightxff","labelxff"))
        else:
            assert record["native_gap_abs"]>0 and record["template_identity"]==record["search_identity"]
            assert record["search_available_source_support_px2"]>0


def test_augmentation_can_invalidate_nominal_negative_no_hidden_refill(frozen):
    manifest,policies,_=frozen
    pipeline=TemporalPairPipeline(manifest,policies["train"])
    negatives=[pipeline.candidate(i)[0] for i in range(300) if np.random.RandomState(20261023+i).random_sample()<.05]
    assert any(row["status"]=="rejected" and row["reason"]=="template_target_visible_in_negative_search" for row in negatives)
    for row in negatives:
        assert row["template_geometry"] is not None and row["search_geometry"] is not None
        assert row["replacement_count"]==0


def test_balanced_dev_is_fixed_unaugmented_control_not_bernoulli(frozen):
    manifest,policies,_=frozen
    pipeline=TemporalPairPipeline(manifest,policies["development"],split="dev")
    records=[pipeline.candidate(i)[0] for i in range(128)]
    assert sum(row["candidate_negative"] for row in records)==64
    assert all(row["candidate_negative"]==(i%2==1) for i,row in enumerate(records))
    assert all(row["status"]=="accepted" for row in records)
    assert pipeline.policy["template_aug"]==dict(shift=0.,scale=0.,blur=0.,flip=0.,color=0.)


@pytest.mark.parametrize("mode",["stationary","motion","scale","tiny","clipped"])
def test_synthetic_content_cases_preserve_native_evidence(tmp_path,mode):
    manifest,reservations,value=source_fixture(tmp_path/"source",mode)
    policies=freeze_contract(manifest,tmp_path/"freeze",reservations)
    pipeline=TemporalPairPipeline(manifest,policies["train"])
    records=[pipeline.candidate(i)[0] for i in range(60)]
    selected=next(row for row in records if not row["candidate_negative"] and row["template_identity"][-1]=="1" and row["native_gap_abs"]==3)
    assert selected["status"]=="accepted"
    assert selected["missing_intermediate_annotation_frames"]==[3]
    if mode=="stationary":assert selected["native_center_motion_px"]==0
    if mode=="motion":assert selected["native_center_motion_px"]==240
    if mode=="scale":assert selected["source_scale_area_ratio"] in (16.,1/16.)
    if mode=="tiny":assert selected["search_available_source_support_px2"]<=.125
    if mode=="clipped":assert selected["search_source_bbox_xywh"][0]<0 and selected["search_available_source_support_px2"]>0


def test_fully_offimage_endpoints_excluded_without_relabel(tmp_path):
    _,_,value=source_fixture(tmp_path/"source","offimage")
    videos,_=normalize_source(value)
    population=build_population(videos,make_policy(),"train")
    assert population["exclusion_counts"]["fully_off_image"]==3
    assert all(row["track_id"]!="1" for row in population["positives"])


@pytest.mark.parametrize("mutation",["box","presence","nonintegerID","badmark","submission10cols"])
def test_source_gt_native_format_and_manifest_contradictions_fail(tmp_path,mutation):
    _,_,value=source_fixture(tmp_path/"source")
    if mutation=="box":value["videos"][0]["frames"][0]["annotations"][0]["bbox_xywh_raw"][0]+=1
    elif mutation=="presence":value["videos"][0]["frames"][0]["annotations"][0]["presence"]=False
    else:
        gt=Path(value["videos"][0]["original_GT"]["path"]);text=gt.read_text(encoding="utf-8");lines=text.splitlines();row=lines[0].split(",")
        if mutation=="nonintegerID":row[1]="1.5"
        elif mutation=="badmark":row[6]="2"
        else:row.append("-1")
        lines[0]=",".join(row);gt.write_text("\n".join(lines)+"\n",encoding="utf-8");value["videos"][0]["original_GT"]["sha256"]=file_hash(gt)
    with pytest.raises(ValueError):normalize_source(value)


def test_id_reuse_and_duplicate_track_rows_quarantine_whole_fragment(tmp_path):
    _,_,value=source_fixture(tmp_path/"source")
    value["videos"][0]["suspect_reused_track_ids"]=["1"]
    value["videos"][0]["frames"][0]["annotations"].append(copy.deepcopy(value["videos"][0]["frames"][0]["annotations"][1]))
    videos,quarantine=normalize_source(value)
    assert videos["MOT17-02"]["quarantined_tracks"]>={"1","2"}
    assert quarantine


def test_cross_video_content_alias_is_fail_closed_before_population(tmp_path):
    manifest,reservations,value=source_fixture(tmp_path/"source")
    value["videos"][1]["frames"][0].update({key:value["videos"][0]["frames"][0][key] for key in ("path","sha256","phash64")})
    manifest.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(ValueError,match="quarantine"):freeze_contract(manifest,tmp_path/"freeze",reservations)


def test_observer_cache_eviction_and_warmth_do_not_change_canonical_arrays(frozen):
    manifest,policies,_=frozen
    class Observer:
        def __init__(self):self.rows=[]
        def record(self,name,elapsed,cache_hit=False):self.rows.append((name,elapsed,cache_hit))
    observer=Observer();cold=TemporalPairPipeline(manifest,policies["train"])
    timed=TemporalPairPipeline(manifest,policies["train"],stage_observer=observer)
    expected=cold.candidate(0)[0]
    assert timed.candidate(0)[0]==timed.candidate(0)[0]==expected
    timed.crop_cache.limit=1;timed.decoded.limit=1
    assert timed.candidate(7)[0]==cold.candidate(7)[0]
    assert any(hit and elapsed==0 for name,elapsed,hit in observer.rows)
    cache=PixelLRU(10);cache.put("a",np.zeros(8,dtype=np.uint8),8);cache.put("b",np.zeros(8,dtype=np.uint8),8)
    assert cache.bytes<=10 and cache.evictions==1 and cache.get("a") is None


def test_dataset_pickle_and_reject_collation_propagates_identity(frozen):
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset,reject_aware_collate
    manifest,policies,_=frozen
    dataset=TemporalCandidateDataset(manifest,policies["train"])
    first=dataset[0]
    restored=pickle.loads(pickle.dumps(dataset));assert restored._pipeline is None
    assert restored.candidate(0)[0]==first[0]
    assert len(dataset)==1024
    batch=reject_aware_collate([first]);assert batch["accepted_records"][0]["template_identity"]==first[0]["template_identity"]
    with pytest.raises(IndexError):dataset[1024]


def test_same_sized_changed_media_is_input_fault_not_candidate_rejection(frozen):
    manifest,policies,value=frozen
    pipeline=TemporalPairPipeline(manifest,policies["train"])
    for frame in value["videos"][0]["frames"]:
        changed=np.zeros((256,512,3),dtype=np.uint8)
        ok,payload=cv2.imencode(".jpg",changed);assert ok
        Path(frame["path"]).write_bytes(payload.tobytes())
    with pytest.raises(OSError,match="SHA256 changed"):pipeline.candidate(0)


def test_corrupted_reservations_and_content_receipts_fail_closed(frozen,tmp_path):
    manifest,policies,_=frozen
    original=json.loads(Path(policies["train"]).read_text(encoding="utf-8"))
    for key in ("reservations","source_content_screen"):
        changed=copy.deepcopy(original);bad=tmp_path/(key+".json");bad.write_text("{}",encoding="utf-8")
        changed[key+"_path"]=str(bad);path=tmp_path/(key+"_policy.json");path.write_text(json.dumps(changed),encoding="utf-8")
        with pytest.raises(ValueError,match="frozen "+key):load_policy(path)


def test_frozen_policy_code_and_source_hashes_fail_closed(frozen,tmp_path):
    manifest,policies,_=frozen
    value=json.loads(Path(policies["train"]).read_text(encoding="utf-8"))
    value["policy"]["positive_max_gap"]=0
    path=tmp_path/"tampered.json";path.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(ValueError):load_policy(path)
    with pytest.raises(ValueError):TemporalPairPipeline(manifest,policies["train"],seed=1)
    with pytest.raises(ValueError):TemporalPairPipeline(manifest,policies["train"],split="development")
