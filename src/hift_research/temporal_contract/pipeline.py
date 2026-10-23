"""Indexed authentic temporal pairs; CPU arrays, no model/SGD/GPU.

Scoped NumPy RNG is serial per-process, not thread-safe. Timing observers are
external only and must not mutate RNG/data. Cache bounds affect speed, not pairs.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import cv2
import numpy as np
from hift_research.data_contract.pipeline import numpy_seed,array_digest
from hift_research.data_contract.schema import canonical_hash,file_hash
from hift_research.data_contract.source_ops import source_operations,jpeg_crop511,preprocess_geometry
from hift_research.corrections.positive_targets import CorrectedAnchorTarget,robust_pair_targets
from .policy import load_policy
from .population import normalize_source,build_population,xyxy,intersection,source_footprint


class PixelLRU:
    """Bound cached ndarray payload; Python allocator/RSS is not this budget."""
    def __init__(self,limit=256*1024*1024):
        self.limit=limit;self.bytes=0;self.hits=0;self.misses=0;self.evictions=0;self.rows=OrderedDict()
    def get(self,key):
        if key not in self.rows:self.misses+=1;return None
        self.hits+=1;value,size=self.rows.pop(key);self.rows[key]=(value,size);return value
    def put(self,key,value,size):
        if key in self.rows:self.bytes-=self.rows.pop(key)[1]
        if size>self.limit:return
        while self.rows and self.bytes+size>self.limit:
            _,(_,old)=self.rows.popitem(last=False);self.bytes-=old;self.evictions+=1
        self.rows[key]=(value,size);self.bytes+=size
    def statistics(self):
        return dict(limit=self.limit,bytes=self.bytes,entries=len(self.rows),hits=self.hits,misses=self.misses,evictions=self.evictions)


def affine_geometry(source_box,augmentation):
    prep=preprocess_geometry(source_box)
    first=np.eye(3);first[:2]=prep["affine_2x3"]
    second=np.eye(3);second[:2]=augmentation["affine_2x3"]
    composed=second@first
    size=augmentation["output_size"]
    inverse=np.linalg.inv(composed)
    corners=np.array([[0,0,1],[size-1,0,1],[size-1,size-1,1],[0,size-1,1]],dtype=float)
    original=(inverse@corners.T).T[:,:2]
    footprint=[float(original[:,0].min()),float(original[:,1].min()),float(original[:,0].max()),float(original[:,1].max())]
    return dict(preprocessing=prep,augmentation=augmentation,source_to_output_affine_2x3=composed[:2].tolist(),
                output_to_source_affine_2x3=inverse[:2].tolist(),source_footprint_xyxy=footprint)


def clipped_box(box,width,height):
    return [max(0.,box[0]),max(0.,box[1]),min(float(width),box[2]),min(float(height),box[3])]


class TemporalPairPipeline:
    def __init__(self,manifest_path,policy_path,split="train",seed=None,length=None,stage_observer=None):
        self.manifest_path=str(Path(manifest_path).resolve());self.policy_path=str(Path(policy_path).resolve())
        frozen=load_policy(self.policy_path);self.policy=frozen["policy"]
        self.manifest_sha256=file_hash(self.manifest_path);self.policy_sha256=file_hash(self.policy_path)
        self.reservations_sha256=frozen["reservations_sha256"]
        if self.manifest_sha256!=frozen["manifest_sha256"]:raise ValueError("manifest differs from frozen policy")
        split="development" if split=="dev" else split
        if split!=self.policy["split"]:raise ValueError("policy/split mismatch")
        self.split=split;self.seed=self.policy["seed"] if seed is None else seed
        self.length=self.policy["preflight_length"] if length is None else length
        if self.seed!=self.policy["seed"] or self.length not in self.policy["allowed_lengths"]:
            raise ValueError("seed/length not preregistered")
        self.manifest=json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
        self.videos,self.quarantine=normalize_source(self.manifest)
        self.population=json.loads(Path(frozen["population_path"]).read_text(encoding="utf-8"))
        expected_population=build_population(self.videos,self.policy,split)
        expected_population["identity_quarantine"]=self.quarantine
        if canonical_hash(self.population)!=canonical_hash(expected_population):
            raise ValueError("frozen eligible pair population inconsistent with source")
        if not self.population["positives"] or not self.population["negatives"]:raise ValueError("positive and known-safe negative pools both required")
        self.ops=source_operations()
        self.anchor=CorrectedAnchorTarget(SimpleNamespace(TRAIN=SimpleNamespace(SEARCH_SIZE=287,OUTPUTFEATURE_SIZE=11,LABEL_RANGE=4)))
        def augmentation(key):
            p=self.policy[key];return self.ops.AuditedAugmentation(p["shift"],p["scale"],p["blur"],p["flip"],p["color"])
        self.template_aug=augmentation("template_aug");self.search_aug=augmentation("search_aug")
        self.decoded=PixelLRU();self.crop_cache=PixelLRU();self.stage_observer=stage_observer

    @contextmanager
    def _stage(self,name):
        if self.stage_observer is None:yield;return
        start=perf_counter()
        try:yield
        finally:self.stage_observer.record(name,perf_counter()-start,cache_hit=False)

    def _hit(self,name):
        if self.stage_observer is not None:self.stage_observer.record(name,0.,cache_hit=True)

    def cache_statistics(self):
        return {"decoded":self.decoded.statistics(),"source_crop511":self.crop_cache.statistics()}

    def _prepared(self,lineage,frame,ann,view):
        key=(lineage,frame["frame_id"],ann["track_id"])
        cached=self.crop_cache.get(key)
        if cached is not None:
            self._hit("decode/"+view);self._hit("source_crop_jpeg/"+view);return cached
        image_key=(lineage,frame["frame_id"])
        image=self.decoded.get(image_key)
        if image is None:
            with self._stage("decode/"+view):
                payload=Path(frame["path"]).read_bytes()
                if hashlib.sha256(payload).hexdigest()!=frame["sha256"]:raise OSError("source input SHA256 changed: "+frame["path"])
                try:image=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
                except cv2.error as exc:raise OSError("source image decode failed: "+frame["path"]) from exc
                if image is None or image.shape!=(frame["height"],frame["width"],3):raise OSError("source image decode/dimension fault: "+frame["path"])
                image.setflags(write=False)
                self.decoded.put(image_key,image,image.nbytes)
        else:self._hit("decode/"+view)
        with self._stage("source_crop_jpeg/"+view):
            crop,jpeg=jpeg_crop511(image,ann["bbox_xywh"],self.ops)
            crop.setflags(write=False)
            value=crop,hashlib.sha256(jpeg).hexdigest()
            self.crop_cache.put(key,value,crop.nbytes)
        return value

    def _ann(self,video,frame_id,track):
        frame=video["frames"].get(frame_id)
        if frame is None:raise ValueError("missing selected source endpoint")
        matches=[ann for ann in frame["annotations"] if ann["track_id"]==track]
        if len(matches)!=1:raise ValueError("missing/duplicate endpoint annotation")
        return frame,matches[0]

    def candidate(self,index):
        if isinstance(index,bool) or not isinstance(index,Integral) or not 0<=index<self.length:
            raise IndexError("candidate index outside declared stream: "+str(index))
        index=int(index)
        record={"schema":"hift-temporal-candidate-v1","candidate_index":index,"seed":self.seed+index,"split":self.split,
                "manifest_sha256":self.manifest_sha256,"policy_sha256":self.policy_sha256,"reservations_sha256":self.reservations_sha256,
                "replacement":None,"replacement_count":0,"identity_scope":"source_trajectory_fragment_not_global_person",
                "dataset":"MOT17/train","template_augmented_bbox":None,"search_augmented_bbox":None,
                "template_geometry":None,"search_geometry":None,"targets_state":"not_constructed"}
        with numpy_seed(self.seed+index):
            negative=(index%2==1) if self.split=="development" else bool(np.random.random()<.05)
            pool=self.population["negatives" if negative else "positives"]
            selection=pool[int(np.random.randint(len(pool)))];lineage=selection["video_lineage"];video=self.videos[lineage]
            tf=selection["frame_id"] if negative else selection["template_frame"]
            sf=selection["frame_id"] if negative else selection["search_frame"]
            tt=selection["template_track"] if negative else selection["track_id"]
            st=selection["search_track"] if negative else selection["track_id"]
            template_frame,template=self._ann(video,tf,tt);search_frame,search=self._ann(video,sf,st)
            tid=["MOT17",lineage,tt];sid=["MOT17",lineage,st]
            tx,ty,tw,th=template["bbox_xywh"];sx,sy,sw,sh=search["bbox_xywh"]
            distance=float(np.hypot(sx+sw/2-tx-tw/2,sy+sh/2-ty-th/2))
            native_between=list(range(min(tf,sf)+1,max(tf,sf)))
            missing=[f for f in native_between if (f,tt) not in video["gt"]]
            record.update(candidate_negative=negative,pair_type="simultaneous_known_distinct_negative" if negative else "nonzero_gap_same_fragment_positive",
                          selection=selection,video_lineage=lineage,template_frame_id=tf,search_frame_id=sf,
                          template_identity=tid,search_identity=sid,template_identity_key="/".join(tid),search_identity_key="/".join(sid),
                          template_source_bbox_raw=template["bbox_xywh_raw"],search_source_bbox_raw=search["bbox_xywh_raw"],
                          template_source_bbox_xywh=template["bbox_xywh"],search_source_bbox_xywh=search["bbox_xywh"],
                          template_presence=template["presence"],search_presence=search["presence"],
                          template_visibility=template["visibility"],search_visibility=search["visibility"],
                          template_source_gt_row=template["source_row"],search_source_gt_row=search["source_row"],
                          template_source_sha256=template_frame["sha256"],search_source_sha256=search_frame["sha256"],
                          native_gap_signed=sf-tf,native_gap_abs=abs(sf-tf),gap_seconds=abs(sf-tf)/video["fps"],
                          missing_intermediate_annotation_frames=missing,missing_intermediate_annotation_count=len(missing),
                          unselected_intermediate_frames=[f for f in native_between if f not in video["frames"]],
                          native_center_motion_px=distance,native_center_motion_normalized=distance/max(np.sqrt(tw*th),1e-12),
                          source_scale_area_ratio=sw*sh/(tw*th),source_aspect_ratio_change=(sw/sh)/(tw/th),
                          source_width=video["width"],source_height=video["height"],source_frame_rate=video["fps"])
            try:
                template_crop,template_sha=self._prepared(lineage,template_frame,template,"template")
                search_crop,search_sha=self._prepared(lineage,search_frame,search,"search")
                def centered(ann,crop):
                    x,y,w,h=ann["bbox_xywh"];return self.ops.get_bbox(None,crop,[x,y,x+w,y+h])
                tbox=centered(template,template_crop);sbox=centered(search,search_crop)
                with self._stage("augmentation/template"):t,ta=self.template_aug(template_crop,tbox,127,gray=False)
                with self._stage("augmentation/search"):s,sa=self.search_aug(search_crop,sbox,287,gray=False)
                tg=affine_geometry(template["bbox_xywh"],self.template_aug.last_crop_geometry)
                sg=affine_geometry(search["bbox_xywh"],self.search_aug.last_crop_geometry)
                visible_t=clipped_box(xyxy(template["bbox_xywh"]),video["width"],video["height"])
                visible_s=clipped_box(xyxy(search["bbox_xywh"]),video["width"],video["height"])
                support_t=intersection(visible_t,tg["source_footprint_xyxy"])
                support_s=intersection(visible_s,sg["source_footprint_xyxy"])
                guarded=source_footprint(search["bbox_xywh"],self.search_aug.last_crop_geometry["crop_xyxy"],
                                         self.policy["jpeg_support_margin_crop511_px"],self.policy["negative_source_margin_px"])
                negative_overlap=float(intersection(visible_t,guarded)) if negative else None
                record.update(template_geometry=tg,search_geometry=sg,template_augmented_bbox=list(map(float,ta)),
                              search_augmented_bbox=list(map(float,sa)),template_centered_bbox=list(map(float,tbox)),search_centered_bbox=list(map(float,sbox)),
                              template_crop511_jpeg_sha256=template_sha,search_crop511_jpeg_sha256=search_sha,
                              template_available_source_support_px2=support_t,search_available_source_support_px2=support_s,
                              template_visible_source_bbox_xyxy=visible_t,search_visible_source_bbox_xyxy=visible_s,
                              negative_guarded_source_footprint_xyxy=guarded if negative else None,
                              negative_template_A_overlap_px2=negative_overlap,negative_exclusion_proved=bool(negative_overlap==0) if negative else None,
                              input_pixel_hashes={"template":array_digest(t),"search":array_digest(s)})
                if support_t<=0 or support_s<=0:raise ValueError("no_available_target_source_support")
                if negative and negative_overlap>0:raise ValueError("template_target_visible_in_negative_search")
                with self._stage("targets"):
                    labels=robust_pair_targets(self.anchor,sa,11,is_negative=negative,template_identity=tuple(tid),search_identity=tuple(sid))
                arrays=dict(zip(("label_cls1","labelxff","labelcls2","weightxff"),labels))
                arrays.update(template=t.transpose(2,0,1).astype(np.float32),search=s.transpose(2,0,1).astype(np.float32),bbox=np.asarray(sa,dtype=np.float64))
                with self._stage("validation"):
                    self._validate(arrays,negative)
                cls1,offsets,cls2,weight=labels
                index_box=np.int32(np.clip((np.asarray(sa)-63)/16,0,10))
                collapsed=bool(index_box[0]==index_box[2] and index_box[1]==index_box[3])
                record.update(status="accepted",reason=None,targets_state="constructed_valid",
                              array_hashes={key:array_digest(value) for key,value in arrays.items()},
                              input_contract={key:{"shape":list(value.shape),"dtype":str(value.dtype)} for key,value in arrays.items()},
                              regression_support_cells=int(np.count_nonzero(weight)),weight64_cells=int(np.count_nonzero(weight==64)),
                              cls1_positive_cells=int(np.count_nonzero(cls1==1)),cls1_background_cells=int(np.count_nonzero(cls1==0)),
                              cls1_ignored_cells=int(np.count_nonzero(cls1<0)),cls2_positive_cells=int(np.count_nonzero(cls2>0)),
                              cls2_dense_loss_cells=int(cls2.size),positive_collapsed_fallback=collapsed and not negative)
                return record,arrays
            except (ValueError,cv2.error) as exc:
                record.update(status="rejected",reason=str(exc),targets_state="not_model_ready")
                return record,None

    @staticmethod
    def _validate(arrays,negative):
        shapes={"template":(3,127,127),"search":(3,287,287),"bbox":(4,),"label_cls1":(1,11,11),
                "labelxff":(4,11,11),"labelcls2":(1,11,11),"weightxff":(1,11,11)}
        dtypes={"template":np.float32,"search":np.float32,"bbox":np.float64,"label_cls1":np.float64,
                "labelxff":np.float32,"labelcls2":np.float64,"weightxff":np.float64}
        if any(arrays[key].shape!=shape or arrays[key].dtype!=dtypes[key] or not np.isfinite(arrays[key]).all() for key,shape in shapes.items()):
            raise ValueError("invalid_input_target_shape_dtype_or_finite")
        a,c,w=arrays["label_cls1"],arrays["labelcls2"],arrays["weightxff"]
        if not set(np.unique(a)).issubset({-2,-1,0,1}) or np.any((c<0)|(c>1)) or np.any(w<0):raise ValueError("invalid_mask_domains")
        if negative and (np.any(a) or np.any(c) or np.any(w) or np.any(arrays["labelxff"])):raise ValueError("negative_dense_zero_mask_violation")
        if not negative and (not np.any(a==1) or not np.any(c>0) or not np.any(w>0)):raise ValueError("empty_positive_support")
