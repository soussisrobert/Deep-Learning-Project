"""Native observed endpoints and conservative known-distinct pilot populations."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
from pathlib import Path
import numpy as np
from hift_research.data_contract.schema import file_hash
from hift_research.data_contract.source_ops import preprocess_geometry


def intersection(a,b):
    return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))


def xyxy(box):
    x,y,w,h=box
    return [x,y,x+w,y+h]


def source_footprint(box, crop_roi=(112.,112.,398.,398.), jpeg_margin=18., source_margin=2.):
    """Inverse510-based preprocessing, with bounded JPEG/interpolation guard."""
    geometry=preprocess_geometry(box)
    affine=np.asarray(geometry["affine_2x3"])
    x1,y1,x2,y2=crop_roi
    guarded=[max(0.,x1-jpeg_margin),max(0.,y1-jpeg_margin),min(510.,x2+jpeg_margin),min(510.,y2+jpeg_margin)]
    source=[(guarded[0]-affine[0,2])/affine[0,0],(guarded[1]-affine[1,2])/affine[1,1],
            (guarded[2]-affine[0,2])/affine[0,0],(guarded[3]-affine[1,2])/affine[1,1]]
    return list(map(float,[source[0]-source_margin,source[1]-source_margin,source[2]+source_margin,source[3]+source_margin]))


def native_box(raw):
    x,y,w,h=map(float,raw)
    return [x-1,y-1,w,h]


def _first(value,*keys,default=None):
    for key in keys:
        if key in value:return value[key]
    return default


def gt_descriptor(video):
    descriptor=_first(video,"original_GT","original_gt","gt")
    if isinstance(descriptor,dict):
        return _first(descriptor,"path","original_path"),_first(descriptor,"sha256","SHA256")
    return _first(video,"source_annotation_path","original_gt_path","gt_path"),_first(video,"source_annotation_sha256","original_gt_sha256","gt_sha256")


def parse_gt(path):
    rows,duplicates={},set()
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        for number,row in enumerate(csv.reader(stream),1):
            if not row:continue
            values=list(map(float,row))
            if (len(values)!=9 or not np.isfinite(values).all() or values[7]!=int(values[7])
                    or values[6] not in (0,1) or int(values[7]) not in range(1,13)
                    or not 0<=values[8]<=1):raise ValueError("malformed native 9-column MOT GT row")
            frame,track=int(values[0]),str(int(values[1]))
            if frame<1 or values[0]!=frame or values[1]!=int(values[1]):raise ValueError("non-native frame/track ID")
            item={"frame_id":frame,"track_id":track,"bbox_xywh_raw":values[2:6],"mark":values[6],
                  "class_id":int(values[7]),"visibility":values[8],"source_row":number}
            key=(frame,track)
            if key in rows:duplicates.add(track)
            rows[key]=item
    return rows,duplicates


def normalize_source(manifest,verify_files=True):
    """Validate actual GT endpoints; never invent ID continuity or interpolate."""
    if manifest.get("schema")!="hift-temporal-source-v1":raise ValueError("unknown temporal source schema")
    origin=manifest.get("coordinate_origin",1)
    if origin!=1:raise ValueError("MOT raw one-based origin must be explicit")
    videos,seen_lineages={},set()
    quarantine=[]
    for original in manifest.get("videos",[]):
        lineage=original["video_lineage"]
        if lineage in seen_lineages:raise ValueError("duplicate detector/video lineage")
        seen_lineages.add(lineage)
        split=original["split"]
        if split=="dev":split="development"
        if split not in ("train","development"):raise ValueError("unknown source split")
        info=original["seqinfo"]
        width=int(_first(info,"w","width","imWidth"));height=int(_first(info,"h","height","imHeight"))
        fps=float(_first(info,"fps","frame_rate","frameRate"));total=int(_first(info,"totalframes","total_frames","seqLength"))
        if min(width,height,fps,total)<=0:raise ValueError("invalid source sequence header")
        gt_path,gt_sha=gt_descriptor(original)
        if not gt_path or not gt_sha:raise ValueError("missing complete source GT provenance")
        if verify_files and file_hash(gt_path)!=gt_sha:raise ValueError("source GT hash changed")
        gt,duplicate_tracks=parse_gt(gt_path)
        reused=set(map(str,original.get("suspect_reused_track_ids",[])))|duplicate_tracks
        classes=defaultdict(set)
        for row in gt.values():classes[row["track_id"]].add(row["class_id"])
        reused|={track for track,values in classes.items() if len(values)>1}
        for row in original.get("quarantine",[]):
            if "track_id" in row and any(word in str(row).lower() for word in ("reuse","duplicate","ambiguous")):
                reused.add(str(row["track_id"]))
        frames={}
        for raw_frame in original["frames"]:
            index=int(_first(raw_frame,"frame_id","frame_index"))
            if index in frames or not 1<=index<=total:raise ValueError("duplicate/out-of-range native frame")
            if (int(_first(raw_frame,"width","w",default=width))!=width or
                    int(_first(raw_frame,"height","h",default=height))!=height):raise ValueError("source frame/header dimension contradiction")
            path=raw_frame["path"];sha=_first(raw_frame,"sha256","SHA256")
            if verify_files and file_hash(path)!=sha:raise ValueError("source frame hash changed")
            annotations=[]
            for raw in raw_frame["annotations"]:
                track=str(raw["track_id"])
                canonical=gt.get((index,track))
                if canonical is None:raise ValueError("annotation missing from actual source GT")
                if (raw.get("frame_id",index)!=index or raw.get("source_row",canonical["source_row"])!=canonical["source_row"]
                        or raw.get("identity",["MOT17",lineage,track])!=["MOT17",lineage,track]):
                    raise ValueError("canonical frame/track/source-row identity contradiction")
                if not np.allclose(raw["bbox_xywh_raw"],canonical["bbox_xywh_raw"],rtol=0,atol=1e-6):
                    raise ValueError("manifest annotation contradicts source GT")
                for field in ("mark","class_id","visibility"):
                    if field in raw and not np.isclose(float(raw[field]),canonical[field]):raise ValueError("GT presence/class/visibility contradiction")
                if "presence" in raw and bool(raw["presence"])!=(canonical["mark"]==1):
                    raise ValueError("explicit presence contradicts native source mark")
                flags=raw.get("quarantine_reasons",[])
                if raw.get("id_reuse") or raw.get("identity_ambiguous") or any(word in str(flags).lower() for word in ("reuse","duplicate","ambiguous")):reused.add(track)
                ann=dict(canonical,bbox_xywh=native_box(canonical["bbox_xywh_raw"]),source_quarantine_reasons=flags,
                         presence=bool(raw.get("presence",canonical["mark"]==1)))
                annotations.append(ann)
            # Same-frame duplicate IDs/near-identical duplicate boxes are not disambiguated.
            counts=Counter(a["track_id"] for a in annotations)
            reused|={track for track,count in counts.items() if count>1}
            for i,a in enumerate(annotations):
                for b in annotations[i+1:]:
                    ba,bb=xyxy(a["bbox_xywh"]),xyxy(b["bbox_xywh"])
                    union=a["bbox_xywh"][2]*a["bbox_xywh"][3]+b["bbox_xywh"][2]*b["bbox_xywh"][3]-intersection(ba,bb)
                    if union>0 and intersection(ba,bb)/union>=.99:reused|={a["track_id"],b["track_id"]}
            frames[index]={"frame_id":index,"path":path,"sha256":sha,"phash64":raw_frame.get("phash64"),
                           "width":width,"height":height,"annotations":annotations,"quarantine_reasons":raw_frame.get("quarantine_reasons",[])}
        for track in sorted(reused):quarantine.append({"video_lineage":lineage,"track_id":track,"reason":"duplicate_or_reused_or_ambiguous_track"})
        videos[lineage]={"video_lineage":lineage,"split":split,"fps":fps,"width":width,"height":height,
                         "total_frames":total,"frames":frames,"gt":gt,"quarantined_tracks":reused}
    if not videos or not any(v["split"]=="train" for v in videos.values()) or not any(v["split"]=="development" for v in videos.values()):
        raise ValueError("video-disjoint train and development source required")
    return videos,quarantine


def eligible_endpoint(video,frame,ann,policy):
    if ann["track_id"] in video["quarantined_tracks"]:return "identity_quarantine"
    if ann.get("source_quarantine_reasons"):return "source_annotation_quarantine"
    if not ann["presence"] or ann["mark"]!=1:return "not_present_or_marked_valid"
    if ann["class_id"]!=1:return "non_person_or_ambiguous_class"
    if int(ann["track_id"])<=0:return "nonpositive_author_track_id"
    if ann["visibility"]<policy["minimum_visibility"]:return "insufficient_visibility"
    box=np.asarray(ann["bbox_xywh"])
    if not np.isfinite(box).all() or np.any(box[2:]<=0):return "invalid_bbox"
    if intersection(xyxy(box),[0,0,video["width"],video["height"]])<=0:return "fully_off_image"
    return None


def build_population(videos,policy,split="train"):
    positives,negatives=[],[]
    excluded=Counter();endpoint_count=0;represented={}
    for lineage,video in sorted(videos.items()):
        if video["split"]!=split:continue
        tracks=defaultdict(list);eligible_frames={}
        for index,frame in sorted(video["frames"].items()):
            if frame.get("quarantine_reasons"):
                excluded["source_frame_quarantine"]+=len(frame["annotations"]);continue
            usable=[]
            for ann in frame["annotations"]:
                reason=eligible_endpoint(video,frame,ann,policy)
                if reason:excluded[reason]+=1;continue
                endpoint_count+=1;usable.append(ann);tracks[ann["track_id"]].append((index,ann))
            eligible_frames[index]=usable
        for track,endpoints in sorted(tracks.items()):
            for first,a in endpoints:
                for second,b in endpoints:
                    gap=abs(second-first)
                    if not policy["positive_min_gap"]<=gap<=policy["positive_max_gap"]:continue
                    if video["frames"][first]["sha256"]==video["frames"][second]["sha256"]:
                        excluded["repeated_endpoint_content"]+=1;continue
                    positives.append({"video_lineage":lineage,"track_id":track,"template_frame":first,"search_frame":second})
        # Nominal actual287 FOV +18 crop511 JPEG/interpolation pixels +2
        # source pixels. Training augmentation may invalidate it: reject, never refill.
        for index,annotations in sorted(eligible_frames.items()):
            for a in annotations:
                for b in annotations:
                    if a["track_id"]==b["track_id"]:continue
                    envelope=source_footprint(b["bbox_xywh"],jpeg_margin=policy["jpeg_support_margin_crop511_px"],
                                              source_margin=policy["negative_source_margin_px"])
                    if intersection(xyxy(a["bbox_xywh"]),envelope)>0:
                        excluded["template_target_inside_guarded_nominal_search_footprint"]+=1;continue
                    negatives.append({"video_lineage":lineage,"frame_id":index,
                                      "template_track":a["track_id"],"search_track":b["track_id"]})
        represented[lineage]={"observed_frames":len(video["frames"]),"eligible_tracks":len(tracks),
                              "eligible_endpoints":sum(map(len,tracks.values()))}
    return {"schema":"hift-temporal-population-v1","split":split,"positives":positives,"negatives":negatives,
            "endpoint_count":endpoint_count,"exclusion_counts":dict(excluded),"representation":represented}
