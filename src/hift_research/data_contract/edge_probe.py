"""Explicit real same-image negative edge test OUTSIDE all1024 candidate draws."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from hift_research.corrections.positive_targets import robust_pair_targets
from .acquisition import write_exclusive_json
from .pipeline import RealPairPipeline, numpy_seed, box_iou_xywh, array_digest
from .schema import file_hash


def probe_negative(manifest_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipeline = RealPairPipeline(manifest_path)
    pair = None
    for image_id in pipeline.image_ids:
        rows = pipeline.grouped[image_id]
        for offset, template in enumerate(rows):
            for search in rows[offset+1:]:
                if box_iou_xywh(template["bbox_xywh"],search["bbox_xywh"]) == 0:
                    pair = template,search
                    break
            if pair:
                break
        if pair:
            break
    if pair is None:
        raise ValueError("no source-legitimate spatially-disjoint instance pair; edge probe unresolved")
    template,search = pair
    selection = {"scope":"dedicated_real_negative_edge_probe_outside_1024",
                 "selection_rule":"smallest_image_then_annotationIDs_with_distinct_noncrowd_disjoint_bboxes",
                 "manifest_sha256":file_hash(manifest_path), "candidate_replacements":0,
                 "image_id":template["image_id"], "template_identity":template["identity"], "search_identity":search["identity"],
                 "template_source_bbox_xywh":template["bbox_xywh"], "search_source_bbox_xywh":search["bbox_xywh"],
                 "template_category_id":template["category_id"],"search_category_id":search["category_id"],
                 "bbox_iou":0.,"seed":20261023+2048,"identity_scope":"visible_annotation_instances_not_global_physical_identity"}
    write_exclusive_json(output/"selection_before_targets.json",selection)
    with numpy_seed(selection["seed"]):
        template_crop,template_sha = pipeline._prepared(template)
        search_crop,search_sha = pipeline._prepared(search)
        def centered(instance,crop):
            x,y,w,h = instance["bbox_xywh"]
            return pipeline.ops.get_bbox(None,crop,[x,y,x+w,y+h])
        t,tbox = pipeline.template_aug(template_crop,centered(template,template_crop),127,gray=False)
        s,sbox = pipeline.search_aug(search_crop,centered(search,search_crop),287,gray=False)
        targets = robust_pair_targets(pipeline.anchor,sbox,11,is_negative=True,
                                      template_identity=tuple(template["identity"]),search_identity=tuple(search["identity"]))
        arrays = dict(zip(("label_cls1","labelxff","labelcls2","weightxff"),targets))
        arrays.update(template=t.transpose(2,0,1).astype(np.float32), search=s.transpose(2,0,1).astype(np.float32),
                      bbox=np.asarray(sbox,dtype=np.float64))
    if any(np.count_nonzero(value) for value in targets) or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("real source negative did not yield finite dense-zero labels/masks")
    archive = output/"arrays.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    result = dict(selection, all_checks_passed=True, template_crop511_jpeg_sha256=template_sha,
                  search_crop511_jpeg_sha256=search_sha, template_augmented_bbox=list(map(float,tbox)),
                  search_augmented_bbox=list(map(float,sbox)), template_augmentation_geometry=pipeline.template_aug.last_crop_geometry,
                  search_augmentation_geometry=pipeline.search_aug.last_crop_geometry,
                  array_hashes={key:array_digest(value) for key,value in arrays.items()}, array_archive_sha256=file_hash(archive),
                  input_shapes={key:list(value.shape) for key,value in arrays.items()},
                  negative_cls1_nonzeros=0,negative_cls2_nonzeros=0,negative_regression_weight_nonzeros=0,
                  no_model_forward_sgd_gpu=True, not_in_1024_candidate_statistics=True)
    write_exclusive_json(output/"result.json",result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    print(json.dumps(probe_negative(args.manifest,args.output),sort_keys=True))
