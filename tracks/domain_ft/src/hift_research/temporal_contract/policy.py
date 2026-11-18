"""Frozen pilot policy: trajectory-fragment positives and simultaneous safe negatives."""
from __future__ import annotations
import json
from pathlib import Path
from hift_research.data_contract.schema import canonical_hash,file_hash


def make_policy(split="train"):
    development=split in ("dev","development")
    split="development" if development else "train"
    return {"schema":"hift-temporal-policy-v1","split":split,"seed":20261024 if development else 20261023,
            "preflight_length":128 if development else 1024,"allowed_lengths":[128] if development else [1024,1600],
            "class_schedule":"fixed_alternating_positive_negative_64_each" if development else "bernoulli_negative_0.05",
            "candidate_negative_probability":None if development else .05,
            "positive_min_gap":1,"positive_max_gap":19,"minimum_visibility":.5,"negative_source_margin_px":2.,
            "jpeg_support_margin_crop511_px":18.,
            "positive_sampling":"uniform_ordered_eligible_same_fragment_endpoints_both_directions",
            "negative_sampling":"uniform_known_distinct_simultaneous_pairs_A_outside_nominal287_source_footprint_with_JPEG18_and_source2_guards",
            "negative_policy_deviation":"conditional_temporal_pilot_not_original_arbitrary_negative_video_sampler",
            "replacement_policy":"none","cross_video_negative":"forbidden_unknown_global_aliases",
            "identity_scope":"source_trajectory_fragment_not_global_person",
            "template_aug":dict(shift=0.,scale=0.,blur=0.,flip=0.,color=0.) if development else dict(shift=4.,scale=.05,blur=0.,flip=0.,color=1.),
            "search_aug":dict(shift=0.,scale=0.,blur=0.,flip=0.,color=0.) if development else dict(shift=64.,scale=.18,blur=.2,flip=0.,color=1.),
            "gray":0.,"grid":[63,16,11],"source_crop511":"pinned_SiamFC_meanpad_JPEG95_510_based_affine",
            "source_bbox_labels":"retained_ideal127_contextscale_ratio511_over510",
            "positive_target_support":"require_available_geometric_source_intersection_gt0",
            "negative_postaug_proof":"visible_A_box_disjoint_actual_augROI_expanded18crop511px_then_inverse510_affine_plus2sourcepx",
            "weight64":"retain_source_finite_heuristic_no_retuning","real_SGD_authorized":False}


def load_policy(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    expected=make_policy(value.get("split"))
    if value.get("schema")!=expected["schema"] or value.get("policy")!=expected:
        raise ValueError("policy differs from preregistered temporal pilot semantics")
    if value.get("policy_sha256")!=canonical_hash(expected):raise ValueError("frozen policy digest mismatch")
    for filename,sha in value["source_code_hashes"].items():
        if file_hash(filename)!=sha:raise ValueError("frozen temporal/source code changed: "+filename)
    if file_hash(value["manifest_path"])!=value["manifest_sha256"]:raise ValueError("frozen source manifest changed")
    if file_hash(value["population_path"])!=value["population_sha256"]:raise ValueError("frozen eligible population changed")
    for key in ("reservations","source_content_screen","eligibility_prereg"):
        if key+"_path" in value and file_hash(value[key+"_path"])!=value[key+"_sha256"]:
            raise ValueError("frozen "+key+" changed")
    return value
