"""No-network/no-GPU synthetic source parser and immutable selection fixtures."""
from copy import deepcopy
from pathlib import Path

import pytest

from hift_research.data_acquisition.range_zip import HTTPRangeReader
from hift_research.evaluation.assets import sha256_file
from hift_research.temporal_source.acquisition import restore_index_cache, validate_selection
from hift_research.temporal_source.annotations import (
    IDENTITY_SCOPE, native_to_zero_origin, parse_ground_truth, parse_seqinfo, selected_frame_ids,
)

SEQ = "[Sequence]\nname=MOT17-02-DPM\nimDir=img1\nframeRate=30\nseqLength=600\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n"


def test_exact_original_seqinfo_metadata_and_explicit_origin_conversion():
    info = parse_seqinfo(SEQ, "MOT17-02-DPM")
    assert [info[k] for k in ("fps", "w", "h", "totalframes")] == [30, 1920, 1080, 600]
    box = [1, 2, 3, 4]
    assert native_to_zero_origin(box) == [0, 1, 3, 4]
    assert box == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        parse_seqinfo(SEQ, "other")
    with pytest.raises(ValueError):
        parse_seqinfo(SEQ.replace("imDir=img1", "imDir=../escape"))


@pytest.mark.parametrize("n,ranges", [(600, [117, 136, 407, 426]), (525, [102, 121, 354, 373]),
                                     (654, [127, 146, 444, 463]), (837, [164, 183, 572, 591])])
def test_window_rule_frozen_independent_of_annotations(n, ranges):
    frames = selected_frame_ids(n)
    assert len(frames) == len(set(frames)) == 40
    assert [frames[0], frames[19], frames[20], frames[39]] == ranges
    assert all(b - a == 1 for a, b in zip(frames[:19], frames[1:20]))
    assert frames[19] - frames[0] == 19


def test_short_or_overlapping_windows_rejected():
    with pytest.raises(ValueError):
        selected_frame_ids(30)


def test_nine_column_gt_preserves_ignore_visibility_and_native_raw_box():
    result = parse_ground_truth("1,1,-5,2,10,20,1,1,0\n1,2,30,2,10,20,0,8,0.7", parse_seqinfo(SEQ))
    assert len(result["rows"]) == 2
    first, second = result["rows"]
    assert first["bbox_xywh_raw"] == [-5, 2, 10, 20]
    assert first["presence"] is True and first["visibility"] == 0
    assert second["mark"] == 0 and second["class_id"] == 8 and second["presence"] is False
    assert second["source_row"] == 2
    assert result["identity_scope"] == IDENTITY_SCOPE
    assert "NOT proof" in result["identity_limitations"]


@pytest.mark.parametrize("text", ["1,1,1,2,3,4,1,1,1,-1", "1,1,1,2,nan,4,1,1,1",
                                  "0,1,1,2,3,4,1,1,1", "1,1.5,1,2,3,4,1,1,1"])
def test_submission_format_nonfinite_and_invalid_source_fields_fail(text):
    with pytest.raises(ValueError):
        parse_ground_truth(text, parse_seqinfo(SEQ))


def test_complete_gt_duplicate_class_change_and_geometric_jump_hints_are_quarantined():
    text = "1,1,0,0,10,10,1,1,1\n1,1,0,0,10,10,1,1,1\n2,1,1000,0,10,10,1,2,1"
    result = parse_ground_truth(text, parse_seqinfo(SEQ))
    reasons = {r["reason"] for r in result["quarantine"]}
    assert "duplicate_frame_track_source_rows" in reasons
    assert "class_change_identity_ambiguity_hint" in reasons
    assert any("geometric_jump" in reason for reason in reasons)
    assert len(result["rows"]) == 3  # Never delete or repair original source rows.
    assert all(r["quarantine_reasons"] for r in result["rows"])


def test_missing_mid_annotations_logged_without_interpolation_or_physical_identity_claim():
    result = parse_ground_truth("1,1,0,0,10,10,1,1,1\n3,1,1,0,10,10,1,1,1", parse_seqinfo(SEQ))
    assert [r["frame_id"] for r in result["rows"]] == [1, 3]
    assert result["missing_annotation_intervals"][0]["missing_intermediate_annotation_frames"] == 1
    assert result["missing_annotation_intervals"][0]["interpolated"] is False


def test_invalid_finite_source_box_is_retained_but_quarantined():
    result = parse_ground_truth("1,1,0,0,0,10,1,1,1", parse_seqinfo(SEQ))
    assert result["rows"][0]["quarantine_reasons"] == ["nonpositive_source_box"]
    with pytest.raises(ValueError):
        native_to_zero_origin([0, 0, 0, 10])


def selection_fixture():
    return {"status": "frozen_before_images_and_pair_outcomes", "production_training_authorized": False,
            "initial_global_network_budget_bytes": 128 * 1024 * 1024, "total_original_frames": 80,
            "videos": [{"video_lineage": name, "split": split, "seqinfo_expected": {"totalframes": 600},
                        "frames": [{"frame_id": i} for i in selected_frame_ids(600)]}
                       for name, split in (("MOT17-02", "train"), ("MOT17-05", "development"))]}


def test_frozen_selection_enforces_split_frame_rule_and_global_bounds():
    good = selection_fixture()
    validate_selection(good)
    for mutation in ("lineage", "frame", "budget", "authority"):
        bad = deepcopy(good)
        if mutation == "lineage":
            bad["videos"][1]["video_lineage"] = bad["videos"][0]["video_lineage"]
        elif mutation == "frame":
            bad["videos"][0]["frames"][0]["frame_id"] += 1
        elif mutation == "budget":
            bad["initial_global_network_budget_bytes"] *= 2
        else:
            bad["production_training_authorized"] = True
        with pytest.raises(ValueError):
            validate_selection(bad)


def test_independent_annotation_comparison_rejects_presence_or_box_repair():
    from hift_research.temporal_source.validation import assert_annotation_matches_source
    fields = ["7", "12", "1", "2", "3", "4", "0", "8", "0.7"]
    row = {"frame_id": 7, "track_id": "12", "bbox_xywh_raw": [1, 2, 3, 4], "mark": 0,
           "class_id": 8, "visibility": .7, "presence": False, "source_row": 27}
    assert_annotation_matches_source(row, fields, 7, 27)
    for key, value in (("presence", True), ("bbox_xywh_raw", [0, 1, 3, 4]), ("source_row", 26)):
        bad = {**row, key: value}
        with pytest.raises(ValueError):
            assert_annotation_matches_source(bad, fields, 7, 27)


def test_nonpositive_ignored_track_ids_remain_raw_but_never_eligible():
    result = parse_ground_truth("1,-1,0,0,10,10,0,9,1", parse_seqinfo(SEQ))
    row = result["rows"][0]
    assert row["track_id"] == "-1" and row["presence"] is False
    assert "nonpositive_source_track_id" in row["quarantine_reasons"]


def test_reused_index_cache_hashes_must_match(tmp_path):
    path = tmp_path / "range.bin"
    path.write_bytes(b"abc")
    ledger = {"ranges": [{"cache_path": str(path), "sha256": sha256_file(path), "start": 1, "end": 3}]}
    reader = HTTPRangeReader("https://example.invalid/a.zip", 10)
    restore_index_cache(reader, ledger)
    reader.seek(1)
    assert reader.read(3) == b"abc"  # Cache hit: no network.
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError):
        restore_index_cache(reader, ledger)
