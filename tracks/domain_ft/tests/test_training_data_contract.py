"""CPU fixtures only: generated JPEGs NEVER enter authentic acquired training data."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from hift_research.data_contract.acquisition import write_exclusive_json, cached_reader, acquire_subset
from hift_research.data_contract.freeze import freeze_training
from hift_research.data_contract.pipeline import RealPairPipeline, array_digest
from hift_research.data_contract.schema import validate_training_manifest, file_hash, canonical_hash
from hift_research.data_contract.screening import phash64, content_screen
from hift_research.data_contract.source_ops import source_operations, jpeg_crop511, preprocess_geometry


@pytest.fixture
def manifest(tmp_path):
    images, source_rows, annotations = [], [], []
    for image_id in range(1, 4):
        path = tmp_path / f"{image_id:012d}.jpg"
        pixels = np.random.RandomState(image_id).randint(0, 256, (192, 256, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), pixels)
        row = {"id": image_id, "file_name": path.name, "width": 256, "height": 192, "license": 1,
               "coco_url": f"http://images.cocodataset.org/train2017/{path.name}"}
        source_rows.append(row)
        images.append(dict(row, path=str(path), sha256=file_hash(path)))
        annotations.extend([{"id": image_id*10+track, "image_id": image_id, "category_id": 1,
                             "bbox": bbox, "iscrowd": 0}
                            for track, bbox in ((1, [30, 40, 30, 40]), (2, [140, 100, 40, 50]))])
    selection = {"split": "train2017", "images": source_rows, "annotations": annotations,
                 "licenses": [{"id": 1, "name": "UNIT-FIXTURE-NOT-A-DATA-LICENSE"}], "quarantine": []}
    selected_path = tmp_path / "selection.json"
    write_exclusive_json(selected_path, selection)
    acquired_path = tmp_path / "acquisition.json"
    write_exclusive_json(acquired_path, {"selection_path": str(selected_path), "selection_sha256": file_hash(selected_path),
                                       "annotation_path": str(selected_path), "annotation_sha256": file_hash(selected_path),
                                       "sources": [{"url": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"}],
                                       "images": images})
    reserved_path = tmp_path / "reserved.jpg"
    assert cv2.imwrite(str(reserved_path), np.random.RandomState(800).randint(0, 256, (96, 128, 3), dtype=np.uint8))
    eval_path = tmp_path / "eval.json"
    write_exclusive_json(eval_path, {"dataset": "UAV123@10fps", "root": str(tmp_path),
                                    "sequences": [{"name": "reserved", "frame_paths": [reserved_path.name],
                                                   "frame_sha256": [file_hash(reserved_path)]}]})
    return freeze_training(acquired_path, [eval_path], ["reserved"], tmp_path / "frozen")


def _mutate(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_frozen_real_input_contract_replays_without_touching_global_rng(manifest):
    pipeline = RealPairPipeline(manifest)
    np.random.seed(829)
    before = np.random.get_state()
    first, arrays = pipeline.candidate(0)
    second, other = RealPairPipeline(manifest).candidate(0)
    assert first == second and first["status"] == "accepted"
    assert arrays["template"].shape == (3, 127, 127)
    assert arrays["search"].shape == (3, 287, 287)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], other[key])
    for a, b in zip(before, np.random.get_state()):
        np.testing.assert_array_equal(a, b)
    assert first["frame_interval"] == 0 and first["repeated_frame"]


def test_candidate_prior_draws_not_adjusted_for_rejections(manifest):
    pipeline = RealPairPipeline(manifest)
    flags, rejected, accepted_negative = [], [], []
    for index in range(256):
        record, arrays = pipeline.candidate(index)
        expected_flag = bool(np.random.RandomState(20261023 + index).random_sample() < .05)
        assert record["candidate_negative"] == expected_flag
        assert record["replacement_count"] == 0 and record["replacement"] is None
        flags.append(expected_flag)
        if record["status"] == "rejected":
            rejected.append(record)
            assert arrays is None and record["candidate_negative"]
        elif expected_flag:
            accepted_negative.append(record)
            assert np.count_nonzero(arrays["weightxff"]) == 0
    assert any(flags) and rejected and accepted_negative
    assert all(row["reason"] in ("cross_image_physical_alias_unknown", "same_instance_negative_collision") for row in rejected)


@pytest.mark.parametrize("change,reason", [
    (lambda v: v.update(dataset="UAV123"), "COCO"),
    (lambda v: v.update(split="val2017"), "train2017"),
    (lambda v: v["validation"].update(production_training_authorized=True), "preflight"),
    (lambda v: v["policy"].update(candidate_negative_probability=.5), "probability"),
    (lambda v: v["policy"].update(replacement_policy="until_valid"), "replacements"),
    (lambda v: v["instances"][0].update(identity=["COCO2017/train2017", "1", "madeup"]), "identity"),
    (lambda v: v["instances"][0].update(bbox_xywh=[0, 0, 10, 10]), "source annotation"),
    (lambda v: v["instances"][0].update(iscrowd=1), "ambiguous"),
    (lambda v: v["instances"].append(copy.deepcopy(v["instances"][0])), "repeated annotation"),
    (lambda v: v["images"].append(copy.deepcopy(v["images"][0])), "duplicate"),
])
def test_schema_fails_closed_on_leakage_identity_and_recipe_changes(manifest, change, reason):
    _mutate(manifest, change)
    with pytest.raises(ValueError, match=reason):
        validate_training_manifest(manifest)


def test_missing_image_and_corrupt_source_bytes_are_not_substituted(manifest):
    value = validate_training_manifest(manifest)
    path = Path(value["images"][0]["path"])
    path.write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="hash mismatch"):
        RealPairPipeline(manifest)


def test_reserved_content_and_phash_near_matches_are_quarantined():
    image = np.random.RandomState(99).randint(0, 256, (192, 256, 3), dtype=np.uint8)
    perceptual = phash64(image)
    reserved = {"frames": [{"sha256": "abc", "phash64": perceptual, "sequence": "eval"}]}
    assert content_screen("abc", perceptual, reserved)[0] == "reserved_eval_exact_content"
    assert content_screen("different", perceptual, reserved)[0] == "reserved_eval_perceptual_near_match"
    near = f"{int(perceptual,16)^7:016x}"
    assert content_screen("different", near, reserved)[0] is not None


def test_reservations_hash_mutation_is_fatal(manifest):
    value = validate_training_manifest(manifest)
    Path(value["reservations_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="reservations changed"):
        validate_training_manifest(manifest)


def test_source_crop511_matches_original_default_jpeg_and_centered_bbox(tmp_path):
    ops = source_operations()
    image = np.random.RandomState(93).randint(0, 256, (192, 256, 3), dtype=np.uint8)
    box = [40, 50, 30, 60]
    assert not hasattr(np, "float")
    actual, encoded = jpeg_crop511(image, box, ops)
    _, x = ops.crop(image, [40, 50, 70, 110], instanc_size=511, padding=np.mean(image, axis=(0, 1)))
    path = tmp_path / "source-default.jpg"
    assert cv2.imwrite(str(path), x)
    assert encoded == path.read_bytes()
    np.testing.assert_array_equal(actual, cv2.imread(str(path)))
    centered = ops.get_bbox(None, actual, [40, 50, 70, 110])
    np.testing.assert_allclose([(centered.x1+centered.x2)/2, (centered.y1+centered.y2)/2], [255, 255])
    expected_scale = 127 / np.sqrt((30+.5*90)*(60+.5*90))
    np.testing.assert_allclose([centered.x2-centered.x1, centered.y2-centered.y1], [30*expected_scale, 60*expected_scale])
    assert not hasattr(np, "float")


def test_preprocess_affine_uses_510_not_511_and_discloses_label_scale_mismatch():
    geometry = preprocess_geometry([40, 50, 30, 60])
    affine = np.asarray(geometry["affine_2x3"])
    x1,y1,x2,y2 = geometry["source_crop_xyxy"]
    np.testing.assert_allclose(affine @ [x1,y1,1], [0,0], atol=1e-12)
    np.testing.assert_allclose(affine @ [x2,y2,1], [510,510], atol=1e-12)
    assert geometry["actual_source_to_crop_pixel_scale"] == 510/geometry["source_crop_side_px"]
    np.testing.assert_allclose(geometry["retained_label_to_pixel_scale_ratio"], 511/510, rtol=1e-14)


def test_audit_only_augmentation_wrapper_matches_pixels_boxes_and_rng():
    ops = source_operations()
    image = np.random.RandomState(17).randint(0,256,(511,511,3),dtype=np.uint8)
    bbox = ops.Corner(220,215,280,290)
    original = ops.Augmentation(64,.18,.2,0.,1.)
    audited = ops.AuditedAugmentation(64,.18,.2,0.,1.)
    np.random.seed(234)
    a,box_a = original(image,bbox,287)
    next_a = np.random.random(4)
    np.random.seed(234)
    b,box_b = audited(image,bbox,287)
    next_b = np.random.random(4)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(box_a,box_b)
    np.testing.assert_array_equal(next_a,next_b)
    assert np.asarray(audited.last_crop_geometry["affine_2x3"]).shape == (2,3)


def test_candidate_audit_fields_exist_for_rejections_and_accepted_shapes(manifest):
    pipeline = RealPairPipeline(manifest)
    records = [pipeline.candidate(i) for i in range(50)]
    assert any(record["status"]=="rejected" for record, arrays in records)
    for record, arrays in records:
        for key in ("dataset","pair_type","template_identity_key","search_identity_key",
                    "template_source_bbox_xywh","search_source_bbox_xywh",
                    "template_preprocess_geometry","search_preprocess_geometry"):
            assert record[key] is not None
        if arrays is not None:
            assert record["template_augmented_bbox"] is not None
            assert record["search_augmented_bbox"] is not None
            for key,array in arrays.items():
                assert record["input_contract"][key] == {"dtype":str(array.dtype),"shape":list(array.shape)}


def test_dataset_pickle_drops_dynamic_ops_and_collate_is_reject_aware(manifest):
    torch = pytest.importorskip("torch")
    from hift_research.data_contract.torch_adapter import CandidateDataset, reject_aware_collate
    dataset = CandidateDataset(manifest)
    first = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._pipeline is None and restored[0][0] == first[0]
    rows = [dataset[i] for i in range(50)]
    reject = next(row for row in rows if row[0]["status"]=="rejected")
    accepted = next(row for row in rows if row[0]["status"]=="accepted")
    batch = reject_aware_collate([accepted,reject])
    assert batch["candidate_count"] == 2 and batch["replacement_count"] == 0
    assert len(batch["rejected_records"]) == len(batch["accepted_records"]) == 1
    assert batch["model_inputs"]["template"].shape == (1,3,127,127)
    assert all(value.dtype==torch.float32 and value.device.type=="cpu" for value in batch["model_inputs"].values())
    assert reject_aware_collate([reject])["model_inputs"] is None


def test_spawn_two_worker_loader_matches_serial_candidates(manifest):
    torch = pytest.importorskip("torch")
    from hift_research.data_contract.torch_adapter import CandidateDataset, reject_aware_collate
    dataset = CandidateDataset(manifest)
    expected = [dataset[i][0] for i in range(4)]
    loader = torch.utils.data.DataLoader(dataset, sampler=list(range(4)), batch_size=2,
                                        num_workers=2, multiprocessing_context="spawn", timeout=60,
                                        prefetch_factor=1, collate_fn=reject_aware_collate)
    actual = []
    for batch in loader:
        actual.extend(batch["accepted_records"]+batch["rejected_records"])
    assert sorted(actual,key=lambda row:row["candidate_index"]) == expected


def test_explicit_negative_edge_probe_is_separate_from_candidate_prior(manifest,tmp_path):
    from hift_research.data_contract.edge_probe import probe_negative
    result = probe_negative(manifest,tmp_path/"edge")
    assert result["not_in_1024_candidate_statistics"] and result["candidate_replacements"]==0
    assert result["template_identity"][1]==result["search_identity"][1]
    assert result["template_identity"]!=result["search_identity"]
    assert result["bbox_iou"]==0 and result["negative_regression_weight_nonzeros"]==0
    assert not (tmp_path/"edge"/"candidate_ledger.jsonl").exists()


def test_array_hash_includes_dtype_and_shape_not_only_payload():
    value = np.arange(12, dtype=np.float32)
    assert array_digest(value) != array_digest(value.reshape(3, 4))
    assert array_digest(value) != array_digest(value.astype(np.float64))


def test_acquisition_rejects_unapproved_budget_and_excess_images_before_network(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        acquire_subset("missing", tmp_path / "out", tmp_path / "data", budget=193*1024*1024)
    with pytest.raises(ValueError):
        acquire_subset("missing", tmp_path / "out", tmp_path / "data", image_limit=33)


def test_exclusive_artifacts_do_not_overwrite(tmp_path):
    path = tmp_path / "manifest.json"
    write_exclusive_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_exclusive_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
