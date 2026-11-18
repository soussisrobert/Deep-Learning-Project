"""Stage 7: close-out sealed-test intervals (scripts/stage7_closeout.py). CPU, synthetic; no stored predictions read."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _closeout():
    sys.path.insert(0, str(ROOT/"src"))
    spec = importlib.util.spec_from_file_location("stage7_closeout", ROOT/"scripts/stage7_closeout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,expected", [("car1_1", "car1"), ("car1_3", "car1"), ("car10", "car10"),
                                           ("person12_2", "person12"), ("group3_4", "group3"), ("truck1", "truck1")])
def test_uav123_segments_share_their_video_cluster(name, expected):
    assert _closeout().cluster("UAV123", name) == expected


def test_visdrone_cluster_is_source_flight_video():
    c = _closeout()
    assert c.cluster("VISDRONE", "uav0000079_02568_s") == "uav0000079"
    assert c.cluster("VISDRONE", "uav0000079_00480_s") == c.cluster("VISDRONE", "uav0000079_02568_s")


@pytest.mark.parametrize("dataset,name", [("VISDRONE", "car1_1"), ("VISDRONE", "uav0000079_02568"), ("UAV123", "../car1"),
                                          ("UAV123", "uav0000079_02568_s"), ("UAV123", "")])
def test_malformed_names_are_rejected(dataset, name):
    with pytest.raises(ValueError):
        _closeout().cluster(dataset, name)


def test_recorded_windows_paths_map_inside_the_checkout(tmp_path):
    c = _closeout()
    assert c.local(r"F:\Code\HiFT\artifacts\b2-eval\X\car1.csv", tmp_path) == tmp_path/"artifacts/b2-eval/X/car1.csv"
    for bad in (r"F:\Other\artifacts\x.csv", r"F:\Code\HiFT\..\secret.txt"):
        with pytest.raises(ValueError):
            c.local(bad, tmp_path)


def test_bootstrap_is_sequence_macro_with_cluster_multiplicity():
    c = _closeout()
    deltas, clusters = np.array([1.0, 3.0, 10.0]), ["a", "a", "b"]  # cluster a has two sequences
    labels, idx = c.cluster_index(clusters)
    weights = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # a twice, b twice, both once
    assert np.allclose(c.boot_means(deltas, idx, weights), [2.0, 10.0, 14/3])


def test_bootstrap_matches_brute_force_resampling():
    c = _closeout()
    rng = np.random.default_rng(1)
    deltas, clusters = rng.normal(size=9), list("aabbbcdde")
    labels, idx = c.cluster_index(clusters)
    weights = c.draw_weights(len(labels), 50, np.random.default_rng(7))
    picks = np.random.default_rng(7).integers(0, len(labels), size=(50, len(labels)))
    brute = [np.mean([d for p in row for d, i in zip(deltas, idx) if i == p]) for row in picks]
    assert np.allclose(c.boot_means(deltas, idx, weights), brute, rtol=0, atol=1e-12)


def test_bootstrap_is_deterministic_and_constant_deltas_have_zero_width():
    c = _closeout()
    labels, idx = c.cluster_index(list("abcabc"))
    w1, w2 = c.draw_weights(3, 200, np.random.default_rng(5)), c.draw_weights(3, 200, np.random.default_rng(5))
    assert np.array_equal(w1, w2) and np.all(w1.sum(axis=1) == 3)
    assert np.allclose(c.boot_means(np.full(6, 0.25), idx, w1), 0.25)


@pytest.mark.parametrize("bad", [np.array([0.1, np.nan, 0.2]), np.array([0.1, np.inf, 0.2]), np.array([]), np.ones((3, 1))])
def test_bootstrap_rejects_nonfinite_empty_or_misaligned(bad):
    c = _closeout()
    labels, idx = c.cluster_index(list("abc"))
    with pytest.raises(ValueError):
        c.boot_means(bad, idx, np.ones((2, 3)))


def test_leave_one_cluster_out_extremes():
    c = _closeout()
    assert c.leave_one_out([1.0, 3.0, 10.0], ["a", "a", "b"]) == (2.0, 10.0)
    with pytest.raises(ValueError):
        c.leave_one_out([1.0, 2.0], ["a", "a"])


@pytest.mark.parametrize("lo,hi,seeds,expected", [
    (0.01, 0.10, [0.05, 0.04], "REF improves over B0"),
    (0.01, 0.10, [0.05, -0.001], "no difference detectable at this sample size"),  # one seed not positive
    (0.0, 0.10, [0.05, 0.04], "no difference detectable at this sample size"),     # bound must be strictly > 0
    (-0.05, -0.001, [-0.02, -0.03], "REF worse than B0"),
    (-0.05, 0.02, [-0.02, 0.01], "no difference detectable at this sample size")])
def test_preregistered_verdict(lo, hi, seeds, expected):
    assert _closeout().verdict(lo, hi, seeds) == expected


def _fixture(tmp_path, c):
    """One stored run in the stage-2 layout: predictions, inference manifest, checkpoint and evaluation report."""
    from hift_research.evaluation import aggregate_sequences, evaluate_sequence
    from hift_research.evaluation.assets import sha256_file
    gt = {"car1_1": [[10, 10, 20, 20], [12, 11, 20, 20], [14, 12, 20, 20]], "car2": [[50, 50, 8, 8], [51, 50, 8, 8]]}
    pred = {"car1_1": [[10, 10, 20, 20], [13, 11, 19, 20], [30, 30, 20, 20]], "car2": [[50, 50, 8, 8], [52, 51, 8, 8]]}
    run = tmp_path/"artifacts/b2-eval/RUN"
    run.mkdir(parents=True)
    (tmp_path/"artifacts/ckpt.pt").write_bytes(b"weights")
    (run/"manifest.json").write_text(json.dumps({"checkpoint": {"path": r"F:\Code\HiFT\artifacts\ckpt.pt",
                                                                "sha256": sha256_file(tmp_path/"artifacts/ckpt.pt")}}))
    scores, inputs = {}, {}
    for name, boxes in pred.items():
        (run/f"{name}.csv").write_text("\n".join(",".join(str(v) for v in b) for b in boxes)+"\n")
        scores[name] = evaluate_sequence(np.asarray(gt[name], float), np.asarray(boxes, float), invalid_policy="count_as_failure")
        inputs[name] = {"path": "unused", "sha256": sha256_file(run/f"{name}.csv")}
    report = {"prediction_coordinate_origin": "annotation_native", "invalid_policy": "count_as_failure",
              "inference_run_manifest": {"sha256": sha256_file(run/"manifest.json")}, "prediction_inputs": inputs,
              "sequences": scores, "aggregate": aggregate_sequences(scores)}
    return report, run, gt


def test_gate_passes_on_an_exact_stored_run(tmp_path):
    c = _closeout()
    report, run, gt = _fixture(tmp_path, c)
    assert c.check_run(report, run, gt, tmp_path) == []


def test_gate_detects_tampered_predictions_scores_checkpoint_and_sequence_sets(tmp_path):
    c = _closeout()
    report, run, gt = _fixture(tmp_path, c)
    (run/"car2.csv").write_text("50,50,8,8\n60,60,8,8\n")
    assert c.check_run(report, run, gt, tmp_path) == ["G1 prediction hash car2"]
    report, run, gt = _fixture(tmp_path/"b", c)
    report["sequences"]["car1_1"]["success_auc"] += 1e-9
    assert c.check_run(report, run, gt, tmp_path/"b") == ["G4 car1_1 success_auc"]
    report, run, gt = _fixture(tmp_path/"c", c)
    (tmp_path/"c/artifacts/ckpt.pt").write_bytes(b"other weights")
    assert c.check_run(report, run, gt, tmp_path/"c") == ["G3 checkpoint hash"]
    report, run, gt = _fixture(tmp_path/"d", c)
    gt["car3"] = gt["car2"]
    assert c.check_run(report, run, gt, tmp_path/"d") == ["G4 sequence sets differ"]
