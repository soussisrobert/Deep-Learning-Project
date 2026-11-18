"""Stage 5: VisDrone-SOT train_part2 selection and the exact-duplicate frame check (CPU, synthetic data)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT/"scripts"))
    spec.loader.exec_module(module)
    return module


def _sot_part(data, part, names, n=6, seed=0):
    import cv2
    base = data/f"visdrone/SOT/VisDrone2019-SOT-train_part{part}/VisDrone2019-SOT-train"
    rng = np.random.default_rng(seed)
    for name in names:
        seq = base/"sequences"/name
        seq.mkdir(parents=True)
        for i in range(1, n+1):
            cv2.imwrite(str(seq/f"img{i:07d}.jpg"), rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
        (base/"annotations").mkdir(exist_ok=True)
        (base/"annotations"/f"{name}.txt").write_text("\n".join("5,5,10,10" for _ in range(n)))
    return base


def test_both_sot_parts_are_loaded(tmp_path):
    trainer = _load("train_b2_domain", "scripts/train_b2_domain.py")
    _sot_part(tmp_path, 1, ["uav0000001_00000_s", "uav0000002_00000_s"])
    _sot_part(tmp_path, 2, ["uav0000101_00000_s", "uav0000102_00000_s", "uav0000103_00000_s"], seed=1)
    one = trainer.domain_tracks("train", tmp_path, ["visdrone-sot-1"])
    both = trainer.domain_tracks("train", tmp_path, ["visdrone-sot-1", "visdrone-sot-2"])
    assert [t["name"] for t in one] == ["uav0000001_00000_s", "uav0000002_00000_s"]
    assert [t["name"] for t in both] == [t["name"] for t in one]+["uav0000101_00000_s", "uav0000102_00000_s",
                                                                   "uav0000103_00000_s"]
    assert {t["source"] for t in both} == {"VisDrone-SOT"}  # same source weight as part1: SOT share unchanged
    assert all(len(t["frames"]) == len(t["boxes"]) == 6 for t in both)


def test_selected_missing_part_is_an_error(tmp_path):
    trainer = _load("train_b2_domain", "scripts/train_b2_domain.py")
    _sot_part(tmp_path, 1, ["uav0000001_00000_s"])
    with pytest.raises(FileNotFoundError):
        trainer.domain_tracks("train", tmp_path, ["visdrone-sot-1", "visdrone-sot-2"])


def test_frame_overlap_detects_exact_copies_only(tmp_path):
    fo = _load("frame_overlap", "scripts/frame_overlap.py")
    train, val = tmp_path/"train", tmp_path/"val"
    (train/"a").mkdir(parents=True); (val/"b").mkdir(parents=True)
    (train/"a"/"1.jpg").write_bytes(b"same-bytes"); (val/"b"/"9.jpg").write_bytes(b"same-bytes")
    (train/"a"/"2.jpg").write_bytes(b"train-only"); (val/"b"/"8.jpg").write_bytes(b"train-only ")  # one byte differs
    (train/"a"/"notes.txt").write_bytes(b"same-bytes")  # non-image files are ignored
    hashes = fo.hash_tree(train)
    assert sorted(hashes) == ["a/1.jpg", "a/2.jpg"]
    assert fo.overlap(hashes, fo.hash_tree(val)) == {"a/1.jpg": ["b/9.jpg"]}
    assert fo.overlap(hashes, {}) == {}


def test_frame_overlap_cli_and_hash_list_round_trip(tmp_path, monkeypatch):
    fo = _load("frame_overlap", "scripts/frame_overlap.py")
    train, val = tmp_path/"train", tmp_path/"val"
    train.mkdir(); val.mkdir()
    for i in range(3):
        (train/f"{i}.jpg").write_bytes(bytes([i]))
    (val/"x.jpg").write_bytes(bytes([2]))
    out, lst = tmp_path/"o.json", tmp_path/"h.sha256"
    monkeypatch.setattr(sys, "argv", ["fo", "--train", str(train), "--eval", f"V1={val}", "--out", str(out),
                                      "--hashes-out", str(lst)])
    fo.main()
    first = json.loads(out.read_text())
    assert first["train_frames"] == 3 and first["eval"]["V1"]["duplicate_train_frames"] == 1
    assert fo.read_hashes(lst) == fo.hash_tree(train)
    monkeypatch.setattr(sys, "argv", ["fo", "--train-hashes", str(lst), "--eval", f"V1={val}", "--out", str(out)])
    fo.main()
    assert json.loads(out.read_text())["eval"] == first["eval"]


def test_frame_overlap_rejects_malformed_inputs(tmp_path, monkeypatch):
    fo = _load("frame_overlap", "scripts/frame_overlap.py")
    with pytest.raises(FileNotFoundError):
        fo.hash_tree(tmp_path/"missing")
    (tmp_path/"t").mkdir()
    (tmp_path/"t"/"1.jpg").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["fo", "--train", str(tmp_path/"t"), "--eval", "no-equals-sign",
                                      "--out", str(tmp_path/"o.json")])
    with pytest.raises(ValueError):
        fo.main()
    (tmp_path/"empty").mkdir()
    monkeypatch.setattr(sys, "argv", ["fo", "--train", str(tmp_path/"empty"), "--eval", f"V={tmp_path/'t'}",
                                      "--out", str(tmp_path/"o.json")])
    with pytest.raises(ValueError):
        fo.main()
