"""Stage 6: V1B extra validation subset (VisDrone-SOT train_part2 minus training/sealed source videos). CPU, synthetic."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _validate():
    sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"scripts"))
    spec = importlib.util.spec_from_file_location("validate_domain", ROOT/"scripts/validate_domain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names_by_split():
    out, cur = {}, None
    for line in (ROOT/"reports/source_video_audit_names.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = line.split("/")[1].split()[0].replace("VisDrone2019-", ""); out[cur] = []
        elif line.startswith("uav"):
            out[cur].append(line.strip())
    return out


def test_committed_list_is_disjoint_from_training_and_sealed_videos():
    names, split = _validate().v1b_names(), _names_by_split()
    video = lambda xs: {x[:10] for x in xs}
    assert len(names) == 35 and set(names) <= set(split["SOT-train_part2"])
    forbidden = video(split["MOT-train"]) | video(split["SOT-train_part1"]) | video(split["SOT-test-dev"]) | video(split["SOT-val"])
    assert not video(names) & forbidden


def test_malformed_lists_are_rejected(tmp_path):
    vd = _validate()
    for text in ["", "# only a comment\n", "uav0000001_00000_s\nuav0000001_00000_s\n", "uav0000001_00000_v\n", "x\n"]:
        (tmp_path/"l.txt").write_text(text)
        with pytest.raises(ValueError):
            vd.v1b_names(tmp_path/"l.txt")
    (tmp_path/"l.txt").write_text("# c\nuav0000001_00000_s\n\nuav0000002_00010_s\n")
    assert vd.v1b_names(tmp_path/"l.txt") == ["uav0000001_00000_s", "uav0000002_00010_s"]


def _part2(data, names, n=30):
    import cv2
    base = data/"visdrone/SOT/VisDrone2019-SOT-train_part2/VisDrone2019-SOT-train"
    rng = np.random.default_rng(0)
    for name in names:
        (base/"sequences"/name).mkdir(parents=True)
        for i in range(1, n+1):
            cv2.imwrite(str(base/"sequences"/name/f"img{i:07d}.jpg"), rng.integers(0, 255, (40, 60, 3), dtype=np.uint8))
        (base/"annotations").mkdir(exist_ok=True)
        (base/"annotations"/f"{name}.txt").write_text("\n".join("5,5,12,12" for _ in range(n)))


def test_v1b_episodes_follow_the_list_and_missing_sequences_fail(tmp_path, monkeypatch):
    vd = _validate()
    listed = ["uav0000001_00000_s", "uav0000002_00000_s"]
    (tmp_path/"l.txt").write_text("\n".join(listed))
    monkeypatch.setattr(vd, "V1B_LIST", tmp_path/"l.txt")
    _part2(tmp_path, listed+["uav0000003_00000_s"])  # an unlisted part2 sequence is ignored
    episodes, _ = vd.build_episodes(tmp_path, ["V1B"], 0.1)
    assert [(s, e["name"]) for s, e in episodes] == [("V1B", n) for n in listed]
    assert all(len(e["frames"]) == 30 for _, e in episodes)  # full length, like V1
    (tmp_path/"l.txt").write_text("\n".join(listed+["uav0000004_00000_s"]))
    with pytest.raises(FileNotFoundError):
        vd.build_episodes(tmp_path, ["V1B"], 0.1)
