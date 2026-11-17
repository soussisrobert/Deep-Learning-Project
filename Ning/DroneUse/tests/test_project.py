import json
from pathlib import Path
import subprocess
import pytest
from hift_research import UPSTREAM_COMMIT
from hift_research.provenance import new_run, sha256, source_identity

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_is_exact_and_clean():
    identity = source_identity(ROOT)
    assert identity['upstream_commit'] == UPSTREAM_COMMIT
    assert len(identity['source_tree_sha256']) == 64


def test_baseline_identities_are_separate_and_research_gated():
    values = [json.loads((ROOT / f'configs/{name}.json').read_text()) for name in ['B0', 'B1-score', 'B2', 'R1']]
    assert len({v['identity'] for v in values}) == 4
    assert values[0]['corrections'] == []
    assert values[1]['corrections'] == ['sigmoid_BCE_classification_branch_at_inference']
    assert not values[2]['launch_allowed'] and not values[3]['launch_allowed']


def test_run_directory_never_overwrites(tmp_path):
    directory = tmp_path / 'unique'
    new_run(directory)
    with pytest.raises(FileExistsError):
        new_run(directory)
    file = directory / 'file'
    file.write_bytes(b'abc')
    assert sha256(file) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
