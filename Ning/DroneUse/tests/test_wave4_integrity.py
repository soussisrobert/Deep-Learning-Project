"""Read-only preservation and adversarial real-ledger algebra regressions, no model."""
import json
from pathlib import Path
import numpy as np
import pytest
from hift_research.wave4_integrity import verify_prior,independent_pair_algebra

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'artifacts/wave4-source/SOURCE-001/temporal_manifest.json'
LEDGER=ROOT/'artifacts/wave4-preflight/PREFLIGHT-002/train/candidate_ledger.jsonl'
LOCAL_ASSETS=pytest.mark.skipif(not (MANIFEST.is_file() and LEDGER.is_file()),reason='bounded authentic local assets are not committed')


@pytest.mark.skipif(not (ROOT/'artifacts/checkpoints/general_model.pth').is_file(),reason='official local checkpoint is not committed')
def test_preservation_checks_all_prior_files_not_only_selected_new_paths():
    receipt=verify_prior(ROOT)
    assert receipt['protected_prior_file_count']==140
    assert receipt['upstream_clean'] is True
    assert len(receipt['prior_receipt_references']['experiments/wave3_validation.json'])==21
    assert len(receipt['prior_receipt_references']['experiments/wave2_validation.json'])==5
    if (ROOT/'artifacts/wave4-integration/START-001.json').is_file():
        assert receipt['actual_working_bytes_equal_wave4_start'] is True


def altered_ledger(tmp_path,mutate):
    records=[json.loads(line) for line in LEDGER.read_text(encoding='utf-8').splitlines() if line.strip()]
    mutate(records)
    target=tmp_path/'explicitly_corrupted_test_fixture_NOT_real_experiment.jsonl'
    target.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    return target


@LOCAL_ASSETS
def test_independent_audit_rejects_wrong_release_identity_namespace(tmp_path):
    def corrupt(records):records[0]['template_identity'][0]='MOT17/train'
    with pytest.raises(ValueError,match='namespace'):
        independent_pair_algebra(MANIFEST,altered_ledger(tmp_path,corrupt))


@LOCAL_ASSETS
def test_independent_audit_recomputes_negative_footprint_not_boolean_assertion(tmp_path):
    def corrupt(records):
        negative=next(row for row in records if row['candidate_negative'] and row['status']=='accepted')
        negative['negative_guarded_source_footprint_xyxy'][0]-=5
        assert negative['negative_exclusion_proved'] is True
    with pytest.raises(ValueError,match='footprint mismatch'):
        independent_pair_algebra(MANIFEST,altered_ledger(tmp_path,corrupt))


@LOCAL_ASSETS
def test_independent_audit_detects_encoded_grid_target_drift(tmp_path):
    def corrupt(records):
        first=records[0]
        with np.load(first['array_archive_path'],allow_pickle=False) as original:
            targets={key:original[key] for key in ('bbox','label_cls1','labelcls2','labelxff','weightxff')}
        targets['labelxff'][0,0,0]+=.01
        path=tmp_path/'corrupt_target_fixture.npz';np.savez(path,**targets)
        first['array_archive_path']=str(path)
    with pytest.raises(ValueError,match='offset semantics mismatch'):
        independent_pair_algebra(MANIFEST,altered_ledger(tmp_path,corrupt))
