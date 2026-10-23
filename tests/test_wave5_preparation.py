import copy
import hashlib
import pytest

from hift_research.wave5_preparation import (assert_unapproved,unapproved_proposal,verify_map,exclusive_json,digest)


def proposal(tmp_path):
    return unapproved_proposal(tmp_path,'B2-W5-TEST-ONLY',{'path':'inventory.json','sha256':'0'*64},
        '2026-10-24T16:00:00+00:00','GPU-test-not-a-lease')


def test_preparation_has_no_human_or_execution_authority(tmp_path):
    value=proposal(tmp_path)
    assert assert_unapproved(value)
    assert value['max_invocations']==1 and not value['resume_allowed']
    assert value['human_request_reference'] is None and value['exclusive_gpu_token'] is None
    assert value['mandatory_unresolved_launch_blockers']
    assert not (tmp_path/'artifacts').exists()


@pytest.mark.parametrize('name',['human_approved','separate_post_wave4_request','allow_real_sgd','execution_ready'])
@pytest.mark.parametrize('value',[True,1,None])
def test_approval_flags_are_strictly_false(tmp_path,name,value):
    data=proposal(tmp_path);data[name]=value
    with pytest.raises(PermissionError):assert_unapproved(data)


def test_never_create_or_overwrite_existing_experiment(tmp_path):
    output=tmp_path/'artifacts/b2-pilot/B2-W5-TEST-ONLY';output.mkdir(parents=True)
    with pytest.raises(FileExistsError):proposal(tmp_path)


def test_raw_hash_drift_never_rewrites_expectations(tmp_path):
    source=tmp_path/'example.py';source.write_bytes(b'x=1\r\n')
    expected={source:digest(source)};before=copy.deepcopy(expected)
    assert verify_map(expected)==1
    source.write_bytes(b'x=1\n')
    with pytest.raises(ValueError,match='frozen content mismatch'):verify_map(expected)
    assert expected==before


def test_inventory_and_proposals_use_exclusive_immutable_creation(tmp_path):
    target=tmp_path/'UNAPPROVED.json';exclusive_json(target,proposal(tmp_path));before=target.read_bytes()
    with pytest.raises(FileExistsError):exclusive_json(target,{'human_approved':True})
    assert target.read_bytes()==before
    assert digest(target)==hashlib.sha256(before).hexdigest()


def test_recovery_and_budget_cannot_be_smuggled_into_preparation(tmp_path):
    data=proposal(tmp_path);data['automatic_retry']=True
    with pytest.raises(PermissionError):assert_unapproved(data)
    data=proposal(tmp_path);data['max_wall_seconds']=901
    with pytest.raises(ValueError):assert_unapproved(data)


def test_cannot_erase_discovered_execution_blockers(tmp_path):
    data=proposal(tmp_path);data['mandatory_unresolved_launch_blockers']={}
    with pytest.raises(ValueError):assert_unapproved(data)
