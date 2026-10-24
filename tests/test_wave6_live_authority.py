"""CPU/metadata faults only: no real grants, GPU contexts or optimizer steps."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib,os,time
import pytest
from hift_research.wave6 import authorization,lease,worker

GPU='GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'

def state(tmp_path,synthetic=False):
    grant=SimpleNamespace(synthetic_only=synthetic,gpu_uuid=GPU,authorization_digest='a'*64,
        expires_utc=datetime.now(timezone.utc)+timedelta(minutes=1),proposal={'cwd':str(tmp_path),'data_files_sha256':{}})
    dispatch={'lease_path':str(tmp_path/'metadata-only.lease'),'lease_token':'1'*32,'soft_deadline_monotonic':time.perf_counter()+60}
    return grant,dispatch

def spies(monkeypatch):
    calls=[]
    monkeypatch.setattr(authorization,'assert_claim',lambda grant,**kw:calls.append(('claim',kw)))
    monkeypatch.setattr(lease,'verify_parent_lease',lambda path,**kw:calls.append(('lease',path,kw)))
    monkeypatch.setattr(authorization,'reverify_sources',lambda grant:calls.append(('source',)))
    return calls

def test_live_authority_rechecks_exact_direct_parent_nonce_and_uuid_each_time(tmp_path,monkeypatch):
    grant,dispatch=state(tmp_path);calls=spies(monkeypatch)
    worker._assert_live_authority(grant,dispatch);worker._assert_live_authority(grant,dispatch)
    assert [call[0] for call in calls]==['claim','lease','claim','lease']
    for call in calls:
        kwargs=call[-1];assert kwargs['expected_parent_pid']==os.getppid()
        if call[0]=='lease':
            assert call[1]==dispatch['lease_path']
            assert kwargs==dict(gpu_uuid=GPU,authorization_digest='a'*64,expected_parent_pid=os.getppid(),token='1'*32)

@pytest.mark.parametrize('reason',['missing_claim','owner_PID_reused','claim_digest_changed','claim_consumed_flag_changed'])
def test_invalid_live_claim_stops_before_lease_or_input_checks(tmp_path,monkeypatch,reason):
    grant,dispatch=state(tmp_path);calls=spies(monkeypatch)
    def reject(*args,**kwargs):raise authorization.AuthorizationError(reason)
    monkeypatch.setattr(authorization,'assert_claim',reject)
    with pytest.raises(authorization.AuthorizationError,match=reason):worker._source_check(grant,dispatch)
    assert calls==[]

@pytest.mark.parametrize('reason',['lease_removed','owner_creation_changed','mutex_not_held','token_changed','lease_bytes_changed'])
def test_invalid_live_lease_stops_before_input_or_model_work(tmp_path,monkeypatch,reason):
    grant,dispatch=state(tmp_path);calls=spies(monkeypatch)
    def reject(*args,**kwargs):raise lease.LeaseError(reason)
    monkeypatch.setattr(lease,'verify_parent_lease',reject)
    with pytest.raises(lease.LeaseError,match=reason):worker._source_check(grant,dispatch)
    assert [call[0] for call in calls]==['claim']

def test_initial_valid_lease_is_not_cached_after_later_loss(tmp_path,monkeypatch):
    grant,dispatch=state(tmp_path);calls=spies(monkeypatch);live={'value':True}
    def verify(*args,**kwargs):
        if not live['value']:raise lease.LeaseError('later lease loss')
        calls.append(('lease_ok',))
    monkeypatch.setattr(lease,'verify_parent_lease',verify)
    worker._source_check(grant,dispatch);live['value']=False
    with pytest.raises(lease.LeaseError,match='later lease loss'):worker._source_check(grant,dispatch)
    assert [call[0] for call in calls]==['claim','lease_ok','source','claim']

def test_cpu_synthetic_still_rechecks_claim_but_never_acquires_or_verifies_real_lease(tmp_path,monkeypatch):
    grant,dispatch=state(tmp_path,synthetic=True);calls=spies(monkeypatch)
    monkeypatch.setattr(lease,'verify_parent_lease',lambda *a,**k:pytest.fail('CPU fixture selected actual GPU lease'))
    worker._source_check(grant,dispatch);worker._source_check(grant,dispatch)
    assert [call[0] for call in calls]==['claim','source','claim','source']

@pytest.mark.parametrize('expired',['wall','UTC'])
def test_expired_boundary_does_not_perform_expensive_or_native_checks(tmp_path,monkeypatch,expired):
    grant,dispatch=state(tmp_path);calls=spies(monkeypatch)
    if expired=='wall':dispatch['soft_deadline_monotonic']=time.perf_counter()-1
    else:grant.expires_utc=datetime.now(timezone.utc)-timedelta(seconds=1)
    with pytest.raises(TimeoutError):worker._source_check(grant,dispatch)
    assert calls==[]

def test_live_authority_precedes_source_and_data_rehash_and_data_drift_remains_fatal(tmp_path,monkeypatch):
    grant,dispatch=state(tmp_path);calls=spies(monkeypatch)
    data=tmp_path/'data.json';data.write_bytes(b'approved')
    grant.proposal['data_files_sha256']={str(data):hashlib.sha256(b'approved').hexdigest()}
    worker._source_check(grant,dispatch)
    assert [call[0] for call in calls]==['claim','lease','source']
    data.write_bytes(b'changed')
    with pytest.raises(ValueError,match='live source-data'):worker._source_check(grant,dispatch)
    assert [call[0] for call in calls]==['claim','lease','source']*2
