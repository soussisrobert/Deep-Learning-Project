"""Entry wiring with explicitly synthetic mocks; never an authentic ACK or workload."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from hift_research.wave6 import entry
from hift_research.wave6.authorization import RESOURCE_LIMITS
from hift_research.wave6.policy import COOPERATIVE_GUARDED_GPU, policy_binding


@pytest.mark.parametrize('fault', [None, 'low_gpu', 'low_host', 'invalid_native'])
def test_guarded_production_branch_policy_wiring_or_fail_before_any_child(monkeypatch, fault):
    # No real approval is constructed. read_acknowledgment itself is replaced by
    # a synthetic object in-process; this fixture cannot pass the actual CLI.
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-entry-') as directory:
        root=Path(directory);output=root/'SYNTHETIC-MOCK-OUTPUT';calls=[]
        p={'limits':{'max_wall_seconds':30},'claim_directory':str(root/'claims'),
           'expires_utc':(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(),
           'resources':dict(RESOURCE_LIMITS),'operational_policy':policy_binding(COOPERATIVE_GUARDED_GPU),
           'device':{'gpu_uuid':'GPU-00000000-0000-0000-0000-000000000077'},'cwd':str(root)}
        proposal=root/'SYNTHETIC-NOT-A-PROPOSAL.json';proposal.write_text(json.dumps(p),encoding='utf-8')
        grant=SimpleNamespace(synthetic_only=False,proposal=p,experiment_id='SYNTHETIC-MOCK-NO-REAL-SGD',
            authorization_digest='a'*64,proposal_sha256='b'*64,gpu_uuid=p['device']['gpu_uuid'],
            output_path=output,proposal_path=proposal,approval_path=root/'NONEXISTENT-NOT-AN-APPROVAL.json')
        monkeypatch.setattr(entry,'read_acknowledgment',lambda *a,**k:grant)
        monkeypatch.setattr(entry,'verify_bound_files',lambda *a,**k:True)
        monkeypatch.setattr(entry,'reverify_sources',lambda *a,**k:True)
        monkeypatch.setattr(entry,'claim_authorization',lambda *a,**k:calls.append('synthetic_claim'))
        monkeypatch.setattr(entry,'assert_claim',lambda *a,**k:True)
        profile={'argv':['synthetic-no-execution'],'application':'synthetic-no-executable','python_prefix':str(root),
                 'venv_launcher':'synthetic-not-python',**{k:'' for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE',
                 'CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','PYTHONOPTIMIZE','PYTHONHOME')}}
        monkeypatch.setattr(entry,'worker_launch_profile',lambda *a,**k:profile)
        def availability(uuid,resources,**kwargs):
            assert kwargs['policy']==p['operational_policy']
            calls.append('native_preflight')
            return {'eligible':fault is None,'reasons':[] if fault is None else [fault], 'VRAM_free_bytes':8*1024**3}
        monkeypatch.setattr(entry,'probe_physical_availability',availability)
        lease=SimpleNamespace(path=root/'synthetic-lease-not-created',token='0'*32,
            assert_owned=lambda:calls.append('owned'),assert_live=lambda:calls.append('live'),release=lambda:calls.append('released'))
        def acquire(*a,**kwargs):
            assert kwargs['policy']==p['operational_policy'] and kwargs['expires_utc']==p['expires_utc']
            calls.append('lease_acquired');return lease
        monkeypatch.setattr(entry,'acquire_lease',acquire)
        from hift_research.wave6 import monitor
        def monitor_factory(job,**kwargs):
            assert kwargs['gpu_uuid']==grant.gpu_uuid
            assert Path(kwargs['heartbeat_path'])==output/'runtime-resource-heartbeat.json'
            assert kwargs['limits']==p['operational_policy']['specification']
            kwargs['authority_check']();calls.append('monitor_wired')
            return 'SYNTHETIC-MONITOR-NOT-RUN'
        monkeypatch.setattr(monitor,'ParentResourceMonitor',monitor_factory)
        def supervised(*a,**kwargs):
            calls.append('supervisor')
            assert kwargs['resource_monitor_factory']('SYNTHETIC-NO-JOB')=='SYNTHETIC-MONITOR-NOT-RUN'
            dispatch=json.loads((output/'dispatch.json').read_text())
            assert dispatch['runtime_heartbeat_path']==str(output/'runtime-resource-heartbeat.json')
            return {'status':'SYNTHETIC_MOCK_ONLY','process_tree_empty_verified':True}
        monkeypatch.setattr(entry,'supervise',supervised)
        monkeypatch.setattr(entry,'validate_worker_receipt',lambda *a:True)
        if fault:
            with pytest.raises(PermissionError,match='availability not established'):
                entry.run_guarded(proposal,grant.approval_path,output,actual_argv=['SYNTHETIC-NOT-CLI'])
            assert 'lease_acquired' not in calls and 'supervisor' not in calls
        else:
            result=entry.run_guarded(proposal,grant.approval_path,output,actual_argv=['SYNTHETIC-NOT-CLI'])
            assert 'monitor_wired' in calls and 'released' in calls
            assert result['lease_released']
        assert not grant.approval_path.exists()


def test_production_lease_readiness_no_authority_or_stealing(monkeypatch,tmp_path):
    from hift_research.wave6 import lease
    from test_wave6_lease import TEST_UUID
    monkeypatch.setattr(lease,'default_lease_directory',lambda:tmp_path)
    result=lease.probe_lease_availability(TEST_UUID)
    assert result['eligible'] and not result['authority_granted'] and not result['ownership_retained']
    path=Path(result['lease_path']);path.write_bytes(b'SYNTHETIC-STALE-OR-PARTIAL-LEASE-NOT-AUTHORITY')
    blocked=lease.probe_lease_availability(TEST_UUID)
    assert not blocked['eligible'] and path.read_bytes()==b'SYNTHETIC-STALE-OR-PARTIAL-LEASE-NOT-AUTHORITY'
