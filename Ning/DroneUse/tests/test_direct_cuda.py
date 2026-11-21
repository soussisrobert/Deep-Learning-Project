"""CPU-only no-coordination tests. No GPU access or production claims/issuance."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from hift_research.wave6 import authorization as A, direct_cuda as D, entry, policy, worker

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'artifacts/wave7-preparation/PROPOSAL-001/UNAPPROVED-PROPOSAL.json'
PREVIOUS = ROOT / 'artifacts/local-owner-pilot'


def ref(path):
    return {'path': str(path), 'sha256': A.file_digest(path)}


def continuation(tmp_path):
    old = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    previous = json.loads((PREVIOUS / 'OWNER-BOUND-PROPOSAL.json').read_text(encoding='utf-8'))
    p = deepcopy(old)
    now = datetime.now(timezone.utc)
    instruction = {'schema': 'hift-local-owner-instruction-v1', 'role': 'user',
        'authority_kind': 'trusted_local_owner_transport_attestation_not_signature',
        'original_proposal_sha256': A.OWNER_ORIGINAL_SHA256,
        'session_id': 'SIMULATED_UNIT_TEST_ONLY', 'message_id': 'SIMULATED_TEST',
        'text': 'SIMULATED TEST ONLY: remove coordination and continue',
        'received_utc': (now - timedelta(seconds=1)).isoformat(),
        'gpu_coordination': 'none', 'continuation_authorized': True}
    ip = tmp_path / 'simulated-instruction.json'
    ip.write_text(json.dumps(instruction), encoding='utf-8')
    p.update(created_utc=now.isoformat(), expires_utc=(now + timedelta(minutes=20)).isoformat(),
        gpu_coordination='none', continuation={'prior_wall_seconds_charged': 4, 'original_total_wall_seconds': 900})
    p['limits']['max_wall_seconds'] = 896
    p['experiment_id'] += '-DIRECT-001'
    p['proposal_id'] += '-DIRECT-001'
    p['output_path'] += '-direct-001'
    p['scope']['worker_launch']['argv'][-1] = str(Path(p['output_path']) / 'dispatch.json')
    p['operational_policy'] = policy.policy_binding(policy.COOPERATIVE_GUARDED_GPU, coordination='none')
    for name in D.ALLOWED_SOURCES:
        p['source_files_sha256'][name] = A.file_digest(ROOT / name)
    p['bindings'].update(owner_original_proposal=ref(ORIGINAL), owner_instruction=ref(ip))
    paths = {'proposal': PREVIOUS / 'OWNER-BOUND-PROPOSAL.json',
        'authorization': PREVIOUS / 'OWNER-AUTHORIZATION.json',
        'invocation': PREVIOUS / 'invocation-result.json',
        'ledger': Path(previous['claim_directory']) / (previous['experiment_id'] + '.time.jsonl'),
        'claim': Path(previous['claim_directory']) / (A.canonical_digest(previous) + '.claim.json')}
    for name, path in paths.items():
        p['bindings']['predecessor_' + name] = ref(path)
    a = {'schema': A.OWNER_SCHEMA, 'owner_instruction_sha256': A.file_digest(ip),
        'policy_sha256': p['operational_policy']['policy_sha256']}
    return p, a, now, instruction


def test_exact_owner_continuation_debits_prior_time(tmp_path):
    p, a, now, _ = continuation(tmp_path)
    assert A.validate_local_owner(p, a, now)['gpu_coordination'] == 'none'
    assert p['limits']['max_wall_seconds'] + p['continuation']['prior_wall_seconds_charged'] == 900


@pytest.mark.parametrize('fault', ['budget_reset', 'debit_zero', 'updates', 'data', 'initialization',
    'scope', 'dispatch_path', 'old_output', 'old_identity', 'claim_directory', 'recipe_source',
    'predecessor_hash', 'predecessor_unfinished', 'predecessor_trained', 'predecessor_invocation_time'])
def test_direct_continuation_rejects_scope_or_evidence_changes(tmp_path, fault):
    p, a, now, instruction = continuation(tmp_path)
    if fault == 'budget_reset': p['limits']['max_wall_seconds'] = 900
    elif fault == 'debit_zero': p['continuation']['prior_wall_seconds_charged'] = 0
    elif fault == 'updates': p['limits']['max_effective_updates'] = 101
    elif fault == 'data': p['data_files_sha256']['NEW'] = 'a' * 64
    elif fault == 'initialization': p['initialization_files_sha256']['NEW'] = 'a' * 64
    elif fault == 'scope': p['scope']['seed'] += 1
    elif fault == 'dispatch_path': p['scope']['worker_launch']['argv'][-1] = 'elsewhere'
    elif fault == 'old_output': p['output_path'] = p['output_path'].removesuffix('-direct-001')
    elif fault == 'old_identity': p['experiment_id'] = p['experiment_id'].removesuffix('-DIRECT-001')
    elif fault == 'claim_directory': p['claim_directory'] = str(tmp_path)
    elif fault == 'recipe_source': p['source_files_sha256']['src/hift_research/training/graph.py'] = 'a' * 64
    elif fault == 'predecessor_hash': p['bindings']['predecessor_claim']['sha256'] = 'a' * 64
    elif fault in ('predecessor_unfinished', 'predecessor_trained'):
        # Mock parser result only; never alter the durable production ledger.
        from hift_research.wave6 import ledger
        from unittest.mock import patch
        rows = deepcopy(ledger.read_ledger(p['bindings']['predecessor_ledger']['path']))
        if fault == 'predecessor_unfinished': rows[-1]['event'] = 'partial'
        else: rows[-1]['details']['worker_receipt'] = {'updates': 1}
        with patch.object(ledger, 'read_ledger', return_value=rows), pytest.raises(A.AuthorizationError):
            D.validate_continuation(p, json.loads(ORIGINAL.read_text(encoding='utf-8')), instruction)
        return
    elif fault == 'predecessor_invocation_time':
        value = {'returncode': 1, 'outer_observed_wall_seconds': 4.1}
        path = tmp_path / 'simulated-invocation.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        p['bindings']['predecessor_invocation'] = ref(path)
    with pytest.raises(A.AuthorizationError):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('schema,synthetic,mode', [(A.ACK_SCHEMA, False, 'none'),
    (A.OWNER_SCHEMA, True, 'none'), (A.OWNER_SCHEMA, False, 'unknown')])
def test_direct_mode_never_implicit_or_synthetic(schema, synthetic, mode):
    with pytest.raises(A.AuthorizationError):
        D.enabled(SimpleNamespace(proposal={'gpu_coordination': mode}, approval={'schema': schema}, synthetic_only=synthetic))


def test_direct_policy_honest_and_numeric_limits_unchanged():
    bound = policy.policy_binding(policy.COOPERATIVE_GUARDED_GPU, coordination='none')
    assert policy.validate_policy_binding({'operational_policy': bound, 'gpu_coordination': 'none'}) == policy.COOPERATIVE_GUARDED_GPU
    spec = bound['specification']
    assert spec['gpu_coordination'] == 'none'
    assert 'identity_and_lease_validation' not in spec
    assert all(spec[k] == v for k, v in policy.COOPERATIVE_SPEC.items() if isinstance(v, (int, float)))
    with pytest.raises(ValueError):
        policy.validate_policy_binding({'operational_policy': bound})


@pytest.mark.parametrize('kind', ['C+G', 'C', 'UNKNOWN', 'G'])
def test_direct_readonly_inventory_adapter_preserves_native_resource_checks(kind):
    from test_wave6_lease import HOST, TEST_UUID, xml_inventory
    from hift_research.wave6.lease import parse_physical_inventory
    p = {'gpu_coordination': 'none', 'operational_policy': policy.policy_binding(policy.COOPERATIVE_GUARDED_GPU, coordination='none')}
    grant = SimpleNamespace(proposal=p, approval={'schema': A.OWNER_SCHEMA}, synthetic_only=False)
    inventory_policy = D.resource_inventory_policy(grant)
    report = parse_physical_inventory(xml_inventory(kind), TEST_UUID, A.RESOURCE_LIMITS, HOST, policy=inventory_policy)
    assert report['eligible'] and report['hardware_wide_exclusivity_proven'] is False
    insufficient = dict(HOST, available_host_bytes=4 * 1024**3 - 1)
    assert not parse_physical_inventory(xml_inventory(kind), TEST_UUID, A.RESOURCE_LIMITS, insufficient, policy=inventory_policy)['eligible']


def test_worker_direct_authority_no_lease(monkeypatch):
    token = 'SIMULATED_TEST_CAPABILITY'
    monkeypatch.setenv('HIFT_W6_CAPABILITY', token)
    calls = []
    monkeypatch.setattr(A, 'assert_claim', lambda grant, **kw: calls.append(kw))
    from hift_research.wave6 import lease
    monkeypatch.setattr(lease, 'verify_parent_lease', lambda *a, **kw: pytest.fail('coordination called'))
    grant = SimpleNamespace(proposal={'gpu_coordination': 'none'}, approval={'schema': A.OWNER_SCHEMA}, synthetic_only=False)
    dispatch = {'gpu_coordination': 'none', 'parent_pid': os.getppid(), 'lease_path': None, 'lease_token': None,
        'capability_sha256': hashlib.sha256(token.encode()).hexdigest()}
    worker._assert_live_authority(grant, dispatch)
    assert calls == [{'expected_parent_pid': os.getppid()}]
    dispatch['capability_sha256'] = 'a' * 64
    with pytest.raises(PermissionError, match='parent capability'):
        worker._assert_live_authority(grant, dispatch)


def test_entry_direct_skips_lease_preserves_monitor_and_final_probe(tmp_path, monkeypatch):
    proposal_path = tmp_path / 'simulated.json'
    proposal_path.write_text(json.dumps({'device': {'gpu_uuid': 'SIMULATED'}}), encoding='utf-8')
    output = tmp_path / 'output'
    p = {'gpu_coordination': 'none', 'operational_policy': policy.policy_binding(policy.COOPERATIVE_GUARDED_GPU, coordination='none'),
        'claim_directory': str(tmp_path / 'claims'), 'limits': {'max_wall_seconds': 20},
        'expires_utc': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        'resources': A.RESOURCE_LIMITS, 'continuation': {'prior_wall_seconds_charged': 4}, 'cwd': str(tmp_path)}
    grant = SimpleNamespace(proposal=p, approval={'schema': A.OWNER_SCHEMA}, synthetic_only=False,
        experiment_id='SIMULATED_DIRECT_TEST', gpu_uuid='SIMULATED', proposal_path=proposal_path,
        approval_path=tmp_path / 'approval.json', authorization_digest='b' * 64, proposal_sha256='c' * 64,
        output_path=output)
    monkeypatch.setattr(entry, 'read_acknowledgment', lambda *a, **kw: grant)
    monkeypatch.setattr(entry, 'verify_bound_files', lambda *a, **kw: None)
    monkeypatch.setattr(entry, 'reverify_sources', lambda *a, **kw: None)
    monkeypatch.setattr(entry, 'claim_authorization', lambda *a, **kw: None)
    monkeypatch.setattr(entry, 'assert_claim', lambda *a, **kw: None)
    monkeypatch.setattr(entry, 'acquire_lease', lambda *a, **kw: pytest.fail('GPU lease acquired'))
    probes = []
    def probe(*a, **kw):
        probes.append(True)
        return {'eligible': True, 'VRAM_free_bytes': 8 * 1024**3, 'reasons': []}
    monkeypatch.setattr(entry, 'probe_physical_availability', probe)
    profile = {k: 'SIMULATED' for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'CUBLAS_WORKSPACE_CONFIG',
        'CUDA_VISIBLE_DEVICES', 'PYTHONOPTIMIZE', 'PYTHONHOME', 'venv_launcher', 'application', 'python_prefix')}
    profile['argv'] = ['SIMULATED']
    monkeypatch.setattr(entry, 'worker_launch_profile', lambda *a: profile)
    from hift_research.wave6 import monitor
    monitors = []
    class FakeMonitor:
        def __init__(self, job, **kw):
            kw['authority_check']()
            monitors.append(kw)
    monkeypatch.setattr(monitor, 'ParentResourceMonitor', FakeMonitor)
    def fake_supervise(*a, **kw):
        kw['resource_monitor_factory'](object())
        dispatch = json.loads((output / 'dispatch.json').read_text(encoding='utf-8'))
        assert dispatch['lease_path'] is None and dispatch['lease_token'] is None
        assert dispatch['gpu_coordination'] == 'none'
        return {'status': 'FAIL_CHILD_EXIT', 'process_tree_empty_verified': True}
    monkeypatch.setattr(entry, 'supervise', fake_supervise)
    result = entry.run_guarded(proposal_path, tmp_path / 'approval.json', output, actual_argv=['SIMULATED'])
    assert len(probes) == 2 and len(monitors) == 1
    assert result['gpu_coordination'] == 'none' and result['lease_released'] is None
    assert result['final_resource_observation']['eligible']
    assert result['ledger_finalization']['status'] == 'FAIL_CHILD_EXIT'
