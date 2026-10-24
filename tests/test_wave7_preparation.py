"""CPU-only preparation contracts. No training/approval is issued by these tests."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from hift_research.wave6.authorization import REAL_INITIALIZATION, REAL_LIMITS, RESOURCE_LIMITS
from hift_research.wave5_preparation import digest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_wave7.py'
SPEC = importlib.util.spec_from_file_location('prepare_wave7_test_module', SCRIPT)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
GPU = 'GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'


def expiry():
    return (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()


def inputs(root):
    return {'config': root / 'configs/B2-Pilot-W4.json', 'data': {'fixture': '1' * 64},
            'initialization': dict(REAL_INITIALIZATION),
            'initialization_files': {str(Path(p).resolve()): digest(p)
                                     for p in (sys.executable, sys._base_executable)}}


@pytest.fixture
def policy(monkeypatch):
    value = {'name': 'COOPERATIVE_GUARDED_GPU', 'version': 1,
             'specification': {'minimum_launch_gpu_free_bytes': 4 * 1024**3,
                               'minimum_runtime_gpu_free_bytes': 2 * 1024**3},
             'implementation_files_sha256': {'fixture.py': '2' * 64},
             'implementation_sha256': '3' * 64, 'policy_sha256': '4' * 64}
    monkeypatch.setattr(prep, '_policy', lambda root: copy.deepcopy(value))
    return value


def proposal(root, policy):
    return prep.build_proposal(root, root / 'artifacts/wave7-preparation/freeze-001',
                               'W7-EXPERIMENT-001', 'W7-PROPOSAL-001', expiry(), GPU,
                               inputs(root), {'source.py': '0' * 64}, {})


def test_fresh_policy_b_proposal_has_no_training_or_human_authority(tmp_path, policy):
    value = proposal(tmp_path, policy)
    assert value['proposal_status'] == 'UNAPPROVED_HUMAN_ACK_REQUIRED'
    assert value['human_approved'] is False and value['allow_real_sgd'] is False
    assert value['execution_ready'] is False and value['human_request_reference'] is None
    assert value['operational_policy'] == policy
    assert value['limits'] == REAL_LIMITS and value['resources'] == RESOURCE_LIMITS
    assert value['scope']['initialization'] == REAL_INITIALIZATION
    assert value['max_invocations'] == 1
    assert not any(value[k] for k in ('resume_allowed', 'automatic_retry', 'scope_expansion'))
    assert not (tmp_path / 'artifacts').exists()


def test_guarded_command_and_worker_are_exact_and_never_legacy(tmp_path, policy):
    value = proposal(tmp_path, policy)
    argv = value['argv']
    assert argv[1] == str(tmp_path / 'scripts/run_b2_guarded.py')
    assert argv[2:] == ['--proposal', str(tmp_path / 'artifacts/wave7-preparation/freeze-001/UNAPPROVED-PROPOSAL.json'),
                        '--authorization', value['approval_path'], '--output', value['output_path']]
    worker = value['scope']['worker_launch']
    assert worker['argv'] == [argv[0], argv[1], '--_worker', str(Path(value['output_path']) / 'dispatch.json')]
    assert worker['CUDA_VISIBLE_DEVICES'] == GPU
    assert worker['logical_python'] == str(Path(sys.executable).resolve())
    assert worker['application'] == str(Path(sys._base_executable).resolve())
    assert worker['bytecode_cache_policy'] == 'verified_project_source_loader_no_bytecode'
    assert '--ack-real-sgd' not in argv and '--synthetic-test' not in argv


@pytest.mark.parametrize('value', ['2026-01-01T00:00:00+00:00', '2099-01-01T00:00:00',
                                   '2099-01-01T08:00:00+08:00', 'invalid', None])
def test_expiration_must_be_future_and_explicitly_utc(value):
    with pytest.raises(ValueError, match='fresh future explicit UTC'):
        prep.validate_expiry(value)


def test_expiration_is_not_reused_or_silently_changed():
    now = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    assert prep.validate_expiry('2026-10-24T12:01:00Z', now) == '2026-10-24T12:01:00+00:00'
    with pytest.raises(ValueError):
        prep.validate_expiry(now.isoformat(), now)


@pytest.mark.parametrize('identifier', ['../escape', 'CON', 'LPT1', '', 'a/b', 'a.b'])
def test_bad_windows_experiment_identifier_is_rejected(tmp_path, policy, identifier):
    with pytest.raises(PermissionError):
        prep.build_proposal(tmp_path, tmp_path / 'fresh', identifier, 'P1', expiry(), GPU,
                            inputs(tmp_path), {}, {})


@pytest.mark.parametrize('field', ['approval', 'output', 'history'])
def test_existing_approval_output_or_attempt_history_never_reused(tmp_path, policy, field):
    paths = {'approval': tmp_path / 'artifacts/wave7-authorization/W7-PROPOSAL-001.approved.json',
             'output': tmp_path / 'artifacts/b2-pilot/W7-EXPERIMENT-001',
             'history': tmp_path / 'artifacts/wave7-claims/W7-EXPERIMENT-001.time.jsonl'}
    target = paths[field]
    target.parent.mkdir(parents=True)
    target.write_bytes(b'old evidence')
    with pytest.raises(FileExistsError):
        proposal(tmp_path, policy)
    assert target.read_bytes() == b'old evidence'


def source_fixture(root):
    for name in ('src/hift_research/__init__.py', 'scripts/run_b2_guarded.py',
                 'src/hift_research/wave6/authorization.py', 'src/hift_research/wave6/lease.py',
                 'third_party/HiFT/upstream.py', 'scripts/untracked.py',
                 'pyproject.toml', '.gitattributes', '.gitmodules', 'AGENTS.md'):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# fixture\n', encoding='utf-8')


def test_source_freeze_includes_untracked_and_full_upstream_python(tmp_path):
    source_fixture(tmp_path)
    value = prep.source_map(tmp_path)
    assert 'scripts/untracked.py' in value and 'third_party/HiFT/upstream.py' in value
    (tmp_path / 'scripts/untracked.py').write_text('# drift\n', encoding='utf-8')
    assert prep.source_map(tmp_path) != value


def test_native_shadow_module_forbidden_even_when_no_tracked_git_entry(tmp_path):
    source_fixture(tmp_path)
    (tmp_path / 'scripts/shadow.pyd').write_bytes(b'not real executable')
    with pytest.raises(PermissionError, match='native/sourceless'):
        prep.source_map(tmp_path)


def test_prior_evidence_snapshot_excludes_wave7_and_detects_removal(tmp_path):
    old = tmp_path / 'artifacts/wave5-preparation/old.json'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'unchanged\r\n')
    new = tmp_path / 'artifacts/wave7-preparation/new.json'
    new.parent.mkdir(parents=True)
    new.write_bytes(b'new')
    before = prep.prior_evidence_map(tmp_path)
    assert before == {str(old.resolve()): digest(old)}
    old.write_bytes(b'unchanged\n')
    assert prep.prior_evidence_map(tmp_path) != before


def mocked_preparation(monkeypatch, root, eligible=True):
    from hift_research.wave6 import lease
    data = inputs(root)
    data.update({'assets': {'source_frames': 160, 'accepted_archives': 1151},
                 'initialization_receipt': {'cuda_initialized': False}})
    monkeypatch.setattr(prep, 'verify_preservation', lambda root: {'old_evidence_files_sha256': {}})
    monkeypatch.setattr(prep, 'prior_evidence_map', lambda root: {})
    monkeypatch.setattr(prep, 'collect_inputs', lambda root: data)
    monkeypatch.setattr(prep, 'source_map', lambda root: {'fixture.py': '0' * 64})
    monkeypatch.setattr(prep, 'validate_regression_receipt', lambda root, path, sources: {
        'bindings': {}, 'counts': {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0},
        'command': ['fixture-only-no-real-evidence']})
    monkeypatch.setattr(prep, 'verify_map', lambda value: len(value))
    monkeypatch.setattr(prep, '_git', lambda root, *args, raw=False: b'' if raw else 'fixture')
    monkeypatch.setattr(lease, 'probe_lease_availability', lambda gpu_uuid: {
        'eligible': True, 'authority_granted': False, 'ownership_retained': False, 'fixture_only': True})
    monkeypatch.setattr(lease, 'probe_physical_availability',
                        lambda gpu_uuid, resources, *, policy: {'eligible': eligible, 'fixture_only': True})
    return root / 'artifacts/wave7-preparation/FREEZE-001'


def test_preparation_publishes_exclusive_full_bindings_but_no_authorization(tmp_path, monkeypatch, policy):
    output = mocked_preparation(monkeypatch, tmp_path)
    result = prep.prepare(tmp_path, output, 'W7-EXPERIMENT-001', 'W7-PROPOSAL-001', expiry(), GPU)
    value = json.loads((output / 'UNAPPROVED-PROPOSAL.json').read_text(encoding='utf-8'))
    assert result['authorization_file_created'] is False
    assert result['initial_participating_lease_readiness']['eligible'] is True
    assert result['final_participating_lease_readiness']['authority_granted'] is False
    assert value['final_participating_lease_readiness'] == result['final_participating_lease_readiness']
    assert result['authentic_backward_calls'] == result['authentic_optimizer_steps'] == 0
    assert result['result'] == 'FROZEN_UNAPPROVED_MANDATORY_HUMAN_PAUSE'
    request = json.loads((output / 'HUMAN-REVIEW-REQUEST.json').read_text(encoding='utf-8'))
    assert request['human_ack_received'] is False
    assert request['status'] == 'REQUEST_ONLY_NOT_RECEIVED_NOT_AUTHORIZATION'
    assert request['proposal'] == result['proposal']
    assert request['operational_policy_decision_NOT_received'].startswith('I accept COOPERATIVE_GUARDED_GPU')
    assert request['training_authorization_decision_NOT_received'].startswith('I authorize one authentic corrected-B2')
    assert request['exact_acknowledgment_NOT_received'] == (request['operational_policy_decision_NOT_received']
                                                           + '\n' + request['training_authorization_decision_NOT_received'])
    assert digest(result['human_review_request']['path']) == result['human_review_request']['sha256']
    assert value['data_files_sha256'] == inputs(tmp_path)['data']
    assert value['initialization_files_sha256'] == inputs(tmp_path)['initialization_files']
    for binding in value['bindings'].values():
        if binding['path'] != str(tmp_path / 'configs/B2-Pilot-W4.json'):
            assert digest(binding['path']) == binding['sha256']
    assert not Path(value['approval_path']).exists()
    assert not Path(value['output_path']).exists()
    assert not Path(value['claim_directory']).exists()
    old = (output / 'UNAPPROVED-PROPOSAL.json').read_bytes()
    with pytest.raises(FileExistsError):
        prep.prepare(tmp_path, output, 'W7-EXPERIMENT-001', 'W7-PROPOSAL-001', expiry(), GPU)
    assert (output / 'UNAPPROVED-PROPOSAL.json').read_bytes() == old


def test_blocked_live_snapshot_does_not_publish_a_proposal(tmp_path, monkeypatch, policy):
    output = mocked_preparation(monkeypatch, tmp_path, eligible=False)
    with pytest.raises(PermissionError, match='preflight blocked'):
        prep.prepare(tmp_path, output, 'E1', 'P1', expiry(), GPU)
    assert not output.exists()


@pytest.mark.parametrize('stage', ['initial', 'final'])
def test_participating_lease_readiness_blocks_active_or_stale_holder(tmp_path, monkeypatch, policy, stage):
    from hift_research.wave6 import lease
    output = mocked_preparation(monkeypatch, tmp_path)
    calls = []
    def probe(gpu_uuid):
        calls.append(gpu_uuid)
        blocked = stage == 'initial' or len(calls) > 1
        return {'eligible': not blocked, 'reasons': ['fixture-existing-holder'] if blocked else [],
                'authority_granted': False, 'ownership_retained': False}
    monkeypatch.setattr(lease, 'probe_lease_availability', probe)
    with pytest.raises(PermissionError, match='participating guarded HiFT lease readiness'):
        prep.prepare(tmp_path, output, 'E1', 'P1', expiry(), GPU)
    assert not (output / 'UNAPPROVED-PROPOSAL.json').exists()
    assert not (output / 'HUMAN-REVIEW-REQUEST.json').exists()
    assert len(calls) == (1 if stage == 'initial' else 2)


def test_old_evidence_drift_refuses_to_freeze(tmp_path, monkeypatch, policy):
    output = mocked_preparation(monkeypatch, tmp_path)
    monkeypatch.setattr(prep, 'prior_evidence_map', lambda root: {'changed': '0' * 64})
    with pytest.raises(ValueError, match='old evidence'):
        prep.prepare(tmp_path, output, 'E1', 'P1', expiry(), GPU)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['source', 'policy'])
def test_drift_during_freeze_never_publishes_proposal(tmp_path, monkeypatch, policy, kind):
    output = mocked_preparation(monkeypatch, tmp_path)
    calls = []
    if kind == 'source':
        def changed_source(root):
            calls.append(root)
            return {'fixture.py': ('0' if len(calls) == 1 else '1') * 64}
        monkeypatch.setattr(prep, 'source_map', changed_source)
    else:
        def changed_policy(root):
            calls.append(root)
            value = copy.deepcopy(policy)
            if len(calls) > 1:
                value['policy_sha256'] = '5' * 64
            return value
        monkeypatch.setattr(prep, '_policy', changed_policy)
    with pytest.raises(ValueError, match='policy/source drift'):
        prep.prepare(tmp_path, output, 'E1', 'P1', expiry(), GPU)
    assert not (output / 'UNAPPROVED-PROPOSAL.json').exists()
    assert not (output / 'HUMAN-REVIEW-REQUEST.json').exists()
    failure = json.loads((output / 'failure.json').read_text(encoding='utf-8'))
    assert failure['no_training_launch'] is True and failure['no_approval_created'] is True


def regression_fixture(tmp_path):
    source = tmp_path / 'fixture.py'
    source.write_text('# synthetic source\n', encoding='utf-8')
    sources = {'fixture.py': digest(source)}
    junit = tmp_path / 'regression.xml'
    junit.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>', encoding='utf-8')
    console = tmp_path / 'regression.txt'
    console.write_text('2 passed fixture only\n', encoding='utf-8')
    receipt = tmp_path / 'regression.json'
    value = {'schema': 'hift-wave7-regression-receipt-v1', 'source_files_sha256': sources,
             'command': ['fixture-only-CPU'], 'counts': {'tests': 2, 'failures': 0, 'errors': 0, 'skipped': 0},
             'junit': {'path': str(junit), 'sha256': digest(junit)},
             'console': {'path': str(console), 'sha256': digest(console)}}
    receipt.write_text(json.dumps(value), encoding='utf-8')
    return receipt, value, sources


def test_exact_regression_xml_console_and_source_bytes_bound(tmp_path):
    receipt, value, sources = regression_fixture(tmp_path)
    result = prep.validate_regression_receipt(tmp_path, receipt, sources)
    assert result['counts'] == value['counts']
    assert set(result['bindings']) == {'regression_receipt', 'regression_junit', 'regression_console'}
    assert result['bindings']['regression_receipt']['sha256'] == digest(receipt)


@pytest.mark.parametrize('fault', ['missing', 'source_inventory', 'source_bytes', 'junit_bytes', 'console_bytes',
                                   'count_mismatch', 'failures', 'errors', 'skipped', 'zero_tests', 'no_command'])
def test_regression_invalid_or_stale_receipt_cannot_freeze(tmp_path, fault):
    receipt, value, sources = regression_fixture(tmp_path)
    if fault == 'missing':
        receipt = None
    elif fault == 'source_inventory': value['source_files_sha256'] = {'fixture.py': '0' * 64}
    elif fault == 'source_bytes': (tmp_path / 'fixture.py').write_text('# drift', encoding='utf-8')
    elif fault == 'junit_bytes': (tmp_path / 'regression.xml').write_text('changed', encoding='utf-8')
    elif fault == 'console_bytes': (tmp_path / 'regression.txt').write_text('changed', encoding='utf-8')
    elif fault == 'count_mismatch': value['counts']['tests'] = 3
    elif fault in ('failures', 'errors', 'skipped'): value['counts'][fault] = 1
    elif fault == 'zero_tests': value['counts']['tests'] = 0
    elif fault == 'no_command': value['command'] = []
    if receipt is not None:
        receipt.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        prep.validate_regression_receipt(tmp_path, receipt, sources)


def test_separate_preparation_namespace_required(tmp_path, policy):
    with pytest.raises(ValueError, match='new separate'):
        prep.prepare(tmp_path, tmp_path / 'artifacts/wave5-preparation/REWRITE', 'E1', 'P1', expiry(), GPU)
