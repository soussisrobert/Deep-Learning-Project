"""CPU authorization regressions; historical evidence is strictly read-only.

Temporary instructions are SIMULATED_UNIT_TEST_ONLY parser inputs, not issued
approvals or grants. No claims, entrypoints, CUDA, or native observations run.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from hift_research.wave6 import authorization as A, direct_cuda as D, ledger, monitor, policy, telemetry


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'artifacts/wave7-preparation/PROPOSAL-001/UNAPPROVED-PROPOSAL.json'
PREVIOUS = ROOT / 'artifacts/direct-cuda-continuation'


@pytest.fixture(autouse=True)
def forbid_execution_and_claims(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('memory-continuation CPU tests must not execute, claim, or probe')
    monkeypatch.setattr(A, 'claim_authorization', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(monitor, 'native_host_resources', forbidden)
    monkeypatch.setattr(monitor, 'native_gpu_resources', forbidden)


def ref(path):
    return {'path': str(path), 'sha256': A.file_digest(path)}


def continuation(tmp_path):
    old = A._read(ORIGINAL)
    previous = A._read(PREVIOUS / 'OWNER-BOUND-PROPOSAL.json')
    p = deepcopy(old)
    now = datetime.now(timezone.utc)
    instruction = {'schema': 'hift-local-owner-instruction-v1', 'role': 'user',
        'authority_kind': 'trusted_local_owner_transport_attestation_not_signature',
        'original_proposal_sha256': A.OWNER_ORIGINAL_SHA256,
        'session_id': 'SIMULATED_UNIT_TEST_ONLY', 'message_id': 'SIMULATED_TEST',
        'text': 'SIMULATED TEST ONLY: independent memory domains and one bounded continuation',
        'received_utc': (now-timedelta(seconds=1)).isoformat(),
        'gpu_coordination': 'none', 'continuation_authorized': True,
        'memory_contract_correction_authorized': True}
    ip = tmp_path / 'simulated-memory-instruction.json'
    ip.write_text(json.dumps(instruction), encoding='utf-8')
    p.update(created_utc=now.isoformat(), expires_utc=(now+timedelta(minutes=20)).isoformat(),
        gpu_coordination='none', memory_contract=D.MEMORY_CONTRACT,
        continuation={'prior_wall_seconds_charged': 14, 'original_total_wall_seconds': 900})
    p['limits']['max_wall_seconds'] = 886
    p['experiment_id'] += '-MDOMAIN-001'
    p['proposal_id'] += '-MDOMAIN-001'
    p['output_path'] += '-mdomain-001'
    p['scope']['worker_launch']['argv'][-1] = str(Path(p['output_path']) / 'dispatch.json')
    p['operational_policy'] = policy.policy_binding(policy.COOPERATIVE_GUARDED_GPU, coordination='none')
    for name in D.MEMORY_SOURCES:
        p['source_files_sha256'][name] = A.file_digest(ROOT / name)
    p['bindings'].update(owner_original_proposal=ref(ORIGINAL), owner_instruction=ref(ip))
    paths = {'proposal': PREVIOUS / 'OWNER-BOUND-PROPOSAL.json',
        'authorization': PREVIOUS / 'OWNER-AUTHORIZATION.json',
        'invocation': PREVIOUS / 'invocation-result.json',
        'ledger': Path(previous['claim_directory']) / (previous['experiment_id']+'.time.jsonl'),
        'claim': Path(previous['claim_directory']) / (A.canonical_digest(previous)+'.claim.json'),
        'worker_journal': Path(previous['output_path']) / 'telemetry.jsonl',
        'resource_journal': Path(previous['output_path']) / 'resource-telemetry.jsonl'}
    for name, path in paths.items():
        p['bindings']['predecessor_'+name] = ref(path)
    a = {'schema': A.OWNER_SCHEMA, 'owner_instruction_sha256': A.file_digest(ip),
         'policy_sha256': p['operational_policy']['policy_sha256']}
    return p, a, now, instruction


def parsed_override(monkeypatch, path, value):
    """Inject a parser result for one immutable file; all other reads stay real."""
    real_read = A._read
    monkeypatch.setattr(A, '_read', lambda candidate: deepcopy(value)
                        if Path(candidate).resolve() == Path(path).resolve() else real_read(candidate))


def test_exact_memory_continuation_is_fresh_and_charges_both_historical_failures(tmp_path):
    p, a, now, instruction = continuation(tmp_path)
    assert A.validate_local_owner(p, a, now) == instruction
    assert D.validate_memory_continuation(p, A._read(ORIGINAL), instruction) == 14
    assert D.validate_continuation(p, A._read(ORIGINAL), instruction) == 14
    assert p['limits'] == dict(A.REAL_LIMITS, max_wall_seconds=886)
    assert p['limits']['max_wall_seconds']+p['continuation']['prior_wall_seconds_charged'] == 900
    previous = A._read(p['bindings']['predecessor_proposal']['path'])
    assert p['experiment_id'] != previous['experiment_id']
    assert p['output_path'] != previous['output_path']
    assert p['automatic_retry'] is False
    assert set(D.MEMORY_SOURCES) <= set(p['source_files_sha256'])


@pytest.mark.parametrize('fault', ['scope', 'dispatch', 'data', 'initialization', 'updates',
    'training_attempts', 'windows', 'wall_reset', 'wall_plus_one', 'debit_zero', 'debit_thirteen',
    'original_budget', 'experiment', 'proposal', 'output', 'old_output', 'claim_directory',
    'device', 'resources', 'retry', 'unrelated_source', 'entry_source', 'policy_source',
    'authorization_source', 'unknown_contract'])
def test_memory_continuation_rejects_scope_assets_budget_identity_and_source_expansion(tmp_path, fault):
    p, a, now, _ = continuation(tmp_path)
    if fault == 'scope': p['scope']['seed'] += 1
    elif fault == 'dispatch': p['scope']['worker_launch']['argv'][-1] = 'elsewhere'
    elif fault == 'data': p['data_files_sha256']['NEW'] = 'a'*64
    elif fault == 'initialization': p['initialization_files_sha256']['NEW'] = 'a'*64
    elif fault == 'updates': p['limits']['max_effective_updates'] += 1
    elif fault == 'training_attempts': p['limits']['max_training_attempts'] += 1
    elif fault == 'windows': p['limits']['max_windows'] += 1
    elif fault == 'wall_reset': p['limits']['max_wall_seconds'] = 900
    elif fault == 'wall_plus_one': p['limits']['max_wall_seconds'] = 887
    elif fault == 'debit_zero': p['continuation']['prior_wall_seconds_charged'] = 0
    elif fault == 'debit_thirteen': p['continuation']['prior_wall_seconds_charged'] = 13
    elif fault == 'original_budget': p['continuation']['original_total_wall_seconds'] = 901
    elif fault == 'experiment': p['experiment_id'] += '-002'
    elif fault == 'proposal': p['proposal_id'] += '-002'
    elif fault == 'output': p['output_path'] += '-002'
    elif fault == 'old_output': p['output_path'] = A._read(ORIGINAL)['output_path']
    elif fault == 'claim_directory': p['claim_directory'] = str(tmp_path)
    elif fault == 'device': p['device']['gpu_uuid'] = 'GPU-OTHER'
    elif fault == 'resources': p['resources']['host_process_peak_rss_bytes'] += 1
    elif fault == 'retry': p['automatic_retry'] = True
    elif fault == 'unrelated_source': p['source_files_sha256']['src/hift_research/training/graph.py'] = 'a'*64
    elif fault in ('entry_source', 'policy_source', 'authorization_source'):
        source = fault.removesuffix('_source')
        p['source_files_sha256']['src/hift_research/wave6/'+source+'.py'] = 'a'*64
    else: p['memory_contract'] = 'cross-total-tolerance'
    with pytest.raises(A.AuthorizationError):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('field', ['gpu_coordination', 'continuation_authorized',
                                  'memory_contract_correction_authorized'])
@pytest.mark.parametrize('value', [False, None, 1, 'true'])
def test_memory_continuation_needs_each_explicit_owner_instruction_field(tmp_path, field, value):
    p, a, now, instruction = continuation(tmp_path)
    instruction[field] = value
    ip = Path(p['bindings']['owner_instruction']['path'])
    ip.write_text(json.dumps(instruction), encoding='utf-8')
    p['bindings']['owner_instruction'] = ref(ip)
    a['owner_instruction_sha256'] = A.file_digest(ip)
    with pytest.raises(A.AuthorizationError, match='explicit owner memory-contract'):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('name', ['proposal', 'authorization', 'claim', 'ledger', 'invocation',
                                  'worker_journal', 'resource_journal'])
def test_each_historical_predecessor_binding_is_hash_checked(tmp_path, name):
    p, a, now, _ = continuation(tmp_path)
    p['bindings']['predecessor_'+name]['sha256'] = 'a'*64
    with pytest.raises(A.AuthorizationError, match='memory predecessor evidence changed: '+name):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('name', ['claim', 'ledger', 'worker_journal', 'resource_journal'])
def test_rehashed_identical_evidence_copies_cannot_change_historical_namespace(tmp_path, name):
    p, a, now, _ = continuation(tmp_path)
    original = Path(p['bindings']['predecessor_'+name]['path'])
    copy = tmp_path / ('copied-'+original.name)
    copy.write_bytes(original.read_bytes())
    p['bindings']['predecessor_'+name] = ref(copy)
    with pytest.raises(A.AuthorizationError, match='memory predecessor namespace mismatch'):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('fault', ['unfinished', 'wrong_status', 'claim_unconsumed', 'budget_breached',
    'missing_receipt', 'different_failure', 'committed_update', 'checkpoint', 'possible_mutation',
    'tree_not_empty', 'job_not_closed'])
def test_forged_or_still_training_predecessor_receipts_are_rejected(tmp_path, monkeypatch, fault):
    p, a, now, _ = continuation(tmp_path)
    path = p['bindings']['predecessor_ledger']['path']
    rows = deepcopy(ledger.read_ledger(path))
    final = rows[-1]
    details = final['details']
    receipt = details['worker_receipt']
    if fault == 'unfinished': final['event'] = 'partial'
    elif fault == 'wrong_status': details['status'] = 'COMPLETE'
    elif fault == 'claim_unconsumed': details['claim_consumed'] = False
    elif fault == 'budget_breached': details['timing']['budget_breached'] = True
    elif fault == 'missing_receipt': details['worker_receipt'] = None
    elif fault == 'different_failure': receipt['resource_monitor']['failure'] = 'another failure'
    elif fault == 'committed_update': receipt['last_committed_update'] = 1
    elif fault == 'checkpoint': receipt['last_good_checkpoint'] = 'forged-checkpoint'
    elif fault == 'possible_mutation': receipt['failed_work']['optimizer_mutation_unpublished_possible'] = True
    elif fault == 'tree_not_empty': receipt['process_tree_empty_verified'] = False
    else: receipt['job_closed'] = False
    real_read = ledger.read_ledger
    monkeypatch.setattr(ledger, 'read_ledger', lambda candidate: deepcopy(rows)
                        if Path(candidate).resolve() == Path(path).resolve() else real_read(candidate))
    with pytest.raises(A.AuthorizationError, match='exact contained zero-training'):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('journal,fault', [
    ('worker_journal', 'training'), ('worker_journal', 'dev'), ('worker_journal', 'checkpoint'),
    ('worker_journal', 'empty'), ('worker_journal', 'duplicate_start'), ('worker_journal', 'tail'),
    ('resource_journal', 'empty'), ('resource_journal', 'wrong_failure'), ('resource_journal', 'tail')])
def test_predecessor_journal_must_prove_no_training_dev_or_checkpoint(tmp_path, monkeypatch, journal, fault):
    p, a, now, _ = continuation(tmp_path)
    path = p['bindings']['predecessor_'+journal]['path']
    state = deepcopy(telemetry.read_last_good(path))
    if fault == 'empty': state['events'] = []
    elif fault == 'tail': state['incomplete_tail'] = True
    elif fault == 'wrong_failure': state['events'][-1]['failure'] = 'unrelated resource stop'
    else:
        event = {'training': 'train_update', 'dev': 'DEV', 'checkpoint': 'checkpoint',
                 'duplicate_start': 'worker_start'}[fault]
        state['events'].append({'event': event})
    real_read = telemetry.read_last_good
    monkeypatch.setattr(telemetry, 'read_last_good', lambda candidate: deepcopy(state)
                        if Path(candidate).resolve() == Path(path).resolve() else real_read(candidate))
    with pytest.raises(A.AuthorizationError, match='no training, DEV, or checkpoint events'):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('fault', ['returncode', 'outer_timeout', 'cumulative_breach', 'debit',
    'elapsed_zero', 'elapsed_negative', 'elapsed_bool', 'elapsed_string', 'elapsed_inf', 'elapsed_nan',
    'elapsed_needs_fifteen_seconds'])
def test_predecessor_invocation_cannot_refill_or_evade_original_cumulative_budget(tmp_path, monkeypatch, fault):
    p, a, now, _ = continuation(tmp_path)
    path = p['bindings']['predecessor_invocation']['path']
    invocation = A._read(path)
    if fault == 'returncode': invocation['returncode'] = 0
    elif fault == 'outer_timeout': invocation['outer_timeout_triggered'] = True
    elif fault == 'cumulative_breach': invocation['cumulative_budget_breached'] = True
    elif fault == 'debit': invocation['cumulative_charged_plus_this_outer_seconds'] -= 1
    else:
        invocation['outer_observed_wall_seconds'] = {'elapsed_zero': 0, 'elapsed_negative': -1,
            'elapsed_bool': True, 'elapsed_string': '9.34', 'elapsed_inf': float('inf'),
            'elapsed_nan': float('nan'), 'elapsed_needs_fifteen_seconds': 10.01}[fault]
    parsed_override(monkeypatch, path, invocation)
    with pytest.raises(A.AuthorizationError):
        A.validate_local_owner(p, a, now)


@pytest.mark.parametrize('fault', ['approval_hash', 'claim_digest', 'claim_consumed', 'owner_instruction_hash'])
def test_nested_historical_authority_chain_is_revalidated_without_mutation(tmp_path, monkeypatch, fault):
    p, a, now, _ = continuation(tmp_path)
    if fault == 'approval_hash':
        path = p['bindings']['predecessor_authorization']['path']
        value = A._read(path)
        value['proposal_sha256'] = 'a'*64
    elif fault in ('claim_digest', 'claim_consumed'):
        path = p['bindings']['predecessor_claim']['path']
        value = A._read(path)
        if fault == 'claim_digest': value['authorization_digest'] = 'a'*64
        else: value['consumed'] = False
    else:
        path = p['bindings']['predecessor_proposal']['path']
        value = A._read(path)
        value['bindings']['owner_instruction']['sha256'] = 'a'*64
    parsed_override(monkeypatch, path, value)
    with pytest.raises(A.AuthorizationError):
        A.validate_local_owner(p, a, now)
