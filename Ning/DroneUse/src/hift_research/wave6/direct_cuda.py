"""Owner-requested direct CUDA mode. No GPU lease, mutex or shared namespace.

A fresh one-use attempt follows only the bound pre-worker failure, with its wall
cost charged to the original budget. This is not checkpoint resume or a retry
under the old consumed authorization.
"""
import math
from pathlib import Path


MODE = 'none'
ALLOWED_SOURCES = {
    'src/hift_research/wave6/authorization.py',
    'src/hift_research/wave6/policy.py',
    'src/hift_research/wave6/entry.py',
    'src/hift_research/wave6/worker.py',
    'src/hift_research/wave6/direct_cuda.py',
    'tests/test_local_owner_authorization.py',
    'tests/test_direct_cuda.py',
}


def enabled(grant):
    from .authorization import OWNER_SCHEMA, AuthorizationError
    mode = grant.proposal.get('gpu_coordination')
    if mode is None:
        return False
    if (mode != MODE or grant.synthetic_only
            or grant.approval.get('schema') != OWNER_SCHEMA):
        raise AuthorizationError('direct CUDA requires explicit real local-owner authority')
    return True


def resource_inventory_policy(grant):
    """Adapt ONLY the read-only native inventory evaluator, never lease acquisition.

    Its legacy API accepts a base Policy B binding. Direct CUDA changes only
    coordination text, not any eligibility/resource rule; validate the actual
    direct binding first and supply the identical resource contract to that API.
    """
    from .policy import validate_policy_binding, policy_binding, COOPERATIVE_GUARDED_GPU
    if not enabled(grant):
        return grant.proposal.get('operational_policy')
    validate_policy_binding(grant.proposal)
    return policy_binding(COOPERATIVE_GUARDED_GPU)


def validate_continuation(p, old, instruction):
    from .authorization import (AuthorizationError, REAL_LIMITS, _read, _absolute,
                                _sha, file_digest, canonical_digest)
    from .ledger import read_ledger
    if p.get('memory_contract') is not None:
        return validate_memory_continuation(p, old, instruction)
    if (p.get('gpu_coordination') != MODE or instruction.get('gpu_coordination') != MODE
            or instruction.get('continuation_authorized') is not True):
        raise AuthorizationError('owner must explicitly remove coordination and authorize continuation')
    refs = {}
    for name in ('proposal', 'authorization', 'claim', 'ledger', 'invocation'):
        ref = p['bindings']['predecessor_' + name]
        path = _absolute(ref['path'])
        if file_digest(path) != _sha(ref['sha256']):
            raise AuthorizationError('predecessor evidence changed: ' + name)
        refs[name] = path
    previous = _read(refs['proposal'])
    approval = _read(refs['authorization'])
    claim = _read(refs['claim'])
    invocation = _read(refs['invocation'])
    rows = read_ledger(refs['ledger'])
    initial, final = rows[0], rows[-1]
    digest = file_digest(refs['authorization'])
    details = final.get('details', {})
    if (previous['experiment_id'] != old['experiment_id']
            or previous.get('gpu_coordination') is not None
            or approval.get('proposal_sha256') != file_digest(refs['proposal'])
            or claim.get('authorization_digest') != digest
            or claim.get('proposal_digest') != canonical_digest(previous)
            or claim.get('proposal_sha256') != file_digest(refs['proposal'])
            or claim.get('experiment_id') != old['experiment_id']
            or claim.get('consumed') is not True
            or initial['experiment_id'] != old['experiment_id']
            or initial['authorization_id'] != digest
            or final['event'] != 'entry_finished' or details.get('status') != 'ENTRY_FAILED'
            or details.get('claim_consumed') is not True
            or details.get('worker_receipt') is not None
            or details.get('timing', {}).get('budget_breached') is not False
            or any(r['event'].startswith('child_') or r['event'] == 'supervision_claimed' for r in rows)
            or invocation.get('returncode') != 1):
        raise AuthorizationError('continuation requires the exact consumed pre-worker failure')
    expected_ledger = Path(previous['claim_directory']) / (previous['experiment_id'] + '.time.jsonl')
    expected_claim = Path(previous['claim_directory']) / (canonical_digest(previous) + '.claim.json')
    if refs['ledger'] != expected_ledger.resolve() or refs['claim'] != expected_claim.resolve():
        raise AuthorizationError('predecessor claim/ledger namespace mismatch')
    elapsed = [invocation['outer_observed_wall_seconds'], final['elapsed_seconds'], details['timing']['elapsed_seconds']]
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in elapsed):
        raise AuthorizationError('finite positive predecessor elapsed time required')
    charged = math.ceil(max(elapsed))
    limits = dict(REAL_LIMITS, max_wall_seconds=REAL_LIMITS['max_wall_seconds'] - charged)
    if (p.get('continuation') != {'prior_wall_seconds_charged': charged, 'original_total_wall_seconds': 900}
            or limits['max_wall_seconds'] <= 5 or p['limits'] != limits):
        raise AuthorizationError('continuation must debit predecessor from original cumulative budget')
    identities = {
        'experiment_id': old['experiment_id'] + '-DIRECT-001',
        'proposal_id': old['proposal_id'] + '-DIRECT-001',
        'output_path': old['output_path'] + '-direct-001',
    }
    if any(p[k] != value for k, value in identities.items()):
        raise AuthorizationError('one fixed fresh continuation identity required; never reuse old run')
    from copy import deepcopy
    expected_scope = deepcopy(old['scope'])
    expected_scope['worker_launch']['argv'][-1] = str(Path(p['output_path']) / 'dispatch.json')
    if p['scope'] != expected_scope:
        raise AuthorizationError('only worker dispatch output path may change within scientific scope')
    administrative = {'created_utc', 'expires_utc', 'argv', 'approval_path', 'bindings',
        'source_files_sha256', 'operational_policy', 'approval_rule', 'gpu_coordination',
        'continuation', 'limits', 'scope', *identities}
    if ({k: v for k, v in p.items() if k not in administrative}
            != {k: v for k, v in old.items() if k not in administrative}):
        raise AuthorizationError('direct CUDA continuation changed frozen scientific scope')
    before, after = old['source_files_sha256'], p['source_files_sha256']
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if not changed <= ALLOWED_SOURCES:
        raise AuthorizationError('direct CUDA changed non-administrative executable source')
    return charged


MEMORY_CONTRACT = 'independent-nvml-v1-cuda-v1'
MEMORY_DELTA_SOURCES = {
    'src/hift_research/wave6/monitor.py',
    'src/hift_research/wave6/worker.py',
    'src/hift_research/wave6/direct_cuda.py',
    'tests/test_wave7_runtime.py',
    'tests/test_memory_domains.py',
    'tests/test_memory_continuation.py',
}
MEMORY_SOURCES = ALLOWED_SOURCES | MEMORY_DELTA_SOURCES


def validate_memory_continuation(p, old, instruction):
    """One owner-authorized correction after the exact zero-training mismatch.

    Historical files remain immutable; all elapsed attempts debit the ORIGINAL
    900s budget. This is fresh initialization, not checkpoint resume or refill.
    """
    from copy import deepcopy
    from .authorization import (AuthorizationError, REAL_LIMITS, _read, _absolute,
                                _sha, file_digest, canonical_digest)
    from .ledger import read_ledger
    from .telemetry import read_last_good
    if (p.get('memory_contract') != MEMORY_CONTRACT
            or p.get('gpu_coordination') != MODE
            or instruction.get('gpu_coordination') != MODE
            or instruction.get('continuation_authorized') is not True
            or instruction.get('memory_contract_correction_authorized') is not True):
        raise AuthorizationError('explicit owner memory-contract correction and continuation required')
    refs = {}
    for name in ('proposal', 'authorization', 'claim', 'ledger', 'invocation',
                 'worker_journal', 'resource_journal'):
        ref = p['bindings']['predecessor_' + name]
        path = _absolute(ref['path'])
        if file_digest(path) != _sha(ref['sha256']):
            raise AuthorizationError('memory predecessor evidence changed: ' + name)
        refs[name] = path
    previous, approval = _read(refs['proposal']), _read(refs['authorization'])
    claim, invocation = _read(refs['claim']), _read(refs['invocation'])
    # Revalidate the first consumed failure and its 4s debit through the exact
    # historical direct-continuation envelope, not a newly invented baseline.
    prior_instruction_ref = previous['bindings']['owner_instruction']
    if file_digest(prior_instruction_ref['path']) != _sha(prior_instruction_ref['sha256']):
        raise AuthorizationError('historical owner instruction changed')
    original_charge = validate_continuation(previous, old, _read(prior_instruction_ref['path']))
    rows = read_ledger(refs['ledger'])
    initial, final = rows[0], rows[-1]
    details = final.get('details', {})
    receipt = details.get('worker_receipt') or {}
    digest = file_digest(refs['authorization'])
    failure = 'ResourceViolation:worker/native physical GPU total mismatch'
    if (previous.get('memory_contract') is not None
            or previous.get('gpu_coordination') != MODE
            or approval.get('proposal_sha256') != file_digest(refs['proposal'])
            or claim.get('authorization_digest') != digest
            or claim.get('proposal_digest') != canonical_digest(previous)
            or claim.get('proposal_sha256') != file_digest(refs['proposal'])
            or claim.get('experiment_id') != previous['experiment_id']
            or claim.get('consumed') is not True
            or initial['experiment_id'] != previous['experiment_id']
            or initial['authorization_id'] != digest
            or final['event'] != 'entry_finished'
            or details.get('status') != 'STOP_RESOURCE_POLICY'
            or details.get('claim_consumed') is not True
            or details.get('timing', {}).get('budget_breached') is not False
            or receipt.get('resource_monitor', {}).get('failure') != failure
            or receipt.get('last_committed_update') != 0
            or receipt.get('last_good_checkpoint') is not None
            or receipt.get('failed_work', {}).get('optimizer_mutation_unpublished_possible') is not False
            or receipt.get('process_tree_empty_verified') is not True
            or receipt.get('job_closed') is not True
            or invocation.get('returncode') != 1
            or invocation.get('outer_timeout_triggered') is not False
            or invocation.get('cumulative_budget_breached') is not False):
        raise AuthorizationError('requires exact contained zero-training memory-mismatch failure')
    expected = {
        'ledger': Path(previous['claim_directory']) / (previous['experiment_id'] + '.time.jsonl'),
        'claim': Path(previous['claim_directory']) / (canonical_digest(previous) + '.claim.json'),
        'worker_journal': Path(previous['output_path']) / 'telemetry.jsonl',
        'resource_journal': Path(previous['output_path']) / 'resource-telemetry.jsonl',
    }
    if any(refs[k] != path.resolve() for k, path in expected.items()):
        raise AuthorizationError('memory predecessor namespace mismatch')
    worker_state = read_last_good(refs['worker_journal'])
    resource_state = read_last_good(refs['resource_journal'])
    if (worker_state['incomplete_tail'] or resource_state['incomplete_tail']
            or [r['event'] for r in worker_state['events']] != ['worker_start']
            or not resource_state['events']
            or resource_state['events'][-1].get('failure') != failure):
        raise AuthorizationError('predecessor must contain no training, DEV, or checkpoint events')
    elapsed = [invocation['outer_observed_wall_seconds'], final['elapsed_seconds'],
               details['timing']['elapsed_seconds']]
    if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in elapsed):
        raise AuthorizationError('finite positive memory predecessor elapsed time required')
    charged = original_charge + math.ceil(max(elapsed))
    if (charged != 14
            or invocation.get('cumulative_charged_plus_this_outer_seconds') != original_charge + elapsed[0]
            or p.get('continuation') != {'prior_wall_seconds_charged': charged, 'original_total_wall_seconds': 900}
            or p['limits'] != dict(REAL_LIMITS, max_wall_seconds=900 - charged)):
        raise AuthorizationError('memory continuation must debit both failures to the original 900s budget')
    identities = {'experiment_id': old['experiment_id'] + '-MDOMAIN-001',
                  'proposal_id': old['proposal_id'] + '-MDOMAIN-001',
                  'output_path': old['output_path'] + '-mdomain-001'}
    if any(p[k] != value for k, value in identities.items()):
        raise AuthorizationError('one fixed fresh memory-continuation identity required')
    scope = deepcopy(old['scope'])
    scope['worker_launch']['argv'][-1] = str(Path(p['output_path']) / 'dispatch.json')
    if p['scope'] != scope:
        raise AuthorizationError('memory correction changed scientific scope or dispatch')
    administrative = {'created_utc', 'expires_utc', 'argv', 'approval_path', 'bindings',
        'source_files_sha256', 'operational_policy', 'approval_rule', 'gpu_coordination',
        'continuation', 'limits', 'scope', 'memory_contract', *identities}
    if ({k: v for k, v in p.items() if k not in administrative}
            != {k: v for k, v in old.items() if k not in administrative}):
        raise AuthorizationError('memory continuation changed frozen assets, recipe, or resources')
    before, after = old['source_files_sha256'], p['source_files_sha256']
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if not changed <= MEMORY_SOURCES:
        raise AuthorizationError('memory continuation changed unrelated executable source')
    before = previous['source_files_sha256']
    new_delta = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if not new_delta <= MEMORY_DELTA_SOURCES:
        raise AuthorizationError('new changes exceed the narrow memory-contract correction')
    return charged
