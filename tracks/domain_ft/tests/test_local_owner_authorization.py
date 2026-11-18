"""CPU-only transport-validator tests; no grants, claims, GPU, or execution."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
from hift_research.wave6 import authorization as auth
from hift_research.wave6.policy import COOPERATIVE_GUARDED_GPU, policy_binding

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'artifacts/wave7-preparation/PROPOSAL-001/UNAPPROVED-PROPOSAL.json'


def fixture(tmp_path):
    old = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    p = deepcopy(old)
    now = datetime.now(timezone.utc)
    instruction = {'schema': 'hift-local-owner-instruction-v1', 'role': 'user',
        'authority_kind': 'trusted_local_owner_transport_attestation_not_signature',
        'original_proposal_sha256': auth.OWNER_ORIGINAL_SHA256,
        'session_id': 'UNIT_TEST_NOT_REAL_AUTHORITY', 'message_id': 'UNIT_TEST',
        'text': 'SIMULATED TEST instruction, never a production authorization',
        'received_utc': (now - timedelta(seconds=1)).isoformat()}
    ip = tmp_path / 'simulated-instruction.json'
    ip.write_text(json.dumps(instruction), encoding='utf-8')
    p['created_utc'] = now.isoformat()
    p['expires_utc'] = (now + timedelta(minutes=20)).isoformat()
    p['operational_policy'] = policy_binding(COOPERATIVE_GUARDED_GPU)
    p['source_files_sha256']['src/hift_research/wave6/authorization.py'] = auth.file_digest(Path(auth.__file__))
    p['bindings']['owner_original_proposal'] = {'path': str(ORIGINAL), 'sha256': auth.OWNER_ORIGINAL_SHA256}
    p['bindings']['owner_instruction'] = {'path': str(ip), 'sha256': auth.file_digest(ip)}
    a = {'schema': auth.OWNER_SCHEMA, 'owner_instruction_sha256': auth.file_digest(ip),
         'policy_sha256': p['operational_policy']['policy_sha256']}
    return p, a, now


def test_local_owner_transport_rebind_without_legacy_ack(tmp_path):
    p, a, now = fixture(tmp_path)
    assert 'human_ack' not in a
    assert auth.validate_local_owner(p, a, now)['message_id'] == 'UNIT_TEST'


@pytest.mark.parametrize('key', ['limits', 'resources', 'scope', 'device', 'experiment_id',
    'output_path', 'claim_directory', 'data_files_sha256', 'initialization_files_sha256',
    'automatic_retry', 'resume_allowed', 'max_invocations'])
def test_local_owner_cannot_change_frozen_scope(tmp_path, key):
    p, a, now = fixture(tmp_path)
    p[key] = 'changed'
    with pytest.raises(auth.AuthorizationError):
        auth.validate_local_owner(p, a, now)


def test_local_owner_cannot_change_training_source(tmp_path):
    p, a, now = fixture(tmp_path)
    p['source_files_sha256']['src/hift_research/wave6/worker.py'] = 'a' * 64
    with pytest.raises(auth.AuthorizationError, match='non-administrative'):
        auth.validate_local_owner(p, a, now)


@pytest.mark.parametrize('fault', ['expired', 'future_owner', 'instruction_bytes', 'instruction_hash',
    'policy_hash', 'original_hash', 'synthetic', 'policy'])
def test_local_owner_integrity_and_time_fail_closed(tmp_path, fault):
    p, a, now = fixture(tmp_path)
    if fault == 'expired':
        p['expires_utc'] = (now - timedelta(seconds=1)).isoformat()
    elif fault == 'future_owner':
        p['created_utc'] = (now - timedelta(seconds=5)).isoformat()
    elif fault == 'instruction_bytes':
        Path(p['bindings']['owner_instruction']['path']).write_text('{}', encoding='utf-8')
    elif fault == 'instruction_hash':
        a['owner_instruction_sha256'] = 'a' * 64
    elif fault == 'policy_hash':
        a['policy_sha256'] = 'a' * 64
    elif fault == 'original_hash':
        p['bindings']['owner_original_proposal']['sha256'] = 'a' * 64
    elif fault == 'synthetic':
        p['synthetic_only'] = True
    elif fault == 'policy':
        p['operational_policy']['name'] = 'STRICT_PHYSICAL_EXCLUSIVITY'
    with pytest.raises(auth.AuthorizationError):
        auth.validate_local_owner(p, a, now)
