"""Operational policy tests: synthetic CPU/infrastructure only, never real grants."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pytest

from hift_research.wave6.authorization import (AuthorizationError, RESOURCE_LIMITS, canonical_digest,
    expected_policy_decision, expected_training_decision, file_digest, verify_authorization)
from hift_research.wave6.policy import (COOPERATIVE_GUARDED_GPU, STRICT_PHYSICAL_EXCLUSIVITY,
    COOPERATIVE_SPEC, policy_binding, validate_policy_binding)
from hift_research.wave6.lease import LeaseError, parse_physical_inventory
from test_wave6_authorization import make_synthetic_bundle, resign_synthetic_bundle, _write
from test_wave6_lease import HOST, TEST_UUID, xml_inventory


def sign_policy_bundle(bundle, name=COOPERATIVE_GUARDED_GPU):
    """Only TEMP/random synthetic envelopes; never constructs a real-role grant."""
    p = bundle['proposal']
    assert p['synthetic_only'] is True and p['device']['name'] == 'cpu'
    p['operational_policy'] = policy_binding(name)
    resign_synthetic_bundle(bundle)
    if name == COOPERATIVE_GUARDED_GPU:
        approval = bundle['approval']
        sha = file_digest(bundle['proposal_path'])
        approval['policy_sha256'] = p['operational_policy']['policy_sha256']
        approval['human_ack']['operational_policy_decision'] = expected_policy_decision(p, sha)
        approval['human_ack']['training_authorization_decision'] = expected_training_decision(p, sha)
        _write(bundle['approval_path'], approval)
    return bundle


def verify(bundle):
    return verify_authorization(bundle['proposal_path'], bundle['approval_path'], **bundle['runtime'])


def test_strict_default_and_different_policy_versions_and_hashes():
    assert validate_policy_binding({}) == STRICT_PHYSICAL_EXCLUSIVITY
    a, b = policy_binding(), policy_binding(COOPERATIVE_GUARDED_GPU)
    assert a['version'] != b['version'] and a['policy_sha256'] != b['policy_sha256']
    assert a['implementation_sha256'] == b['implementation_sha256']
    assert validate_policy_binding({'operational_policy': b}) == COOPERATIVE_GUARDED_GPU


@pytest.mark.parametrize('field', ['name', 'version', 'specification', 'implementation_sha256', 'policy_sha256', 'implementation_files_sha256'])
def test_policy_spec_or_implementation_tampering_fails(field):
    p = {'operational_policy': policy_binding(COOPERATIVE_GUARDED_GPU)}
    p['operational_policy'][field] = 'unreviewed'
    with pytest.raises(ValueError):
        validate_policy_binding(p)


def test_canonical_policy_hash_rejects_boolean_integer_alias():
    p={'operational_policy':policy_binding(COOPERATIVE_GUARDED_GPU)}
    p['operational_policy']['specification']['sample_interval_seconds']=True
    with pytest.raises(ValueError,match='hash mismatch'):
        validate_policy_binding(p)


@pytest.mark.parametrize('kind', ['C+G', 'C', 'UNKNOWN', 'G'])
def test_policy_b_permits_external_kind_without_claiming_classification(kind):
    xml = xml_inventory(kind)
    strict = parse_physical_inventory(xml, TEST_UUID, RESOURCE_LIMITS, HOST)
    coop = parse_physical_inventory(xml, TEST_UUID, RESOURCE_LIMITS, HOST, policy=policy_binding(COOPERATIVE_GUARDED_GPU))
    assert strict['eligible'] is (kind == 'G')
    assert coop['eligible']
    assert coop['processes'] == strict['processes']
    assert coop['hardware_wide_exclusivity_proven'] is False
    if kind != 'G':
        assert coop['ownership_assessment'] == 'UNPROVEN_CONTEXT_CLASSIFICATION'
    assert coop['external_context_interpretation'] == 'opaque_external_clients_permitted_not_classified'


@pytest.mark.parametrize('fault', ['gpu_memory', 'host_memory', 'rss', 'wrong_uuid', 'native_counter', 'missing_processes'])
def test_b_classification_change_does_not_bypass_resource_or_integrity_failures(fault):
    xml, host, uuid = xml_inventory('C+G'), dict(HOST), TEST_UUID
    if fault == 'gpu_memory':
        xml = xml.replace('1000 MiB', '13000 MiB').replace('14000 MiB', '3000 MiB')
    elif fault == 'host_memory':
        host['available_host_bytes'] = 4*1024**3-1
    elif fault == 'rss':
        host['process_peak_rss_bytes'] = 8*1024**3+1
    elif fault == 'wrong_uuid':
        uuid = TEST_UUID[:-1] + '7'
    elif fault == 'native_counter':
        host['available_host_bytes'] = True
    else:
        xml = xml.replace('<processes>', '<unavailable>').replace('</processes>', '</unavailable>')
    try:
        report = parse_physical_inventory(xml, uuid, RESOURCE_LIMITS, host, policy=policy_binding(COOPERATIVE_GUARDED_GPU))
    except LeaseError:
        return
    assert not report['eligible'] and report['reasons']
    assert not any('unresolved_or_compute' in reason for reason in report['reasons'])


@pytest.mark.parametrize('missing', ['operational_policy_decision', 'training_authorization_decision', 'policy_sha256'])
def test_two_exact_decisions_are_required_not_one_or_boolean(missing):
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-two-decisions-') as directory:
        b = sign_policy_bundle(make_synthetic_bundle(directory))
        assert verify(b).synthetic_only
        target = b['approval'] if missing == 'policy_sha256' else b['approval']['human_ack']
        target.pop(missing)
        b['approval']['human_approved'] = True
        _write(b['approval_path'], b['approval'])
        with pytest.raises(AuthorizationError, match='two exact human decisions'):
            verify(b)


@pytest.mark.parametrize('source,target', [(STRICT_PHYSICAL_EXCLUSIVITY, COOPERATIVE_GUARDED_GPU),
                                          (COOPERATIVE_GUARDED_GPU, STRICT_PHYSICAL_EXCLUSIVITY)])
def test_grant_for_one_policy_cannot_authorize_other(source, target):
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-cross-policy-') as directory:
        b = sign_policy_bundle(make_synthetic_bundle(directory), source)
        assert verify(b).synthetic_only
        b['proposal']['operational_policy'] = policy_binding(target)
        _write(b['proposal_path'], b['proposal'])
        with pytest.raises(AuthorizationError):
            verify(b)


def test_changed_config_invalidates_cooperative_authorization():
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-config-') as directory:
        b = sign_policy_bundle(make_synthetic_bundle(directory))
        config = Path(b['proposal']['bindings']['config']['path'])
        config.write_text('{"unreviewed":true}', encoding='utf-8')
        with pytest.raises(AuthorizationError, match='changed'):
            verify(b)


@pytest.mark.parametrize('fault', [None, 'wrong_policy', 'wrong_expiry', 'wrong_uuid', 'expired_metadata'])
def test_native_parent_child_policy_expiry_binding(fault):
    from hift_research.wave6.lease import acquire_lease
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-parent-') as directory:
        expiry=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()
        binding=policy_binding(COOPERATIVE_GUARDED_GPU)
        lease=acquire_lease(TEST_UUID,'a'*64,synthetic_namespace=directory,check_availability=False,
                            policy=binding,expires_utc=expiry)
        original=lease.path.read_bytes()
        code="""import os,sys,json
sys.path.insert(0,sys.argv[1])
from hift_research.wave6.lease import verify_parent_lease,LeaseError
from hift_research.wave6.policy import policy_binding,STRICT_PHYSICAL_EXCLUSIVITY
from datetime import datetime,timedelta,timezone
path,uuid,token,expiry,fault=sys.argv[2:]
meta=json.load(open(path));policy=meta['operational_policy']
if fault=='wrong_policy':policy=policy_binding(STRICT_PHYSICAL_EXCLUSIVITY)
if fault=='wrong_expiry':expiry=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat()
if fault=='wrong_uuid':uuid=uuid[:-1]+'7'
if fault=='expired_metadata':
 meta['expires_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
 with open(path,'w') as f:json.dump(meta,f)
try:verify_parent_lease(path,gpu_uuid=uuid,authorization_digest='a'*64,token=token,
 synthetic_only=True,policy=policy,expires_utc=expiry)
except LeaseError:sys.exit(17)
sys.exit(0)
"""
        try:
            result=subprocess.run([sys._base_executable,'-c',code,str(root/'src'),str(lease.path),TEST_UUID,lease.token,expiry,fault or 'none'],
                env={**os.environ,'__PYVENV_LAUNCHER__':sys.executable},timeout=15)
            assert result.returncode==(17 if fault else 0)
        finally:
            lease.path.write_bytes(original)  # restore only this synthetic fixture for owned cleanup
            lease.release()


def test_expired_cooperative_lease_rejected_without_acquisition():
    from hift_research.wave6.lease import acquire_lease
    with pytest.raises(LeaseError, match='expired'):
        acquire_lease(TEST_UUID, 'a'*64, policy=policy_binding(COOPERATIVE_GUARDED_GPU),
                      expires_utc=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat())
