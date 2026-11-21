"""Freeze a NEW unapproved Policy B proposal; never launch or grant real SGD.

Run only after regression/preservation review has finished. Expiry is deliberately
required at freeze time, never copied from an older proposal. Output is exclusive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from hift_research.wave5_preparation import (
    CONFIG_SHA, W4_HEAD, authentic_asset_inventory, digest, exclusive_json,
    initialization_identity, read_json, verify_map,
)
from hift_research.wave6.authorization import (
    GPU, PROPOSAL_SCHEMA, REAL_INITIALIZATION, REAL_LIMITS, REAL_SCOPE,
    RESOURCE_LIMITS, assert_exact_project_source_set, canonical_digest,
    expected_ack, expected_policy_decision, expected_training_decision, validate_identifier,
)


def validate_expiry(value, now=None):
    """Only a future explicitly UTC instant is eligible for a new freeze."""
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError('fresh future explicit UTC expiration required') from exc
    if (parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0
            or parsed <= (now or datetime.now(timezone.utc))):
        raise ValueError('fresh future explicit UTC expiration required')
    return parsed.isoformat()


def _git(root, *args, raw=False):
    value = subprocess.check_output(['git', *args], cwd=root)
    return value if raw else value.decode('utf-8').strip()


def source_map(root):
    """Include untracked Python too: a tracked-only freeze is not source closure."""
    root = Path(root).resolve()
    paths = set()
    for name in ('src', 'scripts', 'third_party/HiFT', 'tests', 'configs', 'requirements'):
        directory = root / name
        if directory.is_symlink() or getattr(directory, 'is_junction', lambda: False)():
            raise ValueError('source root alias forbidden')
        for path in directory.rglob('*'):
            if '__pycache__' in path.parts:
                continue
            if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                raise ValueError('source alias forbidden')
            if path.is_file() and path.suffix.lower() in ('.py', '.json', '.yaml', '.yml', '.toml', '.txt', '.md'):
                paths.add(path)
    for name in ('pyproject.toml', '.gitattributes', '.gitmodules', 'AGENTS.md'):
        paths.add(root / name)
    result = {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}
    assert_exact_project_source_set(root, result)
    return result


def prior_evidence_map(root):
    """Actual-byte snapshot of old evidence; no expectations are ever rewritten."""
    root = Path(root)
    paths = set()
    for directory in (root / 'artifacts').iterdir():
        if directory.is_dir() and directory.name.startswith(('wave4-', 'wave5-', 'wave6-')):
            paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def verify_preservation(root):
    from hift_research.wave4_integrity import verify_prior
    root = Path(root)
    prior = verify_prior(root)
    subprocess.run(['git', 'merge-base', '--is-ancestor', W4_HEAD, 'HEAD'], cwd=root, check=True)
    binding = read_json(root / 'artifacts/wave4-lifecycle/final-config-binding-001.json')
    verify_map({root / p: sha for p, sha in binding['critical_source_sha256'].items()})
    wave5_path = root / 'artifacts/wave5-preparation/PROPOSAL-001/inventory.json'
    wave5 = read_json(wave5_path)
    verify_map({root / p: sha for p, sha in wave5['source_sha256'].items()})
    verify_map({ref['path']: ref['sha256'] for ref in wave5['references'].values()})
    verify_map(wave5['authentic_assets']['file_sha256'])
    return {'prior': prior, 'wave4_critical_files': len(binding['critical_source_sha256']),
            'wave5_inventory_sha256': digest(wave5_path),
            'wave5_sources_unchanged': len(wave5['source_sha256']),
            'old_evidence_files_sha256': prior_evidence_map(root)}


def collect_inputs(root):
    """Verify frozen references, authentic bytes, and deterministic CPU-only init."""
    root = Path(root)
    config = root / 'configs/B2-Pilot-W4.json'
    verify_map({config: CONFIG_SHA})
    cfg = read_json(config)
    data = {}
    def bind(path, sha):
        path = Path(path)
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        if str(path) in data and data[str(path)] != sha:
            raise ValueError('conflicting frozen data hash')
        data[str(path)] = sha
    for key in ('manifest', 'train_policy', 'development_policy'):
        bind(cfg['data'][key], cfg['data'][key + '_sha256'])
    bind(cfg['development']['cohort_manifest'], cfg['development']['cohort_sha256'])
    for split in ('train', 'development'):
        policy = read_json(root / cfg['data'][split + '_policy'])
        verify_map(policy['source_code_hashes'])
        for key in ('population', 'reservations', 'source_content_screen', 'eligibility_prereg'):
            bind(policy[key + '_path'], policy[key + '_sha256'])
    assets = authentic_asset_inventory(root)
    for name, sha in assets['file_sha256'].items():
        bind(name, sha)
    old = read_json(root / 'artifacts/wave5-preparation/PROPOSAL-001/inventory.json')
    for ref in old['references'].values():
        bind(ref['path'], ref['sha256'])
    verify_map(data)
    initial = initialization_identity(root)
    identity = {key: initial[key] for key in REAL_INITIALIZATION}
    if identity != REAL_INITIALIZATION:
        raise ValueError('initialization differs from frozen corrected B2')
    interpreters = {str(Path(sys.executable).resolve()), str(Path(sys._base_executable).resolve())}
    init_files = {str(root / cfg['source_backbone_path']): cfg['source_backbone_sha256']}
    init_files.update({path: digest(path) for path in interpreters})
    verify_map(init_files)
    return {'config': config, 'data': data, 'assets': assets,
            'initialization': identity, 'initialization_receipt': initial,
            'initialization_files': init_files}


def validate_regression_receipt(root, path, sources):
    """Require actual all-pass JUnit evidence for these exact source bytes."""
    from xml.etree import ElementTree as ET
    if path is None:
        raise ValueError('full exact-source regression receipt required before freeze')
    path = Path(path).resolve()
    value = read_json(path)
    if value.get('schema') != 'hift-wave7-regression-receipt-v1':
        raise ValueError('unrecognized Wave7 regression receipt')
    if value.get('source_files_sha256') != sources:
        raise ValueError('regression evidence does not bind exact current source inventory')
    command = value.get('command')
    if (not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item for item in command)):
        raise ValueError('full actual regression command required')
    expected = value.get('counts')
    if (not isinstance(expected, dict) or set(expected) != {'tests', 'failures', 'errors', 'skipped'}
            or any(type(v) is not int for v in expected.values()) or expected['tests'] <= 0
            or any(expected[k] != 0 for k in ('failures', 'errors', 'skipped'))):
        raise ValueError('full positive regression with zero failures errors skips required')
    bindings = {'regression_receipt': {'path': str(path), 'sha256': digest(path)}}
    for name in ('junit', 'console'):
        reference = value[name]
        target = Path(reference['path'])
        if not target.is_absolute() or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError('nonempty absolute regression artifact required:' + name)
        verify_map({target: reference['sha256']})
        bindings['regression_' + name] = {'path': str(target.resolve()), 'sha256': reference['sha256']}
    junit = ET.parse(bindings['regression_junit']['path']).getroot()
    suites = [junit] if junit.tag == 'testsuite' else list(junit.iter('testsuite'))
    actual = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in expected}
    if actual != expected:
        raise ValueError('regression JUnit totals disagree with receipt')
    verify_map({Path(root) / name: sha for name, sha in sources.items()})
    return {'bindings': bindings, 'counts': actual, 'command': command}


def _policy(root):
    from hift_research.wave6.policy import COOPERATIVE_GUARDED_GPU, policy_binding
    return policy_binding(name=COOPERATIVE_GUARDED_GPU, root=root)


def build_proposal(root, directory, experiment_id, proposal_id, expires_utc, gpu_uuid, inputs, sources, bindings):
    """Construct values only; in particular this never writes an approval file."""
    from hift_research.wave6.entry import worker_launch_profile
    root, directory = Path(root).resolve(), Path(directory).resolve()
    validate_identifier(experiment_id, 'experiment_id')
    validate_identifier(proposal_id, 'proposal_id')
    expires_utc = validate_expiry(expires_utc)
    if not isinstance(gpu_uuid, str) or not GPU.fullmatch(gpu_uuid):
        raise ValueError('full physical GPU UUID required')
    output = root / 'artifacts/b2-pilot' / experiment_id
    approval = root / 'artifacts/wave7-authorization' / (proposal_id + '.approved.json')
    claims = root / 'artifacts/wave7-claims'
    if output.exists() or approval.exists():
        raise FileExistsError('fresh experiment output and absent human approval required')
    if (claims / (experiment_id + '.time.jsonl')).exists():
        raise FileExistsError('experiment already has cumulative attempt history')
    proposal_path = directory / 'UNAPPROVED-PROPOSAL.json'
    argv = [str(Path(sys.executable).resolve()), str(root / 'scripts/run_b2_guarded.py'),
            '--proposal', str(proposal_path), '--authorization', str(approval), '--output', str(output)]
    proposal = {
        'schema': PROPOSAL_SCHEMA, 'proposal_id': proposal_id, 'experiment_id': experiment_id,
        'proposal_status': 'UNAPPROVED_HUMAN_ACK_REQUIRED', 'human_approved': False,
        'allow_real_sgd': False, 'execution_ready': False, 'human_request_reference': None,
        'synthetic_only': False, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'expires_utc': expires_utc, 'cwd': str(root), 'argv': argv,
        'approval_path': str(approval), 'output_path': str(output), 'claim_directory': str(claims),
        'device': {'name': 'cuda:0', 'gpu_uuid': gpu_uuid},
        'max_invocations': 1, 'resume_allowed': False, 'automatic_retry': False, 'scope_expansion': False,
        'limits': dict(REAL_LIMITS), 'resources': dict(RESOURCE_LIMITS),
        'source_files_sha256': sources, 'data_files_sha256': inputs['data'],
        'initialization_files_sha256': inputs['initialization_files'], 'bindings': bindings,
        'scope': {**REAL_SCOPE, 'initialization': inputs['initialization']},
        'operational_policy': _policy(root),
        'approval_rule': 'STOP. Only a separate subsequent exact human ACK may authorize one guarded invocation. This proposal is not approval.',
        'limitations': ['Policy B permits unresolved desktop C+G contexts; it does not prove physical exclusivity.',
                       'No authentic-image backward/SGD measured or performed by preparation.',
                       'Fixed 128 DEV draws are not 128 independent observations; no tracking accuracy or reproduction claim.'],
    }
    grant = SimpleNamespace(synthetic_only=False, gpu_uuid=gpu_uuid, output_path=output, proposal=proposal)
    # Build with the existing authoritative runtime implementation, then validate
    # it again against its own scope contract and both interpreter byte bindings.
    template = SimpleNamespace(synthetic_only=True, gpu_uuid=gpu_uuid, output_path=output, proposal=proposal)
    profile = worker_launch_profile(template, argv)
    profile['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    proposal['scope']['worker_launch'] = profile
    worker_launch_profile(grant, argv)
    return proposal


def _snapshot(directory, name, value):
    path = directory / name
    exclusive_json(path, value)
    return {'path': str(path), 'sha256': digest(path)}


def prepare(root, output, experiment_id, proposal_id, expires_utc, gpu_uuid, *, regression_receipt=None):
    """Read-only checks + exclusively-created proposal artifacts, no dispatch."""
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_identifier(experiment_id, 'experiment_id')
    validate_identifier(proposal_id, 'proposal_id')
    expires_utc = validate_expiry(expires_utc)
    if not isinstance(gpu_uuid, str) or not GPU.fullmatch(gpu_uuid):
        raise ValueError('full physical GPU UUID required')
    allowed = root / 'artifacts/wave7-preparation'
    if allowed not in output.parents:
        raise ValueError('new separate artifacts/wave7-preparation subdirectory required')
    if output.exists():
        raise FileExistsError('immutable preparation directory already exists')
    start = time.perf_counter()
    preservation = verify_preservation(root)
    from hift_research.wave6.lease import probe_lease_availability, probe_physical_availability
    initial_lease = probe_lease_availability(gpu_uuid)
    if initial_lease.get('eligible') is not True:
        raise PermissionError('participating guarded HiFT lease readiness blocked: ' + str(initial_lease))
    inputs = collect_inputs(root)
    sources = source_map(root)
    regression = validate_regression_receipt(root, regression_receipt, sources)
    policy = _policy(root)
    availability = probe_physical_availability(gpu_uuid, RESOURCE_LIMITS, policy=policy)
    if availability.get('eligible') is not True:
        raise PermissionError('Policy B fresh physical/host preflight blocked: ' + str(availability))
    # Check preservation before any new artifact is published.
    if prior_evidence_map(root) != preservation['old_evidence_files_sha256']:
        raise ValueError('old evidence added/removed/changed during preparation')
    verify_map({root / name: sha for name, sha in sources.items()})
    validate_expiry(expires_utc)
    output.mkdir(parents=True, exist_ok=False)
    try:
        bindings = {
            **regression['bindings'],
            'config': {'path': str(inputs['config']), 'sha256': CONFIG_SHA},
            'source_inventory': _snapshot(output, 'source-inventory.json', {'source_files_sha256': sources}),
            'data_manifest': _snapshot(output, 'data-inventory.json', {'data_files_sha256': inputs['data'],
                                      'authentic_assets': inputs['assets']}),
            'initialization_manifest': _snapshot(output, 'initialization-inventory.json', {
                'initialization_files_sha256': inputs['initialization_files'],
                'initialization': inputs['initialization'], 'cpu_only_receipt': inputs['initialization_receipt']}),
        }
        review = {'preservation': preservation, 'physical_availability': availability,
                  'initial_participating_lease_readiness': initial_lease, 'regression': regression,
                  'git_commit': _git(root, 'rev-parse', 'HEAD'), 'git_status': _git(root, 'status', '--short'),
                  'dirty_diff_sha256': hashlib.sha256(_git(root, 'diff', '--binary', 'HEAD', raw=True)).hexdigest(),
                  'environment': {'python': sys.version, 'interpreter': sys.executable,
                      'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'opencv-python', 'pytest')}}}
        bindings['preparation_review'] = _snapshot(output, 'preparation-review.json', review)
        proposal = build_proposal(root, output, experiment_id, proposal_id, expires_utc, gpu_uuid, inputs, sources, bindings)
        if proposal['operational_policy'] != policy or source_map(root) != sources:
            raise ValueError('policy/source drift during freeze; do not issue proposal')
        if prior_evidence_map(root) != preservation['old_evidence_files_sha256']:
            raise ValueError('old evidence drift during freeze; do not issue proposal')
        validate_expiry(expires_utc)
        final_lease = probe_lease_availability(gpu_uuid)
        if final_lease.get('eligible') is not True:
            raise PermissionError('participating guarded HiFT lease readiness changed before publication: ' + str(final_lease))
        proposal['final_participating_lease_readiness'] = final_lease
        proposal_ref = _snapshot(output, 'UNAPPROVED-PROPOSAL.json', proposal)
        human_review = {'status': 'REQUEST_ONLY_NOT_RECEIVED_NOT_AUTHORIZATION',
                        'proposal': proposal_ref, 'human_ack_received': False,
                        'operational_policy_decision_NOT_received': expected_policy_decision(proposal, proposal_ref['sha256']),
                        'training_authorization_decision_NOT_received': expected_training_decision(proposal, proposal_ref['sha256']),
                        'exact_acknowledgment_NOT_received': expected_ack(proposal, proposal_ref['sha256'])}
        request_ref = _snapshot(output, 'HUMAN-REVIEW-REQUEST.json', human_review)
        receipt = {'result': 'FROZEN_UNAPPROVED_MANDATORY_HUMAN_PAUSE', 'proposal': proposal_ref,
                   'human_review_request': request_ref,
                   'initial_participating_lease_readiness': initial_lease,
                   'final_participating_lease_readiness': final_lease,
                   'proposal_digest': canonical_digest(proposal),
                   'future_policy_decision_NOT_received': human_review['operational_policy_decision_NOT_received'],
                   'future_training_decision_NOT_received': human_review['training_authorization_decision_NOT_received'],
                   'future_ack_text_NOT_received': human_review['exact_acknowledgment_NOT_received'],
                   'source_files': len(sources), 'authentic_frames': inputs['assets']['source_frames'],
                   'accepted_archives': inputs['assets']['accepted_archives'],
                   'cuda_initialized': inputs['initialization_receipt']['cuda_initialized'],
                   'authentic_backward_calls': 0, 'authentic_optimizer_steps': 0,
                   'authorization_file_created': False, 'pilot_output_created': False,
                   'elapsed_preparation_seconds': time.perf_counter() - start}
        exclusive_json(output / 'receipt.json', receipt)
        return receipt
    except Exception as exc:
        exclusive_json(output / 'failure.json', {'error': type(exc).__name__ + ': ' + str(exc),
                                                'no_training_launch': True, 'no_approval_created': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--proposal-id', required=True)
    parser.add_argument('--expires-utc', required=True, help='Fresh future explicit UTC instant chosen at actual freeze')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--regression-receipt', required=True, help='All-pass full regression bound to exact frozen sources')
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.output, args.experiment_id, args.proposal_id,
                             args.expires_utc, args.gpu_uuid, regression_receipt=args.regression_receipt), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
