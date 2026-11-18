"""Policy-B cooperative containment: CPU fixtures only, never real SGD/CUDA."""
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hift_research.wave6 import monitor, worker
from hift_research.wave6.monitor import ParentResourceMonitor, ResourceViolation, WorkerResourceHeartbeat, validate_resources
from hift_research.wave6.policy import COOPERATIVE_SPEC, COOPERATIVE_GUARDED_GPU, policy_binding
from hift_research.wave6.runtime import RuntimeGuard, GuardViolation
from hift_research.wave6.telemetry import DurableJournal, read_last_good
from hift_research.training.lifecycle import apply_epoch_modes, lifecycle_optimizer, EpochSchedule

GPU = 'GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'
GIB = 1024**3
SAFE = {'host_rss_bytes': GIB, 'available_host_bytes': 8*GIB, 'total_host_bytes': 16*GIB, 'total_bytes': 16*GIB,
        'device_free_bytes': 10*GIB, 'device_used_bytes': 6*GIB, 'device_uuid': GPU}
CUDA = {**SAFE, 'allocated_bytes': GIB, 'reserved_bytes': 2*GIB, 'device': 'cpu'}


class Job:
    pid = os.getpid()
    def __init__(self):
        self.killed = threading.Event(); self.exited = False
    def terminate_tree(self):
        self.killed.set()
    def poll(self):
        return 0 if self.exited else None


def parent(tmp_path, **kwargs):
    options = dict(gpu_uuid=GPU, heartbeat_path=tmp_path/'heartbeat.json', journal_path=tmp_path/'parent.jsonl',
                   authority_check=lambda: True, native_probe=lambda: dict(SAFE), interval_seconds=.02,
                   native_query_timeout_seconds=.3, max_sample_age_seconds=.4, startup_grace_seconds=1.)
    options.update(kwargs)
    job = Job()
    return job, ParentResourceMonitor(job, **options)


def heartbeat(resource_monitor, *, phase='cuda', resources=None, stamp=None, **extra):
    value = {'schema': 'hift-wave7-worker-resource-v1', 'worker_pid': resource_monitor.job.pid,
             'gpu_uuid': GPU, 'sampled_monotonic': time.perf_counter() if stamp is None else stamp,
             'phase': phase, 'resources': dict(CUDA) if resources is None else resources, **extra}
    monitor._atomic_heartbeat(resource_monitor.heartbeat_path, value)


@pytest.mark.parametrize('field,value', [('host_rss_bytes', 8*GIB+1), ('available_host_bytes', 4*GIB-1),
    ('device_free_bytes', 2*GIB-1), ('reserved_bytes', 13*GIB), ('allocated_bytes', 3*GIB),
    ('device_uuid', 'GPU-wrong'), ('total_bytes', float('nan')), ('device_free_bytes', True),
    ('device_used_bytes', 17*GIB), ('device_used_bytes', 7*GIB), ('total_host_bytes', 0),
    ('total_host_bytes', 7*GIB), ('reserved_bytes', 2*GIB+.5), ('host_rss_bytes', 1.5),
    ('total_host_bytes', 2**64), ('allocated_bytes', 2**64)])
def test_shared_resource_validator_rejects_malformed_or_limit(field, value):
    row = {**CUDA, field: value}
    with pytest.raises(ResourceViolation):
        validate_resources(row, min_device_free_bytes=2*GIB, expected_gpu_uuid=GPU, require_cuda=True)


def test_resource_equality_caps_and_minima_are_inclusive():
    row = {**CUDA, 'host_rss_bytes': 8*GIB, 'available_host_bytes': 4*GIB,
           'total_bytes': 10*GIB, 'device_free_bytes': 2*GIB, 'device_used_bytes': 8*GIB,
           'reserved_bytes': 8*GIB, 'allocated_bytes': 8*GIB}
    validate_resources(row, min_device_free_bytes=2*GIB, expected_gpu_uuid=GPU, require_cuda=True)


@pytest.mark.parametrize('missing', ['reserved_bytes', 'allocated_bytes', 'total_bytes', 'device_free_bytes',
                                      'device_uuid', 'host_rss_bytes', 'available_host_bytes'])
def test_required_resources_never_substituted_from_unknown(missing):
    row = dict(CUDA); row.pop(missing)
    with pytest.raises(ResourceViolation):
        validate_resources(row, min_device_free_bytes=2*GIB, expected_gpu_uuid=GPU, require_cuda=True)


def test_native_resources_and_authority_checked_before_worker_cuda(tmp_path):
    calls = []
    job, resource_monitor = parent(tmp_path, authority_check=lambda: calls.append('authority'),
        native_probe=lambda: (calls.append('native') or dict(SAFE)))
    resource_monitor.start()
    try:
        assert calls[:2] == ['authority', 'native'] and resource_monitor.samples >= 1
        assert not resource_monitor._cuda_seen and not job.killed.is_set()
        heartbeat(resource_monitor, phase='pre_cuda', resources={})
        resource_monitor._sample()
        assert not resource_monitor._cuda_seen
    finally:
        assert resource_monitor.close()['failure'] is None


def test_frozen_policy_launch_free_stronger_than_runtime(tmp_path):
    job, resource_monitor = parent(tmp_path, limits=COOPERATIVE_SPEC,
        native_probe=lambda: {**SAFE, 'device_free_bytes': 3*GIB, 'device_used_bytes': 13*GIB})
    with pytest.raises(ResourceViolation, match='free-memory'):
        resource_monitor.start()
    assert job.killed.is_set() and resource_monitor.close()['failure']


@pytest.mark.parametrize('fault', ['uuid', 'pid', 'stale', 'future', 'missing_reserved', 'boot_allocator_fabrication',
                                  'phase', 'malformed_json', 'cuda_free_exceeds_total', 'rollback', 'missing_after_cuda'])
def test_heartbeat_identity_staleness_and_phase_fail_closed(tmp_path, fault):
    _, resource_monitor = parent(tmp_path)
    resource_monitor._started = time.perf_counter()-1
    resource_monitor.startup_grace = 2
    if fault == 'uuid': heartbeat(resource_monitor, gpu_uuid='GPU-wrong')
    elif fault == 'pid': heartbeat(resource_monitor, worker_pid=os.getpid()+1)
    elif fault == 'stale': heartbeat(resource_monitor, stamp=time.perf_counter()-1)
    elif fault == 'future': heartbeat(resource_monitor, stamp=time.perf_counter()+1)
    elif fault == 'missing_reserved':
        row = dict(CUDA); row.pop('reserved_bytes'); heartbeat(resource_monitor, resources=row)
    elif fault == 'boot_allocator_fabrication': heartbeat(resource_monitor, phase='pre_cuda', resources={'reserved_bytes': 0})
    elif fault == 'phase': heartbeat(resource_monitor, phase='unknown')
    elif fault == 'malformed_json': resource_monitor.heartbeat_path.write_text('{bad', encoding='utf-8')
    elif fault == 'cuda_free_exceeds_total': heartbeat(resource_monitor, resources={**CUDA, 'total_bytes': 9*GIB})
    elif fault == 'rollback':
        resource_monitor._last_heartbeat = time.perf_counter(); heartbeat(resource_monitor, stamp=time.perf_counter()-.1)
    elif fault == 'missing_after_cuda': resource_monitor._cuda_seen = True
    with pytest.raises((ResourceViolation, ValueError)):
        resource_monitor._sample()
    resource_monitor.close()


def test_boot_grace_not_extended_by_fresh_unknown_markers(tmp_path):
    job, resource_monitor = parent(tmp_path, startup_grace_seconds=.15)
    resource_monitor.start()
    try:
        heartbeat(resource_monitor, phase='pre_cuda', resources={})
        assert job.killed.wait(.7), 'startup deadline must stop without next worker boundary'
        assert 'startup deadline' in resource_monitor.failure
    finally:
        resource_monitor.close()


def test_stale_heartbeat_stops_during_long_operation_without_boundary(tmp_path):
    job, resource_monitor = parent(tmp_path, max_sample_age_seconds=.15)
    resource_monitor.start()
    try:
        heartbeat(resource_monitor); resource_monitor._sample()
        assert job.killed.wait(.7), 'no controller check() call is needed to enforce stale heartbeat'
        assert 'heartbeat stale' in resource_monitor.failure
        with pytest.raises(ResourceViolation): resource_monitor.check()
        assert not resource_monitor.receipt()['failed_state_checkpoint_allowed']
    finally:
        resource_monitor.close()


def test_hung_native_query_killed_independently_before_operation_released(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = [0]
    def probe():
        calls[0] += 1
        if calls[0] > 1:
            entered.set(); release.wait(2)
        return dict(SAFE)
    job, resource_monitor = parent(tmp_path, native_probe=probe, native_query_timeout_seconds=.12)
    resource_monitor.start()
    try:
        assert entered.wait(.7)
        assert job.killed.wait(.7) and not release.is_set()
        assert 'query timeout' in resource_monitor.failure
    finally:
        release.set(); resource_monitor.close()


def test_blocked_journal_cannot_delay_resource_stop(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    job, resource_monitor = parent(tmp_path, max_sample_age_seconds=.15)
    def blocked(*args, **kwargs):
        entered.set(); release.wait(2)
    monkeypatch.setattr(resource_monitor.journal, 'emit', blocked)
    resource_monitor.start()
    try:
        assert entered.wait(.5)
        assert job.killed.wait(.7) and not release.is_set()
        assert 'sample stale' in resource_monitor.failure
    finally:
        release.set(); resource_monitor.close()


def test_live_authority_loss_terminates_before_evidence(tmp_path, monkeypatch):
    job, resource_monitor = parent(tmp_path, authority_check=lambda: False)
    seen = []
    monkeypatch.setattr(resource_monitor.journal, 'emit', lambda *a, **k: seen.append(job.killed.is_set()))
    with pytest.raises(ResourceViolation, match='authority'):
        resource_monitor.start()
    resource_monitor.close()
    assert job.killed.is_set() and resource_monitor.failure
    assert not seen or all(seen)


def test_normal_exited_worker_probe_race_not_labeled_resource_fault(tmp_path):
    job, resource_monitor = parent(tmp_path)
    def gone():
        job.exited = True
        raise OSError('child exited between poll and native probe')
    resource_monitor.native_probe = gone
    resource_monitor._started = time.perf_counter()
    resource_monitor._run()
    assert resource_monitor.failure is None and not job.killed.is_set()
    resource_monitor.close()


@pytest.mark.skipif(os.name != 'nt', reason='Windows bounded sharing contention')
@pytest.mark.parametrize('persistent', [False, True])
def test_windows_heartbeat_replace_contention_is_bounded_and_not_suppressed(tmp_path, monkeypatch, persistent):
    original = monitor.os.replace; calls = []
    def contended(source, target):
        calls.append(True)
        if persistent or len(calls) < 3:
            error = PermissionError('synthetic Windows scanner sharing conflict'); error.winerror = 5
            raise error
        return original(source, target)
    monkeypatch.setattr(monitor.os, 'replace', contended)
    path = tmp_path/'heartbeat.json'; started = time.perf_counter()
    if persistent:
        with pytest.raises(PermissionError): monitor._atomic_heartbeat(path, {'fixture': True})
        assert not path.exists() and time.perf_counter()-started < .5
    else:
        monitor._atomic_heartbeat(path, {'fixture': True})
        assert json.loads(monitor._read_heartbeat_bytes(path)) == {'fixture': True} and len(calls) == 3


def test_worker_pre_cuda_is_unknown_until_activation_and_no_probe_before_cuda(tmp_path):
    path = tmp_path/'heartbeat.json'; calls = []
    resource_monitor = WorkerResourceHeartbeat(path, GPU, lambda: calls.append('authority'),
        lambda: (calls.append('cuda') or dict(CUDA)), interval_seconds=.5)
    resource_monitor.start()
    try:
        row = json.loads(monitor._read_heartbeat_bytes(path))
        assert row['phase'] == 'pre_cuda' and row['resources'] == {} and 'cuda' not in calls
        resource_monitor.activate()
        row = json.loads(monitor._read_heartbeat_bytes(path))
        assert row['phase'] == 'cuda' and row['resources']['reserved_bytes'] == CUDA['reserved_bytes']
        assert row['worker_pid'] == os.getpid() and row['gpu_uuid'] == GPU and 'cuda' in calls
    finally:
        resource_monitor.close()


def test_runtime_boundary_initial_uuid_and_free_are_mandatory_without_backward():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = nn.Linear(2, 2); self.grader = nn.Linear(2, 1)
    model = apply_epoch_modes(Tiny(), 0); optimizer = lifecycle_optimizer(model); EpochSchedule(optimizer).set_epoch(0)
    probe = dict(CUDA)
    guard = RuntimeGuard(model, optimizer, resource_probe=lambda: dict(probe), synthetic=True,
        min_device_free_bytes=2*GIB, expected_gpu_uuid=GPU, max_host_rss_bytes=8*GIB, min_available_host_bytes=4*GIB)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for fault in ({'device_free_bytes': 2*GIB-1}, {'device_uuid': 'GPU-wrong'}):
        probe.update(fault)
        with pytest.raises(GuardViolation): guard.check('pre_step')
        probe.update(CUDA)
    assert all(torch.equal(before[n], value) for n, value in model.state_dict().items())
    assert not optimizer.state and all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('fault_stage', ['checkpoint', 'checkpoint_published', None])
def test_cooperative_checkpoint_failure_never_publishes_failed_bytes(tmp_path, monkeypatch, fault_stage):
    from hift_research.wave6.adapter import GuardedController
    from hift_research.training.wave4_pilot_engine import SerialPilotController
    controller = GuardedController.__new__(GuardedController)
    controller._cooperative = True; controller.tainted = False; controller.fault = None
    controller.update = controller.cursor = controller.last_published_update = controller.last_published_cursor = 0
    controller.last_checkpoint = None; controller.journal = DurableJournal(tmp_path/'checkpoint-events.jsonl')
    def check(stage):
        if stage == fault_stage:
            raise GuardViolation('synthetic resource failure:' + stage)
    controller.guard = SimpleNamespace(check=check)
    def save(self, path):
        Path(path).write_bytes(b'SYNTHETIC_TEST_BYTES_NO_MODEL_STATE')
        return {'path': str(Path(path).resolve()), 'sha256': 'test-only'}
    monkeypatch.setattr(SerialPilotController, 'save_boundary', save)
    target = tmp_path/'approved.pt'
    if fault_stage:
        with pytest.raises(GuardViolation): controller.save_boundary(target)
        assert controller.tainted and not target.exists() and controller.last_checkpoint is None
        assert not [r for r in read_last_good(controller.journal.path)['events'] if r['event'] == 'checkpoint']
        with pytest.raises(RuntimeError, match='tainted'): controller.save_boundary(target)
        assert target.with_name(target.name+'.unvalidated').exists() == (fault_stage == 'checkpoint_published')
    else:
        receipt = controller.save_boundary(target)
        assert Path(receipt['path']) == target and target.exists()
        assert not target.with_name(target.name+'.unvalidated').exists()


@pytest.mark.skipif(os.name != 'nt', reason='reviewed native Windows host GPU memory API')
def test_native_reviewed_gpu_memory_and_uuid_readback_without_cuda():
    # This research host is explicitly pinned to the reviewed physical UUID.
    uuid = GPU
    row = monitor.native_gpu_resources(uuid)
    assert row['device_uuid'].lower() == uuid.lower()
    assert all(type(row[key]) is int and row[key] >= 0
               for key in ('total_bytes', 'device_free_bytes', 'device_used_bytes'))
    assert row['total_bytes'] > 0 and row['device_free_bytes']+row['device_used_bytes'] <= row['total_bytes']


@pytest.mark.skipif(os.name != 'nt', reason='native Windows host API')
def test_native_host_probe_actual_pid_is_numeric_without_cuda():
    row = monitor.native_host_resources(os.getpid())
    assert type(row['host_rss_bytes']) is int and row['host_rss_bytes'] > 0
    assert type(row['available_host_bytes']) is int and row['available_host_bytes'] > 0


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows Job descendant containment')
@pytest.mark.parametrize('case', ['healthy', 'rss_limit', 'malformed_native', 'hung_native', 'stale_heartbeat'])
def test_actual_supervisor_resource_monitor_stops_native_cpu_tree(tmp_path, case):
    import sys
    from test_wave6_supervision import reserve, run
    from hift_research.wave6.windows_job import is_process_running
    output = tmp_path/'output'; heartbeat_path = output/'runtime-heartbeat.json'
    descendant_pid = tmp_path/'descendant.pid'; release = threading.Event()
    descendant_staging = descendant_pid.with_suffix('.pending')
    descendant = ("import os,time; from pathlib import Path; "
                  f"Path({str(descendant_staging)!r}).write_text(str(os.getpid())); "
                  f"os.replace({str(descendant_staging)!r},{str(descendant_pid)!r}); "
                  f"time.sleep({.25 if case == 'healthy' else 60})")
    script = ("import os,subprocess,sys,time\nfrom pathlib import Path\n"
              "from hift_research.wave6.monitor import _atomic_heartbeat\n"
              "from hift_research.wave6.telemetry import DurableJournal\n"
              f"j=DurableJournal({str(output/'telemetry.jsonl')!r}); j.emit('phase_begin',phase='synthetic_cpu_blocked_no_model')\n"
              f"child=subprocess.Popen([sys._base_executable,'-c',{descendant!r}])\n"
              f"while not Path({str(descendant_pid)!r}).exists():time.sleep(.005)\n"
              "started=time.perf_counter()\n"
              "while True:\n"
              f" _atomic_heartbeat({str(heartbeat_path)!r},{{'schema':'hift-wave7-worker-resource-v1','worker_pid':os.getpid(),"
              f"'gpu_uuid':{GPU!r},'sampled_monotonic':time.perf_counter(),'phase':'cuda','resources':{CUDA!r}}})\n"
              + (" child.wait(); break\n" if case == 'healthy' else
                 " time.sleep(60)\n" if case == 'stale_heartbeat' else " time.sleep(.05)\n"))
    # The native Job/process tree is real. GPU/resource observations are explicitly
    # counterfactual fixtures: no CUDA/NVML/model/backward is invoked by this test.
    def probe():
        if descendant_pid.exists():
            if case == 'rss_limit': return {**SAFE, 'host_rss_bytes': 8*GIB+1}
            if case == 'malformed_native': return {**SAFE, 'device_free_bytes': 'untrusted'}
            if case == 'hung_native': release.wait(3)
        return dict(SAFE)
    resources = []
    def factory(job):
        result = ParentResourceMonitor(job, gpu_uuid=GPU, heartbeat_path=heartbeat_path,
            journal_path=output/'parent-resources.jsonl', authority_check=lambda: True, native_probe=probe,
            interval_seconds=.05, native_query_timeout_seconds=.2, max_sample_age_seconds=.4,
            startup_grace_seconds=2.)
        resources.append(result)
        return result
    ledger = reserve(tmp_path, seconds=6., margin=1., watchdog=False)
    try:
        receipt = run(tmp_path, script, ledger, resource_monitor_factory=factory)
    finally:
        release.set()
    assert receipt['process_tree_empty_verified'] and receipt['job_closed']
    assert receipt['child_resumed'] and receipt['peak_active_processes'] >= 2
    assert descendant_pid.exists() and not is_process_running(int(descendant_pid.read_text()))
    assert receipt['last_committed'] is None and receipt['last_checkpoint'] is None
    assert not list(output.glob('*.pt'))
    if case == 'healthy':
        assert receipt['status'] == 'COMPLETED', receipt
        assert receipt['resource_monitor']['failure'] is None
    else:
        assert receipt['status'] == 'STOP_RESOURCE_POLICY', receipt
        assert receipt['resource_monitor']['failure'] and receipt['resource_monitor']['forced_tree_termination']
        assert not receipt['resource_monitor']['failed_state_checkpoint_allowed']
    assert not receipt['timing']['budget_breached']
    ledger.finish(receipt['status'])


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows Job CLI containment')
@pytest.mark.parametrize('case', ['success', 'missing_free', 'low_free', 'wrong_uuid'])
def test_actual_policy_b_cli_cpu_counterfactual_resources_only(case):
    import tempfile
    from test_wave6_integration import cli_bundle, run_cli
    from test_wave7_policy import sign_policy_bundle
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-cli-') as directory:
        bundle = cli_bundle(directory)
        p = bundle['proposal']; uuid = p['device']['gpu_uuid']
        fixture = {**CUDA, 'device_uuid': uuid}
        if case == 'missing_free': fixture.pop('device_free_bytes')
        elif case == 'low_free': fixture['device_free_bytes'] = 2*GIB-1
        elif case == 'wrong_uuid': fixture['device_uuid'] = 'GPU-wrong'
        p['scope']['synthetic_resource_probe'] = fixture
        sign_policy_bundle(bundle)
        result, stdout, stderr = run_cli(bundle)
        output = Path(p['output_path'])
        if case == 'success':
            assert result.returncode == 0, (stderr.read_text(), (output/'child.stderr.log').read_text())
            receipt = json.loads(stdout.read_text())
            assert receipt['status'] == 'COMPLETED' and receipt['last_committed_update'] == 1
            assert receipt['process_tree_empty_verified'] and receipt['checkpoint_bytes_revalidated']
            assert not list(output.glob('*.unvalidated'))
        else:
            assert result.returncode != 0
            rows = read_last_good(output/'telemetry.jsonl')['events']
            assert not [row for row in rows if row['event'] in ('update_commit', 'checkpoint')]
            assert not list(output.glob('*.pt'))


@pytest.mark.skipif(os.name != 'nt', reason='actual Windows Job CPU-only CLI containment')
@pytest.mark.parametrize('case', ['nonfinite_gradient', 'deadline'])
def test_actual_policy_b_cli_nonfinite_and_deadline_never_commit(case):
    import tempfile
    from test_wave6_integration import cli_bundle, run_cli
    from test_wave7_policy import sign_policy_bundle
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-wave7-cli-fault-') as directory:
        bundle = cli_bundle(directory, fault=case == 'deadline')
        p = bundle['proposal']
        p['scope']['synthetic_resource_probe'] = {**CUDA, 'device_uuid': p['device']['gpu_uuid']}
        if case == 'nonfinite_gradient':
            p['scope'].update(fault_stage='after_backward', fault_kind='nonfinite_gradient')
        sign_policy_bundle(bundle)
        result, stdout, stderr = run_cli(bundle)
        assert result.returncode != 0
        output = Path(p['output_path'])
        receipt = json.loads((output/'supervisor_receipt.json').read_text())
        assert receipt['process_tree_empty_verified'] and receipt['job_closed']
        assert not receipt['last_committed'] and receipt['last_checkpoint']['update'] == 0
        assert receipt['status'] != 'COMPLETED' and not receipt['timing']['budget_breached']
        rows = read_last_good(output/'telemetry.jsonl')['events']
        assert not [row for row in rows if row['event'] in ('update_commit', 'optimizer_step_begin')]
        assert len([row for row in rows if row['event'] == 'checkpoint']) == 1
        assert len(list(output.glob('*.pt'))) == 1  # only the valid initial state
        assert receipt['last_checkpoint']['path'].endswith('update-0000.pt')


def test_worker_cooperative_lease_binds_exact_policy_and_expiry(tmp_path, monkeypatch):
    from hift_research.wave6 import authorization, lease
    binding = policy_binding(COOPERATIVE_GUARDED_GPU)
    grant = SimpleNamespace(synthetic_only=False, gpu_uuid=GPU, authorization_digest='a'*64,
        proposal={'operational_policy': binding, 'expires_utc': '2026-10-24T23:00:00Z'})
    dispatch = {'lease_path': str(tmp_path/'lease'), 'lease_token': 'b'*32}
    calls = []
    monkeypatch.setattr(authorization, 'assert_claim', lambda *args, **kwargs: None)
    monkeypatch.setattr(lease, 'verify_parent_lease', lambda *args, **kwargs: calls.append(kwargs))
    worker._assert_live_authority(grant, dispatch)
    assert calls[0]['policy'] == binding and calls[0]['expires_utc'] == grant.proposal['expires_utc']
    assert calls[0]['expected_parent_pid'] == os.getppid()
