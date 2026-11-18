"""Synthetic CPU integration of CUDA/NVML accounting; no GPU or native probes.

The saved failure is historical evidence, replayed with only PID/time rebinding.
Job doubles establish monitor/supervisor control flow, not native containment.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import time

import pytest

from hift_research.wave6 import monitor, supervisor
from hift_research.wave6.monitor import ParentResourceMonitor, ResourceViolation
from hift_research.wave6.telemetry import read_last_good


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts/b2-pilot/B2-W7-COOPERATIVE-PILOT-001-direct-001'
GPU = 'GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'
GIB = 1024**3
NATIVE = {'host_rss_bytes': GIB, 'available_host_bytes': 8*GIB,
          'total_host_bytes': 16*GIB, 'total_bytes': 16*GIB,
          'device_free_bytes': 10*GIB, 'device_used_bytes': 6*GIB,
          'device_uuid': GPU, 'memory_api': 'nvmlDeviceGetMemoryInfo_v1'}
CUDA = {'host_rss_bytes': GIB, 'available_host_bytes': 8*GIB,
        'total_host_bytes': 16*GIB, 'total_bytes': 15*GIB,
        'device_free_bytes': 10*GIB, 'allocated_bytes': GIB,
        'reserved_bytes': 2*GIB, 'device_uuid': GPU, 'device': 'cuda:0'}


class SyntheticJob:
    """Inert process-tree double. No OS process creation or observation."""
    pid = 424242

    def __init__(self, *args, **kwargs):
        self.killed = threading.Event()
        self.resumed = False
        self.closed = False
        self.trace = []
        self.physical_executable = 'SYNTHETIC_NOT_EXECUTED'
        self.on_resume = None

    def terminate_tree(self):
        self.killed.set()

    def poll(self):
        return 1 if self.killed.is_set() else None

    def resume(self):
        self.resumed = True
        if self.on_resume is not None:
            self.on_resume()

    def accounting(self):
        return {'active_processes': int(not self.killed.is_set())}

    def wait_empty(self, timeout):
        return self.killed.is_set()

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def forbid_native_observations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('synthetic memory-domain tests must never invoke native probes/processes')
    monkeypatch.setattr(monitor, 'native_host_resources', forbidden)
    monkeypatch.setattr(monitor, 'native_gpu_resources', forbidden)
    monkeypatch.setattr(supervisor, 'SuspendedJobProcess', forbidden)
    # Exercise the real heartbeat parser/validator without Windows file-handle APIs.
    monkeypatch.setattr(monitor, '_read_heartbeat_bytes', lambda path: Path(path).read_bytes())


def parent(tmp_path, *, native=None, job=None, **overrides):
    job = job or SyntheticJob()
    options = dict(gpu_uuid=GPU, heartbeat_path=tmp_path/'heartbeat.json',
                   journal_path=tmp_path/'resources.jsonl', authority_check=lambda: True,
                   native_probe=lambda: dict(NATIVE if native is None else native),
                   interval_seconds=.02, native_query_timeout_seconds=.3,
                   max_sample_age_seconds=1., startup_grace_seconds=2.)
    options.update(overrides)
    return ParentResourceMonitor(job, **options)


def write_heartbeat(resource_monitor, resources=None, **overrides):
    stamp = time.perf_counter()
    row = {'schema': 'hift-wave7-worker-resource-v1', 'worker_pid': resource_monitor.job.pid,
           'gpu_uuid': GPU, 'phase': 'cuda', 'sampled_monotonic': stamp,
           'resources': dict(CUDA if resources is None else resources)}
    row.update(overrides)
    try:
        monitor._atomic_heartbeat(resource_monitor.heartbeat_path, row)
    except ValueError:
        # Deliberately malformed nonfinite JSON is injected only in unthreaded tests.
        resource_monitor.heartbeat_path.write_text(json.dumps(row), encoding='utf-8')
    return row


def sample_with_heartbeat(tmp_path, *, native=None, cuda=None):
    resource_monitor = parent(tmp_path, native=native)
    resource_monitor._started = time.perf_counter()
    row = write_heartbeat(resource_monitor, cuda)
    return resource_monitor, row


def test_saved_448kib_total_difference_replays_through_actual_parent(tmp_path):
    saved = json.loads((EVIDENCE/'runtime-resource-heartbeat.json').read_text(encoding='utf-8'))
    telemetry = [json.loads(line) for line in
                 (EVIDENCE/'resource-telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    native = next(row['resources'] for row in reversed(telemetry) if row['event'] == 'resource_sample')
    assert native['total_bytes'] - saved['resources']['total_bytes'] == 448*1024
    assert 'physical GPU total mismatch' in telemetry[-1]['failure']
    resource_monitor = parent(tmp_path, native=native)
    resource_monitor._started = time.perf_counter()
    # Rebind transport identity/time only. All recorded CUDA counters stay exact.
    replay = {**saved, 'worker_pid': resource_monitor.job.pid, 'sampled_monotonic': time.perf_counter()}
    resource_monitor.heartbeat_path.write_text(json.dumps(replay), encoding='utf-8')
    try:
        resource_monitor._sample()
        assert resource_monitor._cuda_seen and resource_monitor.samples == 1
        row = read_last_good(resource_monitor.journal.path)['last_event']
        assert row['resources'] == row['native_resources'] == native
        assert row['cuda_resources'] == saved['resources']
        assert row['total_difference_bytes'] == 448*1024
        assert not resource_monitor.job.killed.is_set()
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('native_total,cuda_total', [
    (16*GIB, 16*GIB), (16*GIB, 16*GIB-1), (16*GIB, 15*GIB),
    (16*GIB, 64*GIB), (64*GIB, 16*GIB)])
def test_valid_domains_have_no_cross_total_equality_or_tolerance(tmp_path, native_total, cuda_total):
    native = {**NATIVE, 'total_bytes': native_total,
              'device_used_bytes': native_total-NATIVE['device_free_bytes']}
    cuda = {**CUDA, 'total_bytes': cuda_total}
    resource_monitor, heartbeat = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    try:
        # Existing return contract stays a timestamp, not a new public helper.
        assert resource_monitor._heartbeat(time.perf_counter(), native) == heartbeat['sampled_monotonic']
        resource_monitor._sample()
        row = read_last_good(resource_monitor.journal.path)['last_event']
        assert row['native_resources'] == row['resources'] == native
        assert row['cuda_resources'] == cuda
        assert row['total_difference_bytes'] == native_total-cuda_total
        assert resource_monitor.failure is None
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('fault', ['unaccounted_byte', 'overcounted_byte', 'zero_total',
                                   'v2_metadata', 'unknown_metadata'])
def test_native_v1_accounting_contract_is_strict_within_its_own_domain(tmp_path, fault):
    native = dict(NATIVE)
    if fault == 'unaccounted_byte': native['device_used_bytes'] -= 1
    elif fault == 'overcounted_byte': native['device_used_bytes'] += 1
    elif fault == 'zero_total': native.update(total_bytes=0, device_free_bytes=0, device_used_bytes=0)
    elif fault == 'v2_metadata': native['memory_api'] = 'nvmlDeviceGetMemoryInfo_v2'
    else: native['memory_api'] = 'unreviewed_memory_api'
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native)
    try:
        with pytest.raises(ResourceViolation):
            resource_monitor._sample()
        assert resource_monitor.samples == 0
    finally:
        resource_monitor.close()


def test_independent_domain_threshold_equalities_remain_inclusive(tmp_path):
    native = {**NATIVE, 'host_rss_bytes': 8*GIB, 'available_host_bytes': 4*GIB,
              'total_bytes': 64*GIB, 'device_free_bytes': 2*GIB, 'device_used_bytes': 62*GIB}
    cuda = {**CUDA, 'host_rss_bytes': 8*GIB, 'available_host_bytes': 4*GIB,
            'total_bytes': 15*GIB, 'device_free_bytes': 2*GIB,
            'reserved_bytes': 12*GIB, 'allocated_bytes': 12*GIB}
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    try:
        resource_monitor._sample()
        assert resource_monitor.samples == 1 and resource_monitor._cuda_seen
        assert not resource_monitor.job.killed.is_set()
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('phase', ['absent', 'pre_cuda'])
def test_pre_cuda_journal_does_not_fabricate_cuda_resources_or_total_difference(tmp_path, phase):
    resource_monitor = parent(tmp_path)
    resource_monitor._started = time.perf_counter()
    if phase == 'pre_cuda': write_heartbeat(resource_monitor, resources={}, phase='pre_cuda')
    try:
        resource_monitor._sample()
        row = read_last_good(resource_monitor.journal.path)['last_event']
        assert row['resources'] == row['native_resources'] == NATIVE
        assert row['cuda_resources'] is None and row['total_difference_bytes'] is None
        assert row['worker_heartbeat_monotonic'] is None and not resource_monitor._cuda_seen
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('domain', ['native', 'cuda', 'envelope'])
def test_each_physical_uuid_is_independently_enforced(tmp_path, domain):
    native, cuda = dict(NATIVE), dict(CUDA)
    if domain == 'native': native['device_uuid'] = 'GPU-wrong'
    if domain == 'cuda': cuda['device_uuid'] = 'GPU-wrong'
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    if domain == 'envelope': write_heartbeat(resource_monitor, cuda, gpu_uuid='GPU-wrong')
    try:
        with pytest.raises(ResourceViolation, match='UUID|identity'):
            resource_monitor._sample()
        assert resource_monitor.samples == 0
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('domain', ['native', 'cuda'])
@pytest.mark.parametrize('fault', ['free_above_total', 'insufficient_free'])
def test_free_memory_limits_are_not_borrowed_from_other_domain(tmp_path, domain, fault):
    native, cuda = dict(NATIVE), dict(CUDA)
    row = native if domain == 'native' else cuda
    row['device_free_bytes'] = row['total_bytes']+1 if fault == 'free_above_total' else 2*GIB-1
    if domain == 'native' and fault == 'insufficient_free':
        row['device_used_bytes'] = row['total_bytes']-row['device_free_bytes']
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    try:
        with pytest.raises(ResourceViolation, match='free-memory'):
            resource_monitor._sample()
        assert resource_monitor.samples == 0
    finally:
        resource_monitor.close()


BYTE_FIELDS = [('native', key) for key in ('host_rss_bytes', 'available_host_bytes',
    'total_host_bytes', 'total_bytes', 'device_free_bytes', 'device_used_bytes')]
BYTE_FIELDS += [('cuda', key) for key in ('host_rss_bytes', 'available_host_bytes',
    'total_host_bytes', 'total_bytes', 'device_free_bytes', 'allocated_bytes', 'reserved_bytes')]


@pytest.mark.parametrize('domain,key', BYTE_FIELDS)
@pytest.mark.parametrize('bad', [-1, True, '123', float('nan'), float('inf'), None, 1.5, 2**64],
                         ids=['negative', 'bool', 'string', 'nan', 'infinity', 'null', 'fraction', 'overflow'])
def test_actual_sampling_rejects_every_malformed_domain_byte_counter(tmp_path, domain, key, bad):
    native, cuda = dict(NATIVE), dict(CUDA)
    (native if domain == 'native' else cuda)[key] = bad
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    try:
        with pytest.raises(ResourceViolation, match='unsigned integer byte counter'):
            resource_monitor._sample()
        assert resource_monitor.samples == 0
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('domain,key', [
    *[('native', key) for key in ('host_rss_bytes', 'available_host_bytes', 'total_bytes',
                                'device_free_bytes', 'device_used_bytes', 'device_uuid')],
    *[('cuda', key) for key in ('host_rss_bytes', 'available_host_bytes', 'total_bytes',
                              'device_free_bytes', 'device_uuid', 'allocated_bytes', 'reserved_bytes')]])
def test_missing_required_domain_counter_never_falls_back_to_other_domain(tmp_path, domain, key):
    native, cuda = dict(NATIVE), dict(CUDA)
    (native if domain == 'native' else cuda).pop(key)
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    try:
        with pytest.raises(ResourceViolation):
            resource_monitor._sample()
        assert resource_monitor.samples == 0
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('fault', ['allocated_above_reserved', 'reserved_fraction'])
def test_allocator_limits_use_cuda_domain_total_not_native_total(tmp_path, fault):
    # Native total is much larger; borrowing it would incorrectly accept 13 GiB reserved.
    native = {**NATIVE, 'total_bytes': 64*GIB, 'device_used_bytes': 54*GIB}
    cuda = {**CUDA, 'total_bytes': 16*GIB}
    cuda.update({'allocated_bytes': 3*GIB} if fault == 'allocated_above_reserved'
                else {'reserved_bytes': 13*GIB})
    resource_monitor, _ = sample_with_heartbeat(tmp_path, native=native, cuda=cuda)
    try:
        with pytest.raises(ResourceViolation, match='exceeds reserved|reserved-memory'):
            resource_monitor._sample()
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('domain', ['native', 'cuda'])
def test_exhausted_domain_start_terminates_job_before_failure_evidence(tmp_path, monkeypatch, domain):
    native, cuda = dict(NATIVE), dict(CUDA)
    (native if domain == 'native' else cuda)['device_free_bytes'] = 0
    if domain == 'native': native['device_used_bytes'] = native['total_bytes']
    resource_monitor = parent(tmp_path, native=native)
    # start() requires an absent initial heartbeat. The injected probe publishes it
    # only once sampling begins, exactly as a subsequently running worker would.
    def probe():
        write_heartbeat(resource_monitor, cuda)
        return dict(native)
    resource_monitor.native_probe = probe
    emitted = threading.Event()
    observations = []
    def failure_evidence(*args, **kwargs):
        observations.append(resource_monitor.job.killed.is_set())
        emitted.set()
    monkeypatch.setattr(resource_monitor.journal, 'emit', failure_evidence)
    try:
        with pytest.raises(ResourceViolation, match='free-memory'):
            resource_monitor.start()
        assert resource_monitor.job.killed.is_set()
        assert emitted.wait(1.) and all(observations)
        receipt = resource_monitor.receipt()
        assert receipt['forced_tree_termination']
        assert not receipt['automatic_retry'] and not receipt['failed_state_checkpoint_allowed']
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('fault', ['missing_startup', 'missing_after_cuda', 'stale'])
def test_independent_heartbeat_watchdog_still_stops_without_worker_boundary(tmp_path, fault):
    resource_monitor = parent(tmp_path, startup_grace_seconds=.2, max_sample_age_seconds=.2)
    resource_monitor.start()
    try:
        if fault != 'missing_startup':
            write_heartbeat(resource_monitor)
            resource_monitor._sample()
            if fault == 'missing_after_cuda': resource_monitor.heartbeat_path.unlink()
        assert resource_monitor.job.killed.wait(1.), 'monitor must enforce missing/stale heartbeat independently'
        with pytest.raises(ResourceViolation, match='heartbeat|startup deadline'):
            resource_monitor.check()
        assert not resource_monitor.receipt()['failed_state_checkpoint_allowed']
    finally:
        resource_monitor.close()


@pytest.mark.parametrize('fault', ['exhaustion', 'deadline'])
def test_real_supervisor_integrates_monitor_and_original_deadline_with_inert_job(tmp_path, monkeypatch, fault):
    from test_wave6_supervision import reserve
    job = SyntheticJob()
    monkeypatch.setattr(supervisor, 'SuspendedJobProcess', lambda *args, **kwargs: job)
    monitors = []
    def factory(actual_job):
        def probe():
            return {**NATIVE, 'device_free_bytes': 0, 'device_used_bytes': NATIVE['total_bytes']} \
                if actual_job.resumed and fault == 'exhaustion' else dict(NATIVE)
        result = parent(tmp_path, job=actual_job, native_probe=probe,
                        max_sample_age_seconds=5., startup_grace_seconds=3.)
        actual_job.on_resume = lambda: write_heartbeat(result)
        monitors.append(result)
        return result
    ledger = reserve(tmp_path, seconds=2.5, margin=1., start=time.perf_counter()-.5, watchdog=False)
    try:
        receipt = supervisor.supervise(['SYNTHETIC_NOT_EXECUTED'], cwd=tmp_path, output=tmp_path/'output',
            experiment_id=ledger.experiment_id, authorization_id=ledger.authorization_id,
            ledger=ledger, resource_monitor_factory=factory)
        assert receipt['status'] == ('STOP_RESOURCE_POLICY' if fault == 'exhaustion' else 'STOP_DEADLINE')
        assert job.killed.is_set() and receipt['job_closed'] and receipt['process_tree_empty_verified']
        assert receipt['child_resumed'] and not receipt['timing']['budget_breached']
        assert receipt['last_committed'] is None and receipt['last_checkpoint'] is None
        assert not list((tmp_path/'output').glob('*.pt'))
        if fault == 'deadline':
            assert receipt['cooperative_stop_requested'] and monitors[0]._cuda_seen
            assert receipt['resource_monitor']['failure'] is None
    finally:
        ledger.finish('SYNTHETIC_TEST_FINISHED')


def test_changed_synthetic_binding_stops_real_monitor_without_any_claim(tmp_path):
    from test_wave6_authorization import make_synthetic_bundle
    from hift_research.wave6.authorization import verify_authorization, verify_bound_files
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-memory-binding-') as directory:
        bundle = make_synthetic_bundle(directory)
        grant = verify_authorization(bundle['proposal_path'], bundle['approval_path'], **bundle['runtime'])
        assert grant.synthetic_only
        resource_monitor = parent(tmp_path, authority_check=lambda: verify_bound_files(grant))
        try:
            resource_monitor.start()
            source = Path(next(iter(bundle['proposal']['source_files_sha256'])))
            source.write_text('# changed synthetic fixture; never executed\n', encoding='utf-8')
            assert resource_monitor.job.killed.wait(1.)
            with pytest.raises(ResourceViolation, match='hash mismatch'):
                resource_monitor.check()
            assert not (bundle['namespace']/'claims').exists()
            assert not resource_monitor.receipt()['automatic_retry']
        finally:
            resource_monitor.close()
