"""Policy-B resource containment, independent of Torch and controller boundaries.

Native calls and evidence I/O run outside the independent stale watchdog. This
is bounded cooperative protection, not protection against hostile GIL capture.
Global GPU free memory is resource evidence, NOT competing-context attribution.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import json
import math
import os
from pathlib import Path
import threading
import time


class ResourceViolation(RuntimeError):
    pass


def _number(row, key):
    value = row.get(key)
    if key.endswith('_bytes'):
        if type(value) is not int or not 0 <= value <= 2**64-1:
            raise ResourceViolation('missing/malformed unsigned integer byte counter:' + key)
    elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ResourceViolation('missing/malformed resource:' + key)
    return value


def validate_resources(resources, *, min_available_host_bytes=4*1024**3,
                       max_host_rss_bytes=8*1024**3, max_reserved_fraction=.8,
                       min_device_free_bytes=None, expected_gpu_uuid=None,
                       require_cuda=False):
    """Detached numeric checks shared by the parent and safe-boundary guard."""
    if not isinstance(resources, dict):
        raise ResourceViolation('resource probe must return mapping')
    if min_available_host_bytes is not None and _number(resources, 'available_host_bytes') < min_available_host_bytes:
        raise ResourceViolation('available host memory limit/missing probe')
    if max_host_rss_bytes is not None and _number(resources, 'host_rss_bytes') > max_host_rss_bytes:
        raise ResourceViolation('host memory limit/missing RSS')
    if 'total_host_bytes' in resources:
        total_host = _number(resources, 'total_host_bytes')
        if total_host <= 0 or _number(resources, 'available_host_bytes') > total_host:
            raise ResourceViolation('inconsistent available/total host memory')
    for key in ('reserved_bytes', 'total_bytes', 'allocated_bytes', 'device_free_bytes', 'device_used_bytes'):
        if key in resources:
            _number(resources, key)
    if require_cuda or 'reserved_bytes' in resources:
        total = _number(resources, 'total_bytes')
        reserved = _number(resources, 'reserved_bytes')
        allocated = _number(resources, 'allocated_bytes')
        if total <= 0 or reserved > total*max_reserved_fraction:
            raise ResourceViolation('CUDA reserved-memory limit')
        if allocated > reserved:
            raise ResourceViolation('allocated CUDA memory exceeds reserved')
    if min_device_free_bytes is not None:
        free, total = _number(resources, 'device_free_bytes'), _number(resources, 'total_bytes')
        if total <= 0 or free > total or free < min_device_free_bytes:
            raise ResourceViolation('physical GPU free-memory limit')
    if 'device_used_bytes' in resources:
        total = _number(resources, 'total_bytes'); used = _number(resources, 'device_used_bytes')
        if used > total or used + _number(resources, 'device_free_bytes') > total:
            raise ResourceViolation('inconsistent physical GPU free/used memory')
    if expected_gpu_uuid is not None:
        actual = resources.get('device_uuid')
        if not isinstance(actual, str) or actual.lower() != expected_gpu_uuid.lower():
            raise ResourceViolation('resource physical GPU UUID mismatch/missing')
    return dict(resources)


def native_host_resources(pid):
    """Read the actual worker's peak working set and native available host RAM."""
    if os.name != 'nt':
        raise ResourceViolation('native Windows resource probe required')
    class Counters(C.Structure):
        _fields_ = [('cb', W.DWORD), ('PageFaultCount', W.DWORD), ('PeakWorkingSetSize', C.c_size_t),
                    ('WorkingSetSize', C.c_size_t), ('QuotaPeakPagedPoolUsage', C.c_size_t),
                    ('QuotaPagedPoolUsage', C.c_size_t), ('QuotaPeakNonPagedPoolUsage', C.c_size_t),
                    ('QuotaNonPagedPoolUsage', C.c_size_t), ('PagefileUsage', C.c_size_t), ('PeakPagefileUsage', C.c_size_t)]
    class Memory(C.Structure):
        _fields_ = [('dwLength', W.DWORD), ('dwMemoryLoad', W.DWORD)] + [(name, C.c_ulonglong) for name in
                    ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                     'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
    kernel = C.WinDLL('kernel32.dll', use_last_error=True)
    kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]; kernel.OpenProcess.restype = W.HANDLE
    kernel.CloseHandle.argtypes = [W.HANDLE]; kernel.CloseHandle.restype = W.BOOL
    kernel.GlobalMemoryStatusEx.argtypes = [C.POINTER(Memory)]; kernel.GlobalMemoryStatusEx.restype = W.BOOL
    psapi = C.WinDLL('psapi.dll', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.POINTER(Counters), W.DWORD]
    psapi.GetProcessMemoryInfo.restype = W.BOOL
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, int(pid))
    if not handle:
        raise C.WinError(C.get_last_error(), 'OpenProcess resource worker')
    try:
        counters = Counters(); counters.cb = C.sizeof(counters)
        memory = Memory(); memory.dwLength = C.sizeof(memory)
        if not psapi.GetProcessMemoryInfo(handle, C.byref(counters), counters.cb):
            raise C.WinError(C.get_last_error(), 'GetProcessMemoryInfo worker')
        if not kernel.GlobalMemoryStatusEx(C.byref(memory)):
            raise C.WinError(C.get_last_error(), 'GlobalMemoryStatusEx')
        return {'host_rss_bytes': int(max(counters.PeakWorkingSetSize, counters.WorkingSetSize)),
                'available_host_bytes': int(memory.ullAvailPhys), 'total_host_bytes': int(memory.ullTotalPhys)}
    finally:
        if not kernel.CloseHandle(handle):
            raise C.WinError(C.get_last_error(), 'CloseHandle resource worker')


def native_gpu_resources(gpu_uuid):
    """System32 NVML global memory + exact UUID; no CUDA context or workload."""
    from .gpu_activity import native_library_path, load_native_library
    nv = load_native_library(native_library_path())
    def call(name, arguments, *values):
        function = getattr(nv, name); function.argtypes = arguments; function.restype = C.c_int32
        code = function(*values)
        if code != 0:
            raise ResourceViolation(name + ' failed:' + str(code))
    call('nvmlInit_v2', [])
    try:
        handle = C.c_void_p()
        call('nvmlDeviceGetHandleByUUID', [C.c_char_p, C.POINTER(C.c_void_p)], gpu_uuid.encode('ascii'), C.byref(handle))
        uuid = C.create_string_buffer(96)
        call('nvmlDeviceGetUUID', [C.c_void_p, C.c_void_p, C.c_uint32], handle, uuid, len(uuid))
        class Memory(C.Structure):
            _fields_ = [('total', C.c_uint64), ('free', C.c_uint64), ('used', C.c_uint64)]
        memory = Memory()
        call('nvmlDeviceGetMemoryInfo', [C.c_void_p, C.POINTER(Memory)], handle, C.byref(memory))
        return {'device_uuid': uuid.value.decode('ascii'), 'total_bytes': int(memory.total),
                'device_free_bytes': int(memory.free), 'device_used_bytes': int(memory.used)}
    finally:
        call('nvmlShutdown', [])


def _atomic_heartbeat(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
    with temporary.open('wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    # Shared-delete readers remove our own reader conflict. Windows scanners can
    # still briefly hold source/target without DELETE sharing. Bound that native
    # contention retry to 100ms; persistent denial remains fatal and never extends
    # the parent's trustworthy-sample age or startup deadline.
    deadline = time.perf_counter()+.1
    while True:
        try:
            os.replace(temporary, path)
            break
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32, 33) or time.perf_counter() >= deadline:
                raise
            time.sleep(.005)


def _read_heartbeat_bytes(path):
    if os.name != 'nt':
        with Path(path).open('rb') as stream:
            return stream.read(65537)
    # Python's ordinary Windows file open does not share DELETE. An atomic
    # heartbeat replacement must not fail just because the parent is reading it.
    kernel = C.WinDLL('kernel32.dll', use_last_error=True)
    kernel.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
    kernel.CreateFileW.restype = W.HANDLE
    kernel.ReadFile.argtypes = [W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p]
    kernel.ReadFile.restype = W.BOOL
    kernel.CloseHandle.argtypes = [W.HANDLE]; kernel.CloseHandle.restype = W.BOOL
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 1|2|4, None, 3, 0x80, None)
    if handle == C.c_void_p(-1).value:
        raise C.WinError(C.get_last_error(), 'CreateFileW shared-delete heartbeat')
    try:
        data = C.create_string_buffer(65537); count = W.DWORD()
        if not kernel.ReadFile(handle, data, len(data), C.byref(count), None):
            raise C.WinError(C.get_last_error(), 'ReadFile heartbeat')
        return data.raw[:count.value]
    finally:
        if not kernel.CloseHandle(handle):
            raise C.WinError(C.get_last_error(), 'CloseHandle heartbeat')


class WorkerResourceHeartbeat:
    """No model reads. A blocked CUDA probe becomes a parent-detected stale sample."""
    def __init__(self, path, gpu_uuid, authority_check, resource_probe, *, interval_seconds=1.):
        self.path, self.gpu_uuid = Path(path), gpu_uuid
        self.authority_check, self.resource_probe = authority_check, resource_probe
        self.interval = interval_seconds; self.phase = 'pre_cuda'; self.failure = None
        self._closed = threading.Event(); self._thread = None; self._sample_lock = threading.Lock()
        if type(interval_seconds) not in (int, float) or not 0 < interval_seconds <= 1:
            raise ValueError('heartbeat interval must be (0,1]')

    def activate(self):
        self.check()
        with self._sample_lock:
            self.phase = 'cuda'
            self._sample()
        self.check()

    def check(self):
        if self.failure is not None:
            raise ResourceViolation(self.failure)

    def _sample(self):
        stamp = time.perf_counter()
        if self.authority_check() is False:
            raise ResourceViolation('live authority check rejected')
        phase = self.phase
        resources = self.resource_probe() if phase == 'cuda' else {}
        _atomic_heartbeat(self.path, {'schema': 'hift-wave7-worker-resource-v1', 'worker_pid': os.getpid(),
            'gpu_uuid': self.gpu_uuid, 'sampled_monotonic': stamp, 'phase': phase, 'resources': resources})

    def _run(self):
        while not self._closed.is_set():
            started = time.perf_counter()
            try:
                with self._sample_lock:
                    self._sample()
            except BaseException as exc:
                self.failure = type(exc).__name__ + ':' + str(exc)
                return
            self._closed.wait(max(.001, self.interval - (time.perf_counter()-started)))

    def start(self):
        if self._thread is not None:
            raise RuntimeError('worker heartbeat is single-use')
        # Publish unknown/not-yet-initialized allocator state before ANY CUDA use.
        self._sample()
        self._thread = threading.Thread(target=self._run, name='wave7-worker-resource', daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=.1)


class ParentResourceMonitor:
    """Factory(job) target. A separate watchdog kills a hung operation on stale data.

    Boot grace applies ONLY to the worker's CUDA allocator heartbeat, never native
    host/GPU/authority probes. Termination precedes best-effort failure telemetry.
    """
    def __init__(self, job, *, gpu_uuid, heartbeat_path, journal_path, authority_check,
                 min_device_free_bytes=2*1024**3, min_available_host_bytes=4*1024**3,
                 max_host_rss_bytes=8*1024**3, max_reserved_fraction=.8,
                 interval_seconds=1., max_sample_age_seconds=5., startup_grace_seconds=30.,
                 native_query_timeout_seconds=2., native_probe=None, inventory_probe=None, limits=None):
        from .telemetry import DurableJournal
        if limits is not None:
            min_device_free_bytes = limits['minimum_runtime_gpu_free_bytes']
            min_available_host_bytes = limits['minimum_available_host_bytes']
            max_host_rss_bytes = limits['host_process_peak_rss_bytes']
            max_reserved_fraction = limits['max_cuda_reserved_fraction']
            interval_seconds = limits['sample_interval_seconds']
            max_sample_age_seconds = limits['maximum_sample_age_seconds']
            startup_grace_seconds = limits['worker_startup_grace_seconds']
            native_query_timeout_seconds = limits['native_query_timeout_seconds']
        for name, value in [('interval', interval_seconds), ('max_age', max_sample_age_seconds),
                            ('startup_grace', startup_grace_seconds), ('query_timeout', native_query_timeout_seconds)]:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid monitor timing:' + name)
        if interval_seconds > 1 or max_sample_age_seconds > 5 or startup_grace_seconds > 30 or native_query_timeout_seconds > 2:
            raise ValueError('monitor timing exceeds reviewed bound')
        if not callable(authority_check):
            raise TypeError('live authority callback required')
        self.job, self.gpu_uuid, self.heartbeat_path = job, gpu_uuid, Path(heartbeat_path)
        self.authority_check, self.inventory_probe = authority_check, inventory_probe
        self.native_probe = native_probe or (lambda: {**native_host_resources(job.pid), **native_gpu_resources(gpu_uuid)})
        self.limits = dict(min_device_free_bytes=min_device_free_bytes, min_available_host_bytes=min_available_host_bytes,
                           max_host_rss_bytes=max_host_rss_bytes, max_reserved_fraction=max_reserved_fraction, expected_gpu_uuid=gpu_uuid)
        self.interval, self.max_age, self.startup_grace = interval_seconds, max_sample_age_seconds, startup_grace_seconds
        self.query_timeout = native_query_timeout_seconds
        self.minimum_launch_gpu_free_bytes = limits['minimum_launch_gpu_free_bytes'] if limits is not None else min_device_free_bytes
        self.journal = DurableJournal(journal_path)
        self._closed = threading.Event(); self._ready = threading.Event(); self._lock = threading.Lock()
        self._started = None; self._last_sample = None; self._last_heartbeat = None; self._query_started = None
        self._cuda_seen = False; self.failure = None; self.terminated = False; self.samples = 0

    def _emit_failure(self):
        try:
            self.journal.emit('resource_monitor_failure', **self.receipt())
        except Exception:
            pass

    def _fail(self, reason):
        with self._lock:
            if self.failure is not None or self._closed.is_set():
                return
            self.failure = reason
        # No journal/heartbeat/disk/native-probe dependency before containment.
        try:
            self.job.terminate_tree(); self.terminated = True
        except Exception as exc:
            with self._lock:
                self.failure += '; termination failed:' + type(exc).__name__ + ':' + str(exc)
        finally:
            self._ready.set()
            threading.Thread(target=self._emit_failure, name='wave7-resource-evidence', daemon=True).start()

    def _heartbeat(self, now, native):
        try:
            payload = _read_heartbeat_bytes(self.heartbeat_path)
        except FileNotFoundError:
            if self._cuda_seen or now-self._started >= self.startup_grace:
                raise ResourceViolation('worker CUDA heartbeat missing/startup deadline')
            return None
        if len(payload) > 65536:
            raise ResourceViolation('oversize worker resource heartbeat')
        row = json.loads(payload)
        if (not isinstance(row, dict) or row.get('schema') != 'hift-wave7-worker-resource-v1'
                or type(row.get('worker_pid')) is not int or row['worker_pid'] != self.job.pid
                or not isinstance(row.get('gpu_uuid'), str) or row['gpu_uuid'].lower() != self.gpu_uuid.lower()):
            raise ResourceViolation('worker heartbeat identity/schema mismatch')
        stamp = _number(row, 'sampled_monotonic')
        if stamp < self._started or stamp > now or now-stamp >= self.max_age:
            raise ResourceViolation('worker resource heartbeat stale/future')
        if self._last_heartbeat is not None and stamp < self._last_heartbeat:
            raise ResourceViolation('worker resource heartbeat rolled back')
        phase = row.get('phase')
        if phase == 'pre_cuda' and not self._cuda_seen and now-self._started < self.startup_grace:
            if row.get('resources') != {}:
                raise ResourceViolation('pre-CUDA heartbeat fabricated allocator observation')
            return None
        if phase != 'cuda':
            raise ResourceViolation('worker CUDA heartbeat startup deadline/phase regression')
        resources = row.get('resources')
        validate_resources(resources, **self.limits, require_cuda=True)
        if resources['total_bytes'] != native['total_bytes']:
            raise ResourceViolation('worker/native physical GPU total mismatch')
        return stamp

    def _sample(self):
        started = time.perf_counter()
        with self._lock:
            self._query_started = started
        if self.authority_check() is False:
            raise ResourceViolation('live authority check rejected')
        native = self.native_probe()
        sample_limits = dict(self.limits)
        if self.samples == 0:
            sample_limits['min_device_free_bytes'] = self.minimum_launch_gpu_free_bytes
        validate_resources(native, **sample_limits)
        inventory = self.inventory_probe() if self.inventory_probe is not None else None
        now = time.perf_counter()
        heartbeat = self._heartbeat(now, native)
        if now-started >= self.query_timeout:
            raise ResourceViolation('native resource/authority query timeout')
        with self._lock:
            if self.failure is not None:
                return
            self._last_sample = started; self._query_started = None
            if heartbeat is not None:
                self._last_heartbeat = heartbeat; self._cuda_seen = True
            self.samples += 1
        self._ready.set()
        self.journal.emit('resource_sample', resources=native, worker_heartbeat_monotonic=heartbeat,
                          inventory_observation=inventory, context_absence_proven=False)

    def _run(self):
        while not self._closed.is_set():
            started = time.perf_counter()
            try:
                if self.job.poll() is not None:
                    return
                self._sample()
            except BaseException as exc:
                if self.job.poll() is None:
                    self._fail(type(exc).__name__ + ':' + str(exc))
                return
            self._closed.wait(max(.001, self.interval-(time.perf_counter()-started)))

    def _watch(self):
        while not self._closed.wait(.05):
            now = time.perf_counter()
            with self._lock:
                stamp = self._last_sample if self._last_sample is not None else self._started
                query, heartbeat, cuda = self._query_started, self._last_heartbeat, self._cuda_seen
            if query is not None and now-query >= self.query_timeout:
                self._fail('native resource/authority query timeout'); return
            if now-stamp >= self.max_age:
                self._fail('native resource sample stale'); return
            if cuda and (heartbeat is None or now-heartbeat >= self.max_age):
                self._fail('worker resource heartbeat stale'); return
            if not cuda and now-self._started >= self.startup_grace:
                self._fail('worker CUDA heartbeat startup deadline'); return

    def start(self):
        if self._started is not None:
            raise RuntimeError('parent resource monitor is single-use')
        if self.heartbeat_path.exists():
            raise FileExistsError('preexisting worker resource heartbeat')
        self._started = time.perf_counter()
        self._watchdog = threading.Thread(target=self._watch, name='wave7-resource-watchdog', daemon=True)
        self._sampler = threading.Thread(target=self._run, name='wave7-parent-resource', daemon=True)
        self._watchdog.start(); self._sampler.start()
        self._ready.wait(self.query_timeout+.1)
        self.check()
        if self._last_sample is None:
            self._fail('no trustworthy initial native resource sample')
            self.check()
        return self

    def check(self):
        if self.failure is not None:
            raise ResourceViolation(self.failure)
        if self._closed.is_set():
            raise ResourceViolation('resource monitor already closed')

    def receipt(self):
        return {'schema': 'hift-wave7-resource-monitor-receipt-v1', 'failure': self.failure,
                'forced_tree_termination': self.terminated, 'native_samples': self.samples,
                'last_sample_monotonic': self._last_sample, 'last_worker_heartbeat_monotonic': self._last_heartbeat,
                'cuda_heartbeat_seen': self._cuda_seen, 'automatic_retry': False,
                'failed_state_checkpoint_allowed': False, 'journal_path': str(self.journal.path)}

    def close(self):
        self._closed.set()
        for name in ('_sampler', '_watchdog'):
            thread = getattr(self, name, None)
            if thread is not None:
                thread.join(timeout=.1)
        return self.receipt()
