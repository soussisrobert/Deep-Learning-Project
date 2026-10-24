"""Stdlib-only Windows physical UUID lease and fail-closed host availability.

A lease is not authorization. WDDM C+G/N/A is NOT waived by process names.
Infrastructure-only synthetic tests may lock a real UUID without GPU compute.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET

DEFAULT_RESOURCES = {"host_process_peak_rss_bytes": 8 * 1024**3, "minimum_available_host_bytes": 4 * 1024**3, "max_cuda_reserved_fraction": .8}
GPU_PATTERN = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class LeaseError(RuntimeError):
    pass


class LeaseBusy(LeaseError):
    pass


def _uuid(value):
    if not isinstance(value, str) or not GPU_PATTERN.fullmatch(value):
        raise LeaseError("full physical GPU UUID required")
    return value


def _kernel():
    if os.name != "nt":
        raise LeaseError("real Windows lease/process APIs required; no POSIX emulation")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    if pid <= 0:
        raise LeaseError("positive actual Windows PID required")
    kernel = _kernel()
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
    if not handle:
        raise LeaseError(f"cannot establish live PID/creation identity:{pid};winerror={ctypes.get_last_error()}")
    try:
        creation, exit_time, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise LeaseError("cannot query process creation time")
        # GetProcessTimes exit FILETIME is undefined for a running process;
        # use the kernel process object's wait state for actual liveness.
        if kernel.WaitForSingleObject(handle, 0) != 0x102:
            raise LeaseError("process exited or liveness uncertain")
        return {"pid": pid, "creation_time_100ns": str((creation.dwHighDateTime << 32) | creation.dwLowDateTime),
                "identity_api": "Windows_GetProcessTimes_FILETIME"}
    finally:
        kernel.CloseHandle(handle)


def host_memory(pid=None):
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [(name, ctypes.c_ulonglong) for name in
                    ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                    ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                     "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = _kernel()
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise LeaseError("host available-memory query uncertain")
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 0x10, False, pid or os.getpid())
    if not handle:
        raise LeaseError("process RSS query uncertain")
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        memory = ProcessMemory()
        memory.cb = ctypes.sizeof(memory)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
            raise LeaseError("process RSS query failed")
        return {"available_host_bytes": int(status.ullAvailPhys), "total_host_bytes": int(status.ullTotalPhys),
                "process_rss_bytes": int(memory.WorkingSetSize), "process_peak_rss_bytes": int(memory.PeakWorkingSetSize)}
    finally:
        kernel.CloseHandle(handle)


def _mib(text):
    if not isinstance(text, str) or not re.fullmatch(r"\s*\d+(?:\.\d+)?\s+MiB\s*", text):
        raise LeaseError("physical VRAM accounting unavailable/N/A")
    return int(float(text.split()[0]) * 1024**2)


def _inventory_node(parent, path):
    nodes = parent.findall(path)
    if len(nodes) != 1:
        raise LeaseError("native inventory field missing/duplicated:" + path)
    return nodes[0]


def _inventory_container(node):
    if (node.text or "").strip() or any((child.tail or "").strip() for child in node):
        raise LeaseError("native inventory container contains unknown text:" + node.tag)
    return node


def _inventory_text(parent, path):
    node = _inventory_node(parent, path)
    if list(node) or (node.tail or "").strip():
        raise LeaseError("native inventory scalar malformed:" + path)
    return node.text or ""


def parse_physical_inventory(xml_text, gpu_uuid, resources, host, *, policy=None):
    from .policy import validate_policy_binding, COOPERATIVE_GUARDED_GPU, COOPERATIVE_SPEC
    try:
        cooperative = validate_policy_binding({"operational_policy": policy}) == COOPERATIVE_GUARDED_GPU
    except (ValueError, OSError) as exc:
        raise LeaseError("unreviewed operational GPU policy") from exc
    _uuid(gpu_uuid)
    if resources != DEFAULT_RESOURCES:
        raise LeaseError("unreviewed physical resource policy")
    try:
        for field in ("available_host_bytes", "process_rss_bytes", "process_peak_rss_bytes"):
            value = host[field]
            if type(value) is not int or not 0 <= value <= 2**64 - 1:
                raise LeaseError("native host counter unavailable/malformed:" + field)
        if "total_host_bytes" in host:
            total_host = host["total_host_bytes"]
            if type(total_host) is not int or not 0 < total_host <= 2**64 - 1 or host["available_host_bytes"] > total_host:
                raise LeaseError("inconsistent native host memory")
        root = _inventory_container(ET.fromstring(xml_text))
        matches = [gpu for gpu in root.findall("gpu") if _inventory_text(gpu, "uuid") == gpu_uuid]
        if len(matches) != 1:
            raise LeaseError("exact physical UUID absent/ambiguous")
        gpu = _inventory_container(matches[0])
        memory_group = _inventory_container(_inventory_node(gpu, "fb_memory_usage"))
        total = _mib(_inventory_text(memory_group, "total"))
        used = _mib(_inventory_text(memory_group, "used"))
        free = _mib(_inventory_text(memory_group, "free"))
        if total <= 0 or min(used, free) < 0 or used > total or free > total or used + free > total:
            raise LeaseError("inconsistent physical VRAM")
        reasons, processes = [], []
        group = _inventory_container(_inventory_node(gpu, "processes"))
        if group is None or (group.text or "").strip():
            raise LeaseError("GPU process inventory missing/unsupported/uncertain")
        if any(child.tag != "process_info" or (child.tail or "").strip() for child in group):
            raise LeaseError("GPU process inventory contains unknown data")
        seen_pids = set()
        for process in group.findall("process_info"):
            _inventory_container(process)
            kind = _inventory_text(process, "type").strip()
            pid_text = _inventory_text(process, "pid")
            try:
                pid = int(pid_text)
            except ValueError as exc:
                raise LeaseError("GPU process PID unavailable") from exc
            if not 0 < pid <= 0xffffffff or pid in seen_pids:
                raise LeaseError("GPU process identity invalid/duplicated")
            seen_pids.add(pid)
            row = {"pid": pid, "type": kind, "process_name": process.findtext("process_name"),
                   "used_memory": process.findtext("used_memory")}
            processes.append(row)
            # NO executable-name exception: C+G may genuinely be compute.
            if kind != "G" and not cooperative:
                reasons.append(f"unresolved_or_compute_GPU_process:{pid}:{kind or 'UNKNOWN'}")
        if used > total * resources["max_cuda_reserved_fraction"] or free < total * (1 - resources["max_cuda_reserved_fraction"]):
            reasons.append("physical_GPU_headroom_below20percent")
        if cooperative and free < COOPERATIVE_SPEC["minimum_launch_gpu_free_bytes"]:
            reasons.append("available_GPU_memory_below4GiB")
        if host["available_host_bytes"] < resources["minimum_available_host_bytes"]:
            reasons.append("available_host_memory_below4GiB")
        if max(host["process_rss_bytes"], host["process_peak_rss_bytes"]) > resources["host_process_peak_rss_bytes"]:
            reasons.append("process_RSS_above8GiB")
        return {"eligible": not reasons, "reasons": reasons, "gpu_uuid": gpu_uuid,
                "gpu_name": gpu.findtext("product_name"), "driver_model": gpu.findtext("driver_model/current_dm"),
                "VRAM_total_bytes": total, "VRAM_used_bytes": used, "VRAM_free_bytes": free,
                "processes": processes, "host": host,
                "ownership_assessment": "UNPROVEN_CONTEXT_CLASSIFICATION" if any(p["type"] != "G" for p in processes) else "NO_COMPUTE_DECLARED_IN_INVENTORY",
                "active_competing_workload_proven": False,
                "operational_policy": "COOPERATIVE_GUARDED_GPU" if cooperative else "STRICT_PHYSICAL_EXCLUSIVITY",
                "hardware_wide_exclusivity_proven": False,
                "external_context_interpretation": "opaque_external_clients_permitted_not_classified" if cooperative else "strict_classification_required",
                "interpretation": "instantaneous_readonly_inventory_not_GPU_compute_or_training_authority"}
    except (ET.ParseError, KeyError, ValueError, TypeError, OverflowError) as exc:
        raise LeaseError("physical inventory malformed/uncertain") from exc


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_smi_identity():
    """Resolve ONLY the trusted native Windows system-directory SMI producer.

    Never search PATH/CWD or honor a project-local nvidia-smi executable. This
    queries the OS directory and hashes a file; it performs no driver/GPU query.
    """
    kernel = _kernel()
    kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    buffer = ctypes.create_unicode_buffer(32768)
    size = kernel.GetSystemDirectoryW(buffer, len(buffer))
    if not size or size >= len(buffer):
        raise LeaseError("native Windows system directory unavailable/uncertain")
    directory = Path(buffer.value)
    if not directory.is_absolute():
        raise LeaseError("native Windows system directory is not absolute")
    path = (directory / "nvidia-smi.exe").resolve(strict=True)
    if not path.is_file():
        raise LeaseError("native system-directory nvidia-smi unavailable; no PATH fallback")
    return {"path": str(path), "sha256": _file_sha256(path), "argv": [str(path), "-q", "-x"],
            "resolver": "Windows_GetSystemDirectoryW"}


def probe_physical_availability(gpu_uuid, resources=None, *, snapshot_path=None, policy=None):
    resources = DEFAULT_RESOURCES if resources is None else resources
    report = {"eligible": False, "gpu_uuid": gpu_uuid, "captured_utc": datetime.now(timezone.utc).isoformat()}
    try:
        producer = native_smi_identity()
        report["producer"] = producer
        report["query_command"] = list(producer["argv"])
        result = subprocess.run(producer["argv"], capture_output=True, text=True, timeout=2 if policy is not None else 15, check=True)
        if _file_sha256(producer["path"]) != producer["sha256"]:
            raise LeaseError("native SMI producer bytes changed during query")
        if snapshot_path is not None:
            path = Path(snapshot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(result.stdout)
            report.update(snapshot_path=str(path.resolve()), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report.update(parse_physical_inventory(result.stdout, gpu_uuid, resources, host_memory(), policy=policy))
        if (report["operational_policy"] == "STRICT_PHYSICAL_EXCLUSIVITY" and report["driver_model"] == "WDDM"
                and report["ownership_assessment"] == "UNPROVEN_CONTEXT_CLASSIFICATION"):
            from dataclasses import asdict
            from .gpu_attribution import probe
            # Diagnose the native limitation, never reinterpret opaque contexts as graphics.
            # In production these native calls remain under the external parent deadline.
            report["windows_attribution_capability"] = asdict(probe(gpu_uuid))
            from .gpu_activity import ActivityError, SEMANTICS, sample_activity
            try:
                report["native_activity_observation"] = sample_activity(gpu_uuid)
            except (ActivityError, OSError, ValueError, AttributeError) as exc:
                report["native_activity_observation"] = {
                    "status": "unavailable", "error": type(exc).__name__ + ":" + str(exc),
                    "calls": list(getattr(exc, "calls", ())), "semantics": SEMANTICS,
                    "can_authorize_compute": False, "classifies_compute_vs_graphics": False, "empty_proves_idle": False}
    except (OSError, subprocess.SubprocessError, LeaseError) as exc:
        report["reasons"] = ["physical_availability_uncertain:" + str(exc)]
    return report


def _synthetic_root(value):
    root = Path(value).resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    if temporary not in root.parents or not root.name.startswith("hift-wave6-synthetic-"):
        raise LeaseError("infrastructure test requires dedicated synthetic TEMP namespace")
    return root


def default_lease_directory():
    return (Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / "HiFT" / "wave6-gpu-leases").resolve()


def _mutex_name(gpu_uuid):
    return "Global\\HiFT-W6-GPU-" + hashlib.sha256(_uuid(gpu_uuid).lower().encode()).hexdigest()


def _mutex(gpu_uuid, *, inspect_only=False):
    kernel = _kernel()
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, _mutex_name(gpu_uuid))
    if not handle:
        raise LeaseError("cannot establish Global physical UUID mutex")
    status = kernel.WaitForSingleObject(handle, 0)
    if inspect_only:
        if status in (0, 0x80):
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)
        if status != 0x102:
            raise LeaseError("parent physical UUID mutex not held or ownership uncertain")
        return None
    if status != 0:
        if status == 0x80:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)
        if status == 0x102:
            raise LeaseBusy("physical UUID mutex already held")
        raise LeaseError("abandoned/uncertain physical UUID mutex; manual reconciliation required")
    return handle


def _assert_lease_not_expired(metadata):
    if metadata.get("expires_utc") is not None:
        try:
            expiry = datetime.fromisoformat(metadata["expires_utc"].replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry.utcoffset().total_seconds() != 0 or datetime.now(timezone.utc) >= expiry:
                raise LeaseError("lease expired or uncertain expiry")
        except (TypeError, ValueError, AttributeError) as exc:
            raise LeaseError("malformed lease expiry") from exc


def probe_lease_availability(gpu_uuid):
    """Transient readiness only: actual shared mutex + durable production tombstone.

    This is not a lease/grant and cannot reserve the GPU after the call returns.
    All supported participating HiFT guarded invocations share this mutex.
    """
    _uuid(gpu_uuid)
    path = default_lease_directory() / (hashlib.sha256(gpu_uuid.lower().encode()).hexdigest() + ".lease.json")
    report = {"eligible": False, "gpu_uuid": gpu_uuid, "lease_path": str(path),
              "captured_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "participating_guarded_HiFT_processes_not_arbitrary_external_programs",
              "authority_granted": False, "ownership_retained": False}
    handle = None
    try:
        handle = _mutex(gpu_uuid)
        if path.exists():
            raise LeaseBusy("production active/stale/partial lease artifact; never stolen")
        report["owner_probe"] = process_identity()
        report["eligible"] = True
        report["reasons"] = []
    except (LeaseError, OSError) as exc:
        report["reasons"] = [type(exc).__name__ + ":" + str(exc)]
    finally:
        if handle is not None:
            kernel = _kernel()
            kernel.ReleaseMutex.argtypes = [wintypes.HANDLE];kernel.ReleaseMutex.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE];kernel.CloseHandle.restype = wintypes.BOOL
            released = kernel.ReleaseMutex(handle)
            closed = kernel.CloseHandle(handle)
            if not released or not closed:
                report["eligible"] = False
                report["reasons"] = ["transient mutex release/close uncertain"]
    return report


class GPULease:
    def __init__(self, path, metadata, handle):
        self.path, self.metadata, self.token, self._handle = Path(path), metadata, metadata["token"], handle
        self._released = False
        self._kernel_released = False
        self._release_uncertain = False
        self._thread_id = threading.get_ident()

    def assert_owned(self, *, allow_expired=False):
        if not allow_expired:
            _assert_lease_not_expired(self.metadata)
        if (self._released or self._kernel_released or self._release_uncertain
                or self._thread_id != threading.get_ident() or self.metadata["owner"] != process_identity(os.getpid())):
            raise LeaseError("lease no longer owned by current PID/creation identity")
        if json.loads(self.path.read_text(encoding="utf-8")) != self.metadata:
            raise LeaseError("lease ownership artifact changed; no automatic cleanup")
        return self.metadata

    def assert_live(self):
        """Read-only cross-thread identity check; never acquires/releases the mutex."""
        _assert_lease_not_expired(self.metadata)
        if (self._released or self._kernel_released or self._release_uncertain
                or self.metadata["owner"] != process_identity(os.getpid())
                or json.loads(self.path.read_text(encoding="utf-8")) != self.metadata):
            raise LeaseError("live parent lease identity/artifact mismatch")
        return self.metadata

    def release(self):
        if self._released:
            return
        if self._release_uncertain:
            raise LeaseError("lease release uncertain; ownership artifact retained, no automatic retry")
        self.assert_owned(allow_expired=True)
        # The supervisor must already have proved its Job descendants empty.
        # Keep the durable artifact blocking contenders until checked kernel
        # release succeeds. Any failure leaves it as a fail-closed tombstone.
        kernel = _kernel()
        kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel.ReleaseMutex.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        try:
            if not kernel.ReleaseMutex(self._handle):
                # Do NOT close the uncertain ownership handle or erase proof.
                raise LeaseError("native ReleaseMutex failed; ownership uncertain")
            self._kernel_released = True
            if not kernel.CloseHandle(self._handle):
                raise LeaseError("native mutex handle close uncertain")
            if json.loads(self.path.read_text(encoding="utf-8")) != self.metadata:
                raise LeaseError("release artifact changed; never erase another owner's record")
            self.path.unlink()
        except BaseException as exc:
            self._release_uncertain = True
            raise LeaseError("lease cleanup uncertain; ownership artifact retained, manual reconciliation required") from exc
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def acquire_lease(gpu_uuid, authorization_digest, *, resources=None, lease_directory=None,
                  synthetic_namespace=None, check_availability=True, policy=None, expires_utc=None):
    _uuid(gpu_uuid)
    from .policy import validate_policy_binding, COOPERATIVE_GUARDED_GPU
    try:
        policy_name = validate_policy_binding({"operational_policy": policy})
        if policy_name == COOPERATIVE_GUARDED_GPU and expires_utc is None:
            raise LeaseError("cooperative lease requires exact authorization expiry")
        if expires_utc is not None:
            _assert_lease_not_expired({"expires_utc": expires_utc})
    except (ValueError, OSError) as exc:
        raise LeaseError("invalid lease operational policy/expiry") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", authorization_digest):
        raise LeaseError("full authorization digest required")
    resources = DEFAULT_RESOURCES if resources is None else resources
    test = synthetic_namespace is not None
    if test:
        namespace = _synthetic_root(synthetic_namespace)
        directory = Path(lease_directory).resolve() if lease_directory is not None else namespace / "gpu-leases"
        if directory != namespace and namespace not in directory.parents:
            raise LeaseError("synthetic lease artifact cannot enter project/production namespace")
    else:
        directory = default_lease_directory()
        if lease_directory is not None and Path(lease_directory).resolve() != directory:
            raise LeaseError("production lease namespace is fixed across checkouts")
        if check_availability is not True:
            raise LeaseError("production physical availability check cannot be bypassed")
    handle = _mutex(gpu_uuid)
    path = directory / (hashlib.sha256(gpu_uuid.lower().encode()).hexdigest() + ".lease.json")
    descriptor = None
    created = False
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Any surviving stale/partial/dead-PID artifact fails closed. NEVER steal.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        metadata = {"schema": "hift-wave6-windows-GPU-lease-v1", "gpu_uuid": gpu_uuid,
                    "authorization_digest": authorization_digest, "owner": process_identity(os.getpid()),
                    "token": uuid.uuid4().hex, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "mutex_name": _mutex_name(gpu_uuid), "synthetic_only": test,
                    "infrastructure_only": not check_availability, "availability": None}
        if policy is not None:
            metadata["operational_policy"] = policy
        if expires_utc is not None:
            metadata["expires_utc"] = expires_utc
        if check_availability:
            metadata["availability"] = probe_physical_availability(gpu_uuid, resources, **({"policy": policy} if policy is not None else {}))
            if metadata["availability"]["eligible"] is not True:
                raise LeaseError("physical GPU/host unavailable or ambiguous: " + ";".join(metadata["availability"]["reasons"]))
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            json.dump(metadata, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return GPULease(path, metadata, handle)
    except BaseException as exc:
        if descriptor is not None:
            os.close(descriptor)
        # Only this call's freshly created, still-unpublished artifact is removed.
        if created:
            path.unlink(missing_ok=True)
        kernel = _kernel()
        kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)
        if isinstance(exc, FileExistsError):
            raise LeaseBusy("existing active/stale/partial lease; manual reconciliation required, never stolen") from exc
        raise


def verify_parent_lease(path, *, gpu_uuid, authorization_digest, expected_parent_pid=None, token=None, synthetic_only=False,
                        policy=None, expires_utc=None):
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
        raise LeaseError("exact parent lease nonce required; artifact alone is not dispatch capability")
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LeaseError("parent lease artifact unreadable/uncertain") from exc
    _assert_lease_not_expired(metadata)
    if policy is not None or expires_utc is not None:
        if metadata.get("operational_policy") != policy or metadata.get("expires_utc") != expires_utc:
            raise LeaseError("parent lease policy/expiry binding mismatch")
    parent = os.getppid()
    if expected_parent_pid is not None and int(expected_parent_pid) != parent:
        raise LeaseError("lease must belong to actual direct os.getppid, not an arbitrary ancestor")
    if (metadata.get("schema") != "hift-wave6-windows-GPU-lease-v1" or metadata.get("gpu_uuid") != gpu_uuid
            or metadata.get("authorization_digest") != authorization_digest or metadata.get("synthetic_only") is not synthetic_only
            or (token is not None and metadata.get("token") != token)
            or metadata.get("owner") != process_identity(parent)):
        raise LeaseError("parent lease UUID/digest/token/PID/creation identity mismatch")
    if not synthetic_only and metadata.get("infrastructure_only") is not False:
        raise LeaseError("infrastructure-only test lease cannot authorize production worker")
    _mutex(gpu_uuid, inspect_only=True)
    return metadata
