"""CPU-only physical UUID lease/process tests; no model, grant, CUDA context."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from hift_research.wave6.lease import (
    DEFAULT_RESOURCES, LeaseBusy, LeaseError, acquire_lease, host_memory,
    parse_physical_inventory, probe_physical_availability, process_identity, verify_parent_lease,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_UUID = "GPU-00000000-0000-0000-0000-000000000016"
AUTH_DIGEST = "e" * 64  # Explicit infrastructure-only synthetic token, NOT a grant.
HOST = {"available_host_bytes": 16 * 1024**3, "total_host_bytes": 64 * 1024**3,
        "process_rss_bytes": 1024**3, "process_peak_rss_bytes": 2 * 1024**3}


def xml_inventory(kind="G", name="dwm.exe", used="1000 MiB"):
    return f'''<nvidia_smi_log><gpu><uuid>{TEST_UUID}</uuid><product_name>synthetic inventory</product_name>
<driver_model><current_dm>WDDM</current_dm></driver_model><fb_memory_usage><total>16000 MiB</total>
<used>{used}</used><free>14000 MiB</free></fb_memory_usage><processes><process_info><pid>123</pid>
<type>{kind}</type><process_name>{name}</process_name><used_memory>N/A</used_memory></process_info>
</processes></gpu></nvidia_smi_log>'''


class _FakeSystemDirectoryCall:
    def __init__(self, directory, fail=False):
        self.directory, self.fail = str(directory), fail

    def __call__(self, buffer, size):
        if self.fail:
            return 0
        buffer.value = self.directory
        return len(self.directory)


def test_native_producer_ignores_project_PATH_and_records_absolute_binary_identity(tmp_path, monkeypatch):
    import hift_research.wave6.lease as module
    system = tmp_path / "mock-system-directory"
    shadow = tmp_path / "project-PATH-shadow"
    system.mkdir()
    shadow.mkdir()
    (system / "nvidia-smi.exe").write_bytes(b"inert synthetic OS producer marker; never executed")
    (shadow / "nvidia-smi.exe").write_bytes(b"inert synthetic project shadow; never executed")
    monkeypatch.setenv("PATH", str(shadow))
    monkeypatch.setattr(module, "_kernel", lambda: SimpleNamespace(GetSystemDirectoryW=_FakeSystemDirectoryCall(system)))
    identity = module.native_smi_identity()
    assert identity["path"] == str((system / "nvidia-smi.exe").resolve())
    assert identity["argv"] == [identity["path"], "-q", "-x"]
    assert identity["sha256"] == hashlib.sha256((system / "nvidia-smi.exe").read_bytes()).hexdigest()
    assert identity["resolver"] == "Windows_GetSystemDirectoryW"


@pytest.mark.parametrize("failure", ["missing", "resolver"])
def test_native_producer_fails_closed_without_PATH_fallback(tmp_path, monkeypatch, failure):
    import hift_research.wave6.lease as module
    native = tmp_path / "missing-native-producer"
    native.mkdir()
    (tmp_path / "nvidia-smi.exe").write_bytes(b"project fallback must never be searched/executed")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(module, "_kernel", lambda: SimpleNamespace(GetSystemDirectoryW=_FakeSystemDirectoryCall(native, fail=failure == "resolver")))
    with pytest.raises((LeaseError, OSError)):
        module.native_smi_identity()


def test_producer_dispatch_uses_only_absolute_OS_binary_and_rejects_midquery_drift(tmp_path, monkeypatch):
    import hift_research.wave6.lease as module
    native = tmp_path / "mock-system-directory"
    native.mkdir()
    executable = native / "nvidia-smi.exe"
    executable.write_bytes(b"inert synthetic producer bytes; never executed")
    monkeypatch.setattr(module, "_kernel", lambda: SimpleNamespace(GetSystemDirectoryW=_FakeSystemDirectoryCall(native)))
    monkeypatch.setattr(module, "host_memory", lambda: dict(HOST))
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv == [str(executable.resolve()), "-q", "-x"]
        return SimpleNamespace(stdout=xml_inventory("C+G", "explorer.exe"))
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    report = module.probe_physical_availability(TEST_UUID)
    assert calls and report["eligible"] is False  # C+G is still ambiguous; no name waiver.
    assert report["producer"]["path"] == str(executable.resolve())
    assert report["query_command"] == calls[0]
    assert report["producer"]["sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    def drift_during_query(argv, **kwargs):
        executable.write_bytes(b"changed inert marker during mocked query")
        return SimpleNamespace(stdout=xml_inventory("G"))
    monkeypatch.setattr(module.subprocess, "run", drift_during_query)
    changed = module.probe_physical_availability(TEST_UUID)
    assert changed["eligible"] is False
    assert "producer bytes changed" in changed["reasons"][0]


class _FakeNativeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def _inert_release_fixture(tmp_path, monkeypatch, kernel):
    import hift_research.wave6.lease as module
    owner = {"pid": os.getpid(), "creation_time_100ns": "SYNTHETIC_METADATA_ONLY"}
    metadata = {"owner": owner, "token": "f" * 32, "synthetic_only": True, "infrastructure_only": True}
    path = tmp_path / "synthetic-release-metadata.json"
    path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(module, "process_identity", lambda pid=None: dict(owner))
    monkeypatch.setattr(module, "_kernel", lambda: kernel)
    return module.GPULease(path, metadata, 12345)  # Inert integer: never passed to Windows.


def test_release_mutex_failure_retains_metadata_without_close_or_automatic_retry(tmp_path, monkeypatch):
    events = []
    kernel = SimpleNamespace(ReleaseMutex=_FakeNativeFunction(lambda handle: events.append("release_failed") or 0),
                             CloseHandle=_FakeNativeFunction(lambda handle: events.append("close") or 1))
    lease = _inert_release_fixture(tmp_path, monkeypatch, kernel)
    original = lease.path.read_bytes()
    with pytest.raises(LeaseError, match="artifact retained"):
        lease.release()
    assert lease.path.read_bytes() == original
    assert events == ["release_failed"]
    assert lease._release_uncertain and not lease._released and not lease._kernel_released
    with pytest.raises(LeaseError, match="no automatic retry"):
        lease.release()
    assert events == ["release_failed"]


def test_release_confirms_kernel_success_before_unlinking_metadata(tmp_path, monkeypatch):
    events = []
    kernel = SimpleNamespace(ReleaseMutex=_FakeNativeFunction(lambda handle: events.append("release") or 1),
                             CloseHandle=_FakeNativeFunction(lambda handle: events.append("close") or 1))
    lease = _inert_release_fixture(tmp_path, monkeypatch, kernel)
    original_unlink = Path.unlink
    def checked_unlink(path, *args, **kwargs):
        if path == lease.path:
            assert events == ["release", "close"]
            events.append("unlink")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", checked_unlink)
    lease.release()
    assert events == ["release", "close", "unlink"]
    assert lease._released and lease._kernel_released and not lease._release_uncertain
    assert not lease.path.exists()


@pytest.mark.parametrize("fault", ["close", "unlink"])
def test_postrelease_cleanup_uncertainty_retains_metadata_tombstone(tmp_path, monkeypatch, fault):
    events = []
    kernel = SimpleNamespace(ReleaseMutex=_FakeNativeFunction(lambda handle: events.append("release") or 1),
                             CloseHandle=_FakeNativeFunction(lambda handle: events.append("close") or (0 if fault == "close" else 1)))
    lease = _inert_release_fixture(tmp_path, monkeypatch, kernel)
    original = lease.path.read_bytes()
    original_unlink = Path.unlink
    def fail_unlink(path, *args, **kwargs):
        if path == lease.path:
            raise OSError("synthetic metadata unlink fault")
        return original_unlink(path, *args, **kwargs)
    if fault == "unlink":
        monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(LeaseError, match="artifact retained"):
        lease.release()
    assert lease.path.read_bytes() == original
    assert lease._kernel_released and lease._release_uncertain and not lease._released
    assert events == ["release", "close"]


def test_graphics_is_distinguished_from_compute_without_name_allowlist():
    assert parse_physical_inventory(xml_inventory(), TEST_UUID, DEFAULT_RESOURCES, HOST)["eligible"] is True
    for kind, name in (("C", "python.exe"), ("C+G", "dwm.exe"), ("C+G", "explorer.exe"), ("", "Harness.exe"), ("UNKNOWN", "game.exe")):
        result = parse_physical_inventory(xml_inventory(kind, name), TEST_UUID, DEFAULT_RESOURCES, HOST)
        assert result["eligible"] is False
        assert "GPU_process" in result["reasons"][0]


@pytest.mark.parametrize("fault", ["VRAM", "free_host", "RSS", "unknown_memory", "missing_processes", "UUID"])
def test_physical_prerequisite_failclosed(fault):
    host, xml, gpu = deepcopy(HOST), xml_inventory(), TEST_UUID
    if fault == "VRAM":
        xml = xml.replace("1000 MiB", "15000 MiB").replace("14000 MiB", "500 MiB")
    elif fault == "free_host":
        host["available_host_bytes"] = 1024**3
    elif fault == "RSS":
        host["process_peak_rss_bytes"] = 9 * 1024**3
    elif fault == "unknown_memory":
        xml = xml.replace("16000 MiB", "N/A")
    elif fault == "missing_processes":
        xml = xml.replace("<processes>", "<missing>").replace("</processes>", "</missing>")
    else:
        gpu = TEST_UUID[:-1] + "7"
    try:
        result = parse_physical_inventory(xml, gpu, DEFAULT_RESOURCES, host)
    except LeaseError:
        return
    assert result["eligible"] is False


@pytest.mark.skipif(os.name != "nt", reason="actual native Windows lease API required")
def test_actual_windows_creation_identity_and_host_memory():
    owner = process_identity()
    assert owner == process_identity(os.getpid())
    assert owner["identity_api"] == "Windows_GetProcessTimes_FILETIME"
    assert int(owner["creation_time_100ns"]) > 0
    assert host_memory()["available_host_bytes"] > 0
    with pytest.raises(LeaseError):
        process_identity(4294967294)


@pytest.mark.skipif(os.name != "nt", reason="real Windows UUID mutex required")
def test_lease_exclusive_artifact_stale_uncertain_never_stolen_and_release():
    with tempfile.TemporaryDirectory(prefix="hift-wave6-synthetic-lease-") as namespace:
        lease = acquire_lease(TEST_UUID, AUTH_DIGEST, synthetic_namespace=namespace, check_availability=False)
        path = lease.path
        try:
            assert lease.assert_owned()["infrastructure_only"] is True
            assert lease.metadata["synthetic_only"] is True
            with pytest.raises(LeaseBusy):
                acquire_lease(TEST_UUID, AUTH_DIGEST, synthetic_namespace=namespace, check_availability=False)
            errors = []
            def wrong_thread_release():
                try:
                    lease.release()
                except LeaseError as exc:
                    errors.append(str(exc))
            thread = threading.Thread(target=wrong_thread_release)
            thread.start()
            thread.join(timeout=5)
            assert errors and path.exists()
        finally:
            lease.release()
        assert not path.exists()
        path.write_text('{"owner":{"pid":4294967294},"uncertain":true}', encoding="utf-8")
        original = path.read_bytes()
        try:
            with pytest.raises(LeaseBusy, match="stale"):
                acquire_lease(TEST_UUID, AUTH_DIGEST, synthetic_namespace=namespace, check_availability=False)
            assert path.read_bytes() == original  # NO automatic stale deletion/steal.
        finally:
            path.unlink()  # Only this test's known synthetic fixture, not a real lease.
        with acquire_lease(TEST_UUID, AUTH_DIGEST, synthetic_namespace=namespace, check_availability=False):
            pass
        assert not path.exists()


def test_production_lease_never_bypasses_availability_or_redirects_namespace():
    with pytest.raises(LeaseError):
        acquire_lease(TEST_UUID, AUTH_DIGEST, check_availability=False)
    with pytest.raises(LeaseError):
        acquire_lease(TEST_UUID, AUTH_DIGEST, lease_directory=ROOT / "artifacts/fake-production-lease")


def _physical_uuid():
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True, timeout=15)
    parsed = ET.fromstring(result.stdout)
    return parsed.findtext("gpu/uuid")


@pytest.mark.skipif(os.name != "nt", reason="actual physical Windows GPU UUID infrastructure validation")
def test_actual_physical_uuid_concurrency_and_parent_creation_proof_without_GPU_context():
    gpu_uuid = _physical_uuid()
    assert gpu_uuid and gpu_uuid.startswith("GPU-")
    with tempfile.TemporaryDirectory(prefix="hift-wave6-synthetic-physical-") as namespace:
        lease = acquire_lease(gpu_uuid, AUTH_DIGEST, synthetic_namespace=namespace, check_availability=False)
        competitor = """import sys
sys.path.insert(0,sys.argv[1])
from hift_research.wave6.lease import acquire_lease,LeaseBusy
try:
 x=acquire_lease(sys.argv[2],sys.argv[3],synthetic_namespace=sys.argv[4],check_availability=False)
except LeaseBusy:
 sys.exit(17)
else:
 x.release();sys.exit(0)
"""
        proof = """import os,sys
sys.path.insert(0,sys.argv[1])
from hift_research.wave6.lease import verify_parent_lease
x=verify_parent_lease(sys.argv[2],gpu_uuid=sys.argv[3],authorization_digest=sys.argv[4],token=sys.argv[5],synthetic_only=True)
assert x['owner']['pid']==os.getppid()
"""
        try:
            busy = subprocess.run([sys.executable, "-c", competitor, str(ROOT / "src"), gpu_uuid, AUTH_DIGEST, namespace], timeout=15)
            assert busy.returncode == 17
            # Native Windows venv redirectors insert an extra parent process.
            # For this infrastructure-only child proof, launch base Python with
            # the explicit venv profile while retaining strict direct PPID.
            direct_environment = {**os.environ, "__PYVENV_LAUNCHER__": sys.executable}
            subprocess.run([sys._base_executable, "-c", proof, str(ROOT / "src"), str(lease.path), gpu_uuid, AUTH_DIGEST, lease.token],
                           env=direct_environment, check=True, timeout=15)
            with pytest.raises(LeaseError):
                verify_parent_lease(lease.path, gpu_uuid=gpu_uuid, authorization_digest="f" * 64,
                                    expected_parent_pid=os.getpid(), token=lease.token, synthetic_only=True)
        finally:
            lease.release()
        assert not lease.path.exists()
        free = subprocess.run([sys.executable, "-c", competitor, str(ROOT / "src"), gpu_uuid, AUTH_DIGEST, namespace], timeout=15)
        assert free.returncode == 0
        assert not list((Path(namespace) / "gpu-leases").glob("*.lease.json"))


@pytest.mark.skipif(os.name != "nt", reason="actual readonly Windows SMI availability check")
def test_actual_physical_availability_snapshot_is_not_a_compute_permission(tmp_path):
    gpu_uuid = _physical_uuid()
    snapshot = tmp_path / "actual-smi.xml"
    report = probe_physical_availability(gpu_uuid, DEFAULT_RESOURCES, snapshot_path=snapshot)
    assert snapshot.exists() and report["snapshot_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert isinstance(report["eligible"], bool)
    if any(row["type"] != "G" for row in report.get("processes", [])):
        assert report["eligible"] is False  # WDDM C+G ambiguity cannot be waived.
