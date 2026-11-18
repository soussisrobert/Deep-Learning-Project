"""SYNTHETIC TEMP-only ACK fixtures: never project grants or authentic SGD."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from hift_research.wave6.authorization import (
    ACK_SCHEMA, PROPOSAL_SCHEMA, RESOURCE_LIMITS, AuthorizationError, assert_claim,
    canonical_digest, claim_authorization, expected_ack, file_digest, read_acknowledgment,
    reverify_sources, verify_authorization, verify_bound_files,
)

FIXTURE_UUID = "GPU-00000000-0000-0000-0000-000000000006"
ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_synthetic_bundle(namespace, *, argv=None, output_path=None, source_files_sha256=None, limits=None, scope_extra=None):
    """Reusable integration fixture. Namespace MUST be dedicated TEMP prefix.

    argv accepts literal {proposal}/{authorization}/{output} placeholder arguments.
    Only synthetic CPU fixture config/data/init/ACK files are created. Optional
    source maps may reference project code READONLY, never authentic image assets.
    """
    namespace = Path(namespace).resolve()
    assert namespace.name.startswith("hift-wave6-synthetic-")
    assert Path(tempfile.gettempdir()).resolve() in namespace.parents
    namespace.mkdir(parents=True, exist_ok=True)
    pp, ap = namespace / "proposal.json", namespace / "synthetic-authorization.json"
    output = Path(output_path).resolve() if output_path else namespace / "synthetic-output"
    source = namespace / "fixture.py"
    source.write_text("# Synthetic fixture only; no Torch/no real data/no SGD.\n", encoding="utf-8")
    data, init = namespace / "synthetic-data.bin", namespace / "synthetic-init.bin"
    data.write_bytes(b"SYNTHETIC_RANDOM_ONLY_NO_AUTHENTIC_DATA")
    init.write_bytes(b"SYNTHETIC_INITIALIZATION_ONLY_NO_REAL_MODEL")
    config, inventory = namespace / "synthetic-config.json", namespace / "source-inventory.json"
    data_manifest, init_manifest = namespace / "synthetic-data-manifest.json", namespace / "synthetic-init-manifest.json"
    _write(config, {"synthetic_random_only": True})
    sources = dict(source_files_sha256 or {str(source): file_digest(source)})
    data_map, init_map = {str(data): file_digest(data)}, {str(init): file_digest(init)}
    initial = (scope_extra or {}).get("initialization") or {k: file_digest(init) for k in ("all_parameters_sha256", "backbone_state_sha256", "grader_parameters_sha256")}
    _write(inventory, {"file_sha256": sources})
    _write(data_manifest, {"file_sha256": data_map})
    _write(init_manifest, {"file_sha256": init_map, "initialization": initial})
    command = argv or [sys.executable, str(source), "--proposal", "{proposal}", "--authorization", "{authorization}", "--output", "{output}", "--synthetic-test"]
    replacements = {"{proposal}": str(pp), "{authorization}": str(ap), "{output}": str(output)}
    command = [replacements.get(str(v), str(v)) for v in command]
    now = datetime.now(timezone.utc)
    proposal = {"schema": PROPOSAL_SCHEMA, "proposal_id": "SYNTHETIC-TEMP-ONLY-001", "experiment_id": "SYNTHETIC-NO-REAL-SGD",
                "created_utc": (now - timedelta(seconds=2)).isoformat(), "expires_utc": (now + timedelta(minutes=15)).isoformat(),
                "cwd": str(namespace), "argv": command, "output_path": str(output), "approval_path": str(ap),
                "claim_directory": str(namespace / "claims"), "max_invocations": 1, "resume_allowed": False,
                "automatic_retry": False, "scope_expansion": False, "synthetic_only": True, "synthetic_namespace": str(namespace),
                "device": {"name": "cpu", "gpu_uuid": FIXTURE_UUID}, "resources": dict(RESOURCE_LIMITS),
                "limits": limits or {"max_effective_updates": 1, "max_training_attempts": 16, "max_windows": 1, "max_wall_seconds": 30},
                "scope": {"synthetic_random_only": True, "initialization": initial, **(scope_extra or {})},
                "bindings": {key: {"path": str(path), "sha256": file_digest(path)} for key, path in
                             (("config", config), ("source_inventory", inventory), ("data_manifest", data_manifest), ("initialization_manifest", init_manifest))},
                "source_files_sha256": sources, "data_files_sha256": data_map, "initialization_files_sha256": init_map}
    bundle = {"namespace": namespace, "proposal_path": pp, "approval_path": ap, "proposal": proposal}
    resign_synthetic_bundle(bundle)
    return bundle


def resign_synthetic_bundle(bundle):
    """Fixture mutation helper refuses to mint a real-role/real-SGD envelope."""
    proposal = bundle["proposal"]
    assert proposal["synthetic_only"] is True
    assert proposal["device"]["name"] == "cpu"
    pp, ap = bundle["proposal_path"], bundle["approval_path"]
    _write(pp, proposal)
    approval = {"schema": ACK_SCHEMA, "approval_id": "SYNTHETIC-TEMP-TRANSPORT-NOT-HUMAN", "proposal_sha256": file_digest(pp),
                "proposal_digest": canonical_digest(proposal), "expires_utc": proposal["expires_utc"], "max_invocations": 1,
                "human_ack": {"role": "synthetic_test_user", "session_id": "synthetic-session-not-human", "message_id": "synthetic-message-not-human",
                              "received_utc": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                              "text": expected_ack(proposal, file_digest(pp))}}
    _write(ap, approval)
    bundle["approval"] = approval
    bundle["runtime"] = {"actual_argv": proposal["argv"], "actual_cwd": proposal["cwd"], "actual_output": proposal["output_path"],
                         "gpu_uuid": proposal["device"]["gpu_uuid"], "allow_synthetic": True}
    return bundle


@pytest.fixture
def bundle():
    with tempfile.TemporaryDirectory(prefix="hift-wave6-synthetic-auth-") as directory:
        yield make_synthetic_bundle(directory)


def _verify(bundle, **kwargs):
    return verify_authorization(bundle["proposal_path"], bundle["approval_path"], **{**bundle["runtime"], **kwargs})


def test_synthetic_rejected_by_production_default_and_exact_text_not_boolean(bundle):
    with pytest.raises(AuthorizationError):
        _verify(bundle, allow_synthetic=False)
    grant = _verify(bundle)
    assert grant.synthetic_only and grant.proposal["device"]["name"] == "cpu"
    approval = bundle["approval"]
    approval["human_ack"]["text"] = "I approve"
    approval["human_approved"] = True
    _write(bundle["approval_path"], approval)
    with pytest.raises(AuthorizationError, match="exact human ACK"):
        _verify(bundle)


@pytest.mark.parametrize("field", ["actual_argv", "actual_cwd", "actual_output", "gpu_uuid"])
def test_exact_runtime_binding_denies_changed_command_output_device_cwd(bundle, field):
    value = {"actual_argv": [*bundle["runtime"]["actual_argv"], "--changed"], "actual_cwd": str(ROOT),
             "actual_output": str(bundle["namespace"] / "different-output"), "gpu_uuid": FIXTURE_UUID[:-1] + "7"}[field]
    with pytest.raises(AuthorizationError):
        _verify(bundle, **{field: value})


@pytest.mark.parametrize("kind", ["proposal_bytes", "ack_role", "missing_message", "expired", "retry", "limits", "device", "config_escape", "data_escape"])
def test_denials_for_drift_implicit_ack_expiry_scope_and_temp_escape(bundle, kind):
    if kind == "proposal_bytes":
        with bundle["proposal_path"].open("a", encoding="utf-8") as stream:
            stream.write(" ")
    elif kind in ("ack_role", "missing_message"):
        bundle["approval"]["human_ack"]["role" if kind == "ack_role" else "message_id"] = "assistant" if kind == "ack_role" else ""
        _write(bundle["approval_path"], bundle["approval"])
    else:
        p = bundle["proposal"]
        if kind == "expired":
            p["expires_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        elif kind == "retry":
            p["automatic_retry"] = True
        elif kind == "limits":
            p["limits"]["max_effective_updates"] = 100
        elif kind == "device":
            p["device"]["name"] = "cuda:0"
            # Do not use helper to construct a CUDA/real-role approval.
            _write(bundle["proposal_path"], p)
        elif kind == "config_escape":
            p["bindings"]["config"]["path"] = str(ROOT / "configs/B2-Pilot-W4.json")
        else:
            p["data_files_sha256"] = {str(ROOT / "configs/B2-Pilot-W4.json"): "0" * 64}
        if kind != "device":
            resign_synthetic_bundle(bundle)
    with pytest.raises(AuthorizationError):
        _verify(bundle)


def test_cheap_ack_does_not_hash_assets_until_cumulative_ledger_can_reserve(bundle):
    path = next(iter(bundle["proposal"]["data_files_sha256"]))
    Path(path).write_bytes(b"corrupted synthetic fixture")
    grant = read_acknowledgment(bundle["proposal_path"], bundle["approval_path"], **bundle["runtime"])
    with pytest.raises(AuthorizationError, match="hash mismatch"):
        verify_bound_files(grant)


def test_full_source_reverify_and_snapshot_map_consistency(bundle):
    grant = _verify(bundle)
    source = Path(next(iter(bundle["proposal"]["source_files_sha256"])))
    source.write_text("# changed synthetic source\n", encoding="utf-8")
    with pytest.raises(AuthorizationError, match="hash mismatch"):
        reverify_sources(grant)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows process creation identity required")
def test_oneuse_claim_is_durable_replay_denied_and_owner_creation_bound(bundle):
    grant = _verify(bundle)
    claimed = claim_authorization(grant)
    assert claimed["consumed"] is True and grant.claim_path.exists()
    assert_claim(grant, expected_parent_pid=os.getpid())
    with pytest.raises(AuthorizationError, match="already consumed"):
        claim_authorization(grant)
    payload = json.loads(grant.claim_path.read_text(encoding="utf-8"))
    payload["owner"]["creation_time_100ns"] = "1"
    _write(grant.claim_path, payload)
    with pytest.raises(AuthorizationError, match="creation identity"):
        assert_claim(grant, expected_parent_pid=os.getpid())


@pytest.mark.parametrize("shadow", ["src/torch.py", "src/shadow.pyc", "src/shadow.pyd", "scripts/shadow.dll",
                                  "third_party/HiFT/shadow.so", "src/shadow.PYC", "src/shadow.PYD", "src/shadow.PY"])
def test_exact_source_closure_rejects_new_shadow_module_without_minting_any_grant(tmp_path, shadow):
    from hift_research.wave6.authorization import assert_exact_project_source_set
    for name in ("src/hift_research/__init__.py", "src/hift_research/wave6/authorization.py",
                 "src/hift_research/wave6/lease.py", "scripts/run_b2_guarded.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source-set fixture only\n", encoding="utf-8")
    (tmp_path / "third_party/HiFT").mkdir(parents=True)
    files = {str(path): file_digest(path) for path in tmp_path.rglob("*.py")}
    assert_exact_project_source_set(tmp_path, files)
    (tmp_path / shadow).write_bytes(b"synthetic shadow bytes, never loaded or executed")
    with pytest.raises(AuthorizationError, match="executable source"):
        assert_exact_project_source_set(tmp_path, files)


@pytest.mark.parametrize("suffix", [".pyc", ".pyd"])
def test_declared_native_sourceless_files_still_rejected_by_metadata_only_policy(tmp_path, suffix):
    from hift_research.wave6.authorization import assert_exact_project_source_set
    for name in ("src/hift_research/__init__.py", "src/hift_research/wave6/authorization.py",
                 "src/hift_research/wave6/lease.py", "scripts/run_b2_guarded.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic source-set fixture only\n")
    (tmp_path / "third_party/HiFT").mkdir(parents=True)
    native = tmp_path / ("src/shadow" + suffix)
    native.write_bytes(b"synthetic inert native/sourceless bytes, never loaded")
    files = {str(path): file_digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(AuthorizationError, match="native/sourceless executable source"):
        assert_exact_project_source_set(tmp_path, files)


def test_source_set_ignores_existing_pycache_without_loading_its_bytecode(tmp_path):
    from hift_research.wave6.authorization import assert_exact_project_source_set
    for name in ("src/hift_research/__init__.py", "src/hift_research/wave6/authorization.py",
                 "src/hift_research/wave6/lease.py", "scripts/run_b2_guarded.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic source-set fixture only\n")
    (tmp_path / "third_party/HiFT").mkdir(parents=True)
    cached = tmp_path / "src/__pycache__/fixture.cpython-313.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"inert stale cache marker; launcher must use its fresh pycache prefix")
    files = {str(path): file_digest(path) for path in tmp_path.rglob("*.py")}
    assert_exact_project_source_set(tmp_path, files)


@pytest.mark.skipif(os.name != "nt", reason="native Windows atomic one-use race")
def test_concurrent_oneuse_claim_has_exactly_one_winner(bundle):
    code = """import sys,json
sys.path.insert(0,sys.argv[1])
from hift_research.wave6.authorization import verify_authorization,claim_authorization,AuthorizationError
p=json.load(open(sys.argv[2],encoding='utf-8'))
g=verify_authorization(sys.argv[2],sys.argv[3],actual_argv=p['argv'],actual_cwd=p['cwd'],actual_output=p['output_path'],gpu_uuid=p['device']['gpu_uuid'],allow_synthetic=True)
try:
 claim_authorization(g)
except AuthorizationError:
 sys.exit(19)
"""
    command = [sys.executable, "-c", code, str(ROOT / "src"), str(bundle["proposal_path"]), str(bundle["approval_path"])]
    processes = [subprocess.Popen(command), subprocess.Popen(command)]
    try:
        assert sorted(p.wait(timeout=15) for p in processes) == [0, 19]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    claims = list((bundle["namespace"] / "claims").glob("*.claim.json"))
    assert len(claims) == 1 and json.loads(claims[0].read_text(encoding="utf-8"))["consumed"] is True


@pytest.mark.skipif(os.name != "nt", reason="native Windows claim owner identity")
def test_postclaim_failure_keeps_invocation_consumed(bundle, monkeypatch):
    import hift_research.wave6.authorization as module
    grant = _verify(bundle)
    original, calls = module.verify_bound_files, []
    def fail_after_claim(value, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise AuthorizationError("synthetic postclaim source drift")
        return original(value, **kwargs)
    monkeypatch.setattr(module, "verify_bound_files", fail_after_claim)
    with pytest.raises(AuthorizationError, match="postclaim"):
        claim_authorization(grant)
    assert grant.claim_path.exists()
    monkeypatch.setattr(module, "verify_bound_files", original)
    with pytest.raises(AuthorizationError, match="already consumed"):
        claim_authorization(grant)


@pytest.mark.parametrize("identifier", ["../escape", "..", r"..\escape", r"C:\outside\x", "/outside",
                                       "folder/escape", "x:y", "NUL", "con", "COM1", "LPT9", "trailing.", "white space", ""])
def test_identifiers_are_safe_windows_basenames_metadata_only(identifier):
    from hift_research.wave6.authorization import validate_identifier
    for label in ("experiment_id", "proposal_id"):
        with pytest.raises(AuthorizationError):
            validate_identifier(identifier, label)
    assert validate_identifier("B2-W6-SYNTHETIC_001") == "B2-W6-SYNTHETIC_001"


def test_module_import_is_stdlib_only():
    script = "import sys;sys.path.insert(0,sys.argv[1]);import hift_research.wave6.authorization, hift_research.wave6.lease;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", script, str(ROOT / "src")], check=True, timeout=15)


def test_duplicate_json_keys_and_missing_scope_rejected(bundle):
    bundle["proposal_path"].write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(AuthorizationError, match="duplicate JSON"):
        _verify(bundle)
