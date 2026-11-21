"""Stdlib-only exact future-human-ACK binding; this module never grants authority.

Message references are transport attestations, NOT cryptographic proof of a human.
The engineering tests use explicitly synthetic temporary namespaces exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

PROPOSAL_SCHEMA = "hift-wave6-exact-proposal-v1"
ACK_SCHEMA = "hift-wave6-exact-human-ack-v1"
OWNER_SCHEMA = "hift-local-owner-authorization-v1"
OWNER_ORIGINAL_SHA256 = "45d04e2d6738cdd4a40f6e74c0a1db5c159652d8977a14c246ac4e506982e7a5"
CONFIG_SHA256 = "420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e"
REAL_LIMITS = {"max_effective_updates": 100, "max_training_attempts": 1600, "max_windows": 100, "max_wall_seconds": 900}
RESOURCE_LIMITS = {"host_process_peak_rss_bytes": 8 * 1024**3, "minimum_available_host_bytes": 4 * 1024**3, "max_cuda_reserved_fraction": .8}
REAL_INITIALIZATION = {
    "all_parameters_sha256": "52e80c8edd422bf78e0aa09ef1b9bba998f4b91ea3842863fa1c0d2225d20130",
    "backbone_state_sha256": "c3f2d95caf15d49a922b464568e1cb6fbfa30f6922c793486b54c676e1a99e5e",
    "grader_parameters_sha256": "f6d0b47dcbfbc706bc524e1693acf36bdf2ca8a59aa52ab6cd29fa5e9b865a36",
}
REAL_SCOPE = {"kind": "first_authentic_corrected_b2_grader_reliability", "seed": 20261023, "epoch": 0,
              "trainable": "entire_corrected_grader_only", "backbone": "frozen_imagenet_all_bn_frozen", "config_sha256": CONFIG_SHA256}
SHA = re.compile(r"^[0-9a-f]{64}$")
GPU = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class AuthorizationError(PermissionError):
    pass


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizationError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object_pairs,
                           parse_constant=lambda v: (_ for _ in ()).throw(AuthorizationError("nonfinite JSON")))
    except AuthorizationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise AuthorizationError("unreadable exact authorization input: " + str(path)) from exc
    if not isinstance(value, dict):
        raise AuthorizationError("JSON object required")
    return value


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise AuthorizationError("explicit timezone-aware UTC timestamp required") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise AuthorizationError("explicit UTC timestamp required")
    return parsed.astimezone(timezone.utc)


def _absolute(value):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise AuthorizationError("explicit absolute bound path required")
    return Path(value).resolve()


def validate_identifier(value, label="identifier"):
    """IDs become Windows basenames: never paths, drive names or devices."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise AuthorizationError("unsafe Windows basename for " + label)
    upper = value.upper()
    if upper in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", upper):
        raise AuthorizationError("reserved Windows device basename for " + label)
    return value


def _sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise AuthorizationError("full lowercase SHA256 required")
    return value


def _mapping(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise AuthorizationError("nonempty full file-hash map required")
    for name, digest in mapping.items():
        if not isinstance(name, str) or not name:
            raise AuthorizationError("file path required")
        _sha(digest)
    return mapping


def _bound_path(cwd, name):
    path = Path(name)
    return (path if path.is_absolute() else cwd / path).resolve()


def synthetic_namespace(path):
    root = _absolute(str(path))
    temporary = Path(tempfile.gettempdir()).resolve()
    if root == temporary or temporary not in root.parents or not root.name.startswith("hift-wave6-synthetic-"):
        raise AuthorizationError("synthetic fixtures require a dedicated hift-wave6-synthetic- TEMP namespace")
    return root


def expected_policy_decision(proposal, proposal_sha256):
    from .policy import COOPERATIVE_GUARDED_GPU
    policy = proposal.get("operational_policy", {})
    if policy.get("name") != COOPERATIVE_GUARDED_GPU:
        raise AuthorizationError("cooperative decision requires explicit Policy B binding")
    prefix = "SYNTHETIC TEST ONLY: " if proposal.get("synthetic_only") is True else ""
    return (prefix + "I accept COOPERATIVE_GUARDED_GPU for this specific experiment, acknowledging that unrelated applications may use the GPU "
            "and that this is not hardware-wide exclusivity. proposal_sha256=" + _sha(proposal_sha256)
            + " policy_sha256=" + _sha(policy["policy_sha256"]))


def expected_training_decision(proposal, proposal_sha256):
    prefix = "SYNTHETIC TEST ONLY NO REAL SGD: " if proposal.get("synthetic_only") is True else ""
    limits = proposal["limits"]
    return (prefix + "I authorize one authentic corrected-B2 grader-only pilot under the exact proposal digest "
            + _sha(proposal_sha256) + ", limited to " + str(limits["max_effective_updates"]) + " effective updates, "
            + format(limits["max_training_attempts"], ",") + " attempts and " + str(limits["max_wall_seconds"])
            + " cumulative seconds. NO_RESUME_NO_RETRY_NO_SCOPE_EXPANSION")


def expected_ack(proposal, proposal_sha256):
    """Return exact future ACK text; returning text does NOT assert it was received."""
    from .policy import COOPERATIVE_GUARDED_GPU
    if proposal.get("operational_policy", {}).get("name") == COOPERATIVE_GUARDED_GPU:
        # These are the two literal user decisions, not a third hidden approval.
        # The complete proposal SHA binds argv/scope/resources/assets/expiry.
        return expected_policy_decision(proposal, proposal_sha256) + "\n" + expected_training_decision(proposal, proposal_sha256)
    prefix = "ACK SYNTHETIC TEST ONLY NO REAL SGD" if proposal.get("synthetic_only") is True else "ACK REAL SGD EXACTLY ONCE"
    if proposal.get("operational_policy") is not None:
        prefix += " | policy_sha256=" + _sha(proposal["operational_policy"]["policy_sha256"])
    return " | ".join((prefix, "experiment_id=" + proposal["experiment_id"], "proposal_sha256=" + _sha(proposal_sha256),
        "proposal_digest=" + canonical_digest(proposal), "command_sha256=" + canonical_digest(proposal["argv"]),
        "output_path=" + json.dumps(proposal["output_path"], ensure_ascii=False), "gpu_uuid=" + proposal["device"]["gpu_uuid"],
        "expires_utc=" + proposal["expires_utc"], "scope_sha256=" + canonical_digest(proposal["scope"]),
        "resources_sha256=" + canonical_digest(proposal["resources"]), "max_invocations=1", "NO_RESUME_NO_RETRY_NO_SCOPE_EXPANSION"))


def validate_local_owner(p, a, current):
    """Trusted local transport attestation, NOT a signature or a literal legacy ACK.

    This narrow administrative rebind covers only the owner's already frozen pilot.
    Runtime authority, one-use accounting, science and resource checks are shared.
    """
    from .policy import COOPERATIVE_GUARDED_GPU
    if p["synthetic_only"] or p.get("operational_policy", {}).get("name") != COOPERATIVE_GUARDED_GPU:
        raise AuthorizationError("local owner path requires the real frozen Policy B pilot")
    original_ref = p["bindings"]["owner_original_proposal"]
    instruction_ref = p["bindings"]["owner_instruction"]
    for ref in (original_ref, instruction_ref):
        if file_digest(_absolute(ref["path"])) != _sha(ref["sha256"]):
            raise AuthorizationError("local owner authority binding changed")
    if original_ref["sha256"] != OWNER_ORIGINAL_SHA256:
        raise AuthorizationError("local owner authority requires the original frozen proposal")
    old, instruction = _read(original_ref["path"]), _read(instruction_ref["path"])
    if (instruction.get("schema") != "hift-local-owner-instruction-v1"
            or instruction.get("role") != "user"
            or instruction.get("authority_kind") != "trusted_local_owner_transport_attestation_not_signature"
            or instruction.get("original_proposal_sha256") != OWNER_ORIGINAL_SHA256
            or a.get("owner_instruction_sha256") != instruction_ref["sha256"]
            or a.get("policy_sha256") != p["operational_policy"]["policy_sha256"]
            or any(not isinstance(instruction.get(k), str) or not instruction[k].strip()
                   for k in ("session_id", "message_id", "text"))):
        raise AuthorizationError("concrete local owner instruction and exact policy binding required")
    if not _time(old["created_utc"]) <= _time(instruction["received_utc"]) <= _time(p["created_utc"]) <= current < _time(p["expires_utc"]):
        raise AuthorizationError("local owner instruction/rebind expired or future")
    if p.get("gpu_coordination") is not None:
        from .direct_cuda import validate_continuation
        validate_continuation(p, old, instruction)
        return instruction
    administrative = {"created_utc", "expires_utc", "argv", "approval_path", "bindings",
                      "source_files_sha256", "operational_policy", "approval_rule"}
    if ({k: v for k, v in p.items() if k not in administrative}
            != {k: v for k, v in old.items() if k not in administrative}):
        raise AuthorizationError("local owner rebind changed frozen pilot scope")
    allowed_sources = {"src/hift_research/wave6/authorization.py", "tests/test_local_owner_authorization.py"}
    before, after = old["source_files_sha256"], p["source_files_sha256"]
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if not changed <= allowed_sources:
        raise AuthorizationError("local owner rebind changed non-administrative executable source")
    return instruction


@dataclass(frozen=True)
class VerifiedAuthorization:
    proposal_path: Path
    approval_path: Path
    proposal: dict[str, Any]
    approval: dict[str, Any]
    proposal_sha256: str
    proposal_digest: str
    authorization_digest: str
    experiment_id: str
    output_path: Path
    expires_utc: datetime
    gpu_uuid: str
    claim_path: Path
    synthetic_only: bool


def read_acknowledgment(proposal_path, approval_path, *, actual_argv, actual_cwd, actual_output, gpu_uuid,
                        now_utc=None, allow_synthetic=False):
    """Cheap preflight: read only proposal/ACK, never asset maps or Torch imports.

    The caller starts its monotonic timer BEFORE this call and reserves its durable
    cumulative-time ledger after this succeeds, before verify_bound_files().
    """
    pp, ap = _absolute(str(proposal_path)), _absolute(str(approval_path))
    p, a = _read(pp), _read(ap)
    try:
        local_owner = a["schema"] == OWNER_SCHEMA
        if p["schema"] != PROPOSAL_SCHEMA or a["schema"] not in (ACK_SCHEMA, OWNER_SCHEMA):
            raise AuthorizationError("legacy/self-approved authorization envelopes are forbidden")
        if type(p["synthetic_only"]) is not bool:
            raise AuthorizationError("explicit synthetic_only boolean required")
        test = p["synthetic_only"]
        if test and not allow_synthetic:
            raise AuthorizationError("synthetic ACK cannot authorize production")
        cwd, output = _absolute(p["cwd"]), _absolute(p["output_path"])
        if cwd != Path(actual_cwd).resolve() or output != Path(actual_output).resolve():
            raise AuthorizationError("exact command cwd/output mismatch")
        if not isinstance(p["argv"], list) or not p["argv"] or any(not isinstance(v, str) or not v for v in p["argv"]):
            raise AuthorizationError("full exact argv list required")
        if p["argv"] != list(actual_argv):
            raise AuthorizationError("exact full command mismatch")
        for flag, expected in (("--proposal", str(pp)), ("--authorization", str(ap)), ("--output", str(output))):
            if p["argv"].count(flag) != 1 or p["argv"].index(flag) + 1 >= len(p["argv"]):
                raise AuthorizationError("one exact bound CLI path flag required: " + flag)
            if _absolute(p["argv"][p["argv"].index(flag) + 1]) != Path(expected):
                raise AuthorizationError("command path differs from bound proposal: " + flag)
        if _absolute(p["approval_path"]) != ap:
            raise AuthorizationError("approval file path mismatch")
        validate_identifier(p["experiment_id"], "experiment_id")
        validate_identifier(p["proposal_id"], "proposal_id")
        if type(p["max_invocations"]) is not int or p["max_invocations"] != 1:
            raise AuthorizationError("exactly one invocation required")
        for key in ("resume_allowed", "automatic_retry", "scope_expansion"):
            if p[key] is not False:
                raise AuthorizationError("resume/retry/scope expansion forbidden")
        from .policy import validate_policy_binding, COOPERATIVE_GUARDED_GPU
        try:
            policy_name = validate_policy_binding(p)
        except (ValueError, OSError) as exc:
            raise AuthorizationError("unreviewed operational policy: " + str(exc)) from exc
        if p["resources"] != RESOURCE_LIMITS:
            raise AuthorizationError("unreviewed resource limits")
        limits = p["limits"]
        if set(limits) != set(REAL_LIMITS) or any(type(v) is not int or v <= 0 for v in limits.values()):
            raise AuthorizationError("full positive exact limit fields required")
        direct = p.get("gpu_coordination") is not None
        if direct and (not local_owner or test or p["gpu_coordination"] != "none"):
            raise AuthorizationError("direct CUDA mode requires explicit real local-owner authority")
        if not test and not direct and limits != REAL_LIMITS:
            raise AuthorizationError("production limits must be exactly100/1600/100/900")
        if direct and (any(limits[k] != cap for k, cap in REAL_LIMITS.items() if k != "max_wall_seconds")
                       or limits["max_wall_seconds"] >= 900):
            raise AuthorizationError("direct continuation must preserve science caps and debit prior time")
        if test and any(limits[k] > cap for k, cap in {"max_effective_updates": 2, "max_training_attempts": 32, "max_windows": 2, "max_wall_seconds": 30}.items()):
            raise AuthorizationError("synthetic CPU test bounds exceeded")
        if p["device"]["name"] != ("cpu" if test else "cuda:0") or not GPU.fullmatch(p["device"]["gpu_uuid"]):
            raise AuthorizationError("exact device name and full physical UUID required")
        if p["device"]["gpu_uuid"] != gpu_uuid:
            raise AuthorizationError("physical device UUID mismatch")
        for key in ("source_files_sha256", "data_files_sha256", "initialization_files_sha256"):
            _mapping(p[key])
        for key in ("config", "source_inventory", "data_manifest", "initialization_manifest"):
            record = p["bindings"][key]
            _absolute(record["path"])
            _sha(record["sha256"])
        if not test and p["bindings"]["config"]["sha256"] != CONFIG_SHA256:
            raise AuthorizationError("frozen Wave4 config changed")
        if not isinstance(p["scope"], dict) or not p["scope"]:
            raise AuthorizationError("explicit complete scientific scope required")
        for key in ("all_parameters_sha256", "backbone_state_sha256", "grader_parameters_sha256"):
            _sha(p["scope"]["initialization"][key])
        if test and p["scope"].get("synthetic_random_only") is not True:
            raise AuthorizationError("synthetic authorization is CPU random fixtures only")
        if not test and (any(p["scope"].get(k) != v or type(p["scope"].get(k)) is not type(v) for k, v in REAL_SCOPE.items())
                         or p["scope"]["initialization"] != REAL_INITIALIZATION):
            raise AuthorizationError("production scientific recipe/initialization contradicts frozen B2 scope")
        current = now_utc or datetime.now(timezone.utc)
        if isinstance(current, str):
            current = _time(current)
        if current.tzinfo is None:
            raise AuthorizationError("aware verification clock required")
        created, expiry = _time(p["created_utc"]), _time(p["expires_utc"])
        if local_owner:
            validate_local_owner(p, a, current)
        else:
            ack = a["human_ack"]
            if ack["role"] != ("synthetic_test_user" if test else "user"):
                raise AuthorizationError("human user ACK transport role required; not a self-approval boolean")
            if any(not isinstance(ack[k], str) or not ack[k].strip() for k in ("session_id", "message_id")):
                raise AuthorizationError("concrete human message/session reference required")
            if not created <= _time(ack["received_utc"]) <= current < expiry:
                raise AuthorizationError("ACK/proposal expired, future, or predates proposal")
        psha, pdigest, asha = file_digest(pp), canonical_digest(p), file_digest(ap)
        if a["proposal_sha256"] != psha or a["proposal_digest"] != pdigest or a["expires_utc"] != p["expires_utc"]:
            raise AuthorizationError("human ACK bound to a different complete proposal")
        if not isinstance(a["approval_id"], str) or not a["approval_id"] or type(a["max_invocations"]) is not int or a["max_invocations"] != 1:
            raise AuthorizationError("one-use approval identity required")
        if not local_owner and ack["text"] != expected_ack(p, psha):
            raise AuthorizationError("exact human ACK text mismatch")
        if not local_owner and policy_name == COOPERATIVE_GUARDED_GPU:
            if (ack.get("operational_policy_decision") != expected_policy_decision(p, psha)
                    or ack.get("training_authorization_decision") != expected_training_decision(p, psha)
                    or a.get("policy_sha256") != p["operational_policy"]["policy_sha256"]):
                raise AuthorizationError("two exact human decisions and cooperative policy hash required")
        claim_dir = _absolute(p["claim_directory"])
        if test:
            namespace = synthetic_namespace(p["synthetic_namespace"])
            confined = [pp, ap, cwd, output, claim_dir]
            confined += [_absolute(p["bindings"][k]["path"]) for k in ("config", "source_inventory", "data_manifest", "initialization_manifest")]
            confined += [_bound_path(cwd, name) for key in ("data_files_sha256", "initialization_files_sha256") for name in p[key]]
            if any(path != namespace and namespace not in path.parents for path in confined):
                raise AuthorizationError("synthetic fixture may not bind project real grants/assets/output")
        return VerifiedAuthorization(pp, ap, p, a, psha, pdigest, asha, p["experiment_id"], output,
                                     expiry, gpu_uuid, claim_dir / (pdigest + ".claim.json"), test)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, AuthorizationError):
            raise
        raise AuthorizationError("incomplete exact proposal/ACK schema") from exc


def _same_envelope(grant):
    if file_digest(grant.proposal_path) != grant.proposal_sha256 or file_digest(grant.approval_path) != grant.authorization_digest:
        raise AuthorizationError("proposal or ACK bytes changed after verification")
    if canonical_digest(grant.proposal) != grant.proposal_digest:
        raise AuthorizationError("verified proposal object mutated")
    expected_claim = _absolute(grant.proposal["claim_directory"]) / (grant.proposal_digest + ".claim.json")
    if grant.claim_path != expected_claim:
        raise AuthorizationError("one-use claim namespace differs from exact approved proposal")
    if datetime.now(timezone.utc) >= grant.expires_utc:
        raise AuthorizationError("authorization expired during guard")


def _verify_map(grant, mapping):
    for name, expected in mapping.items():
        try:
            actual = file_digest(_bound_path(Path(grant.proposal["cwd"]), name))
        except OSError as exc:
            raise AuthorizationError("bound source/asset unavailable: " + name) from exc
        if actual != expected:
            raise AuthorizationError("full bound content hash mismatch: " + name)


def assert_exact_project_source_set(cwd, mapping):
    """Reject new executable shadow modules, not merely changed listed bytes."""
    root = Path(cwd).resolve()
    roots = [root / name for name in ("src", "scripts", "third_party/HiFT")]
    actual = set()
    for directory in roots:
        if not directory.is_dir():
            raise AuthorizationError("complete project source root missing")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise AuthorizationError("unapproved executable source symlink/junction")
            suffix = path.suffix.lower()
            if path.is_file() and (suffix in {".pyc", ".pyo", ".pyd", ".dll", ".so", ".dylib"}
                                   or ".so." in path.name.lower()):
                # No project native/sourceless modules are required or permitted,
                # even if someone adds their bytes to a declared source map.
                # Runtime dependencies live outside these project source roots.
                raise AuthorizationError("native/sourceless executable source forbidden: " + str(path))
            if suffix == ".py" and path.is_file():
                actual.add(path.absolute())
    bound = {(Path(name) if Path(name).is_absolute() else root / name).absolute()
             for name in mapping if Path(name).suffix.lower() == ".py"}
    in_roots = {path for path in bound if any(directory in path.parents for directory in roots)}
    if actual != in_roots:
        raise AuthorizationError("unexpected/missing project executable source; full exact source-set closure required")
    required = {root / "src/hift_research/__init__.py", root / "scripts/run_b2_guarded.py",
                root / "src/hift_research/wave6/authorization.py", root / "src/hift_research/wave6/lease.py"}
    if not required.issubset(bound):
        raise AuthorizationError("incomplete Wave6 bootstrap/worker/freeze source closure")
    return True


def reverify_sources(grant):
    _same_envelope(grant)
    p = grant.proposal
    _verify_map(grant, p["source_files_sha256"])
    for record in p["bindings"].values():
        if file_digest(record["path"]) != record["sha256"]:
            raise AuthorizationError("bound config/inventory/manifest changed")
    inventory = _read(p["bindings"]["source_inventory"]["path"])
    mapping = inventory.get("source_files_sha256", inventory.get("file_sha256", inventory))
    if mapping != p["source_files_sha256"]:
        raise AuthorizationError("full source inventory differs from approved source map")
    if not grant.synthetic_only:
        assert_exact_project_source_set(p["cwd"], p["source_files_sha256"])
    return True


def verify_bound_files(grant, *, verify_assets=True):
    reverify_sources(grant)
    if verify_assets:
        for key, binding in (("data_files_sha256", "data_manifest"), ("initialization_files_sha256", "initialization_manifest")):
            _verify_map(grant, grant.proposal[key])
            snapshot = _read(grant.proposal["bindings"][binding]["path"])
            if snapshot.get(key, snapshot.get("file_sha256")) != grant.proposal[key]:
                raise AuthorizationError("full asset snapshot/map closure mismatch:" + key)
        initialization = _read(grant.proposal["bindings"]["initialization_manifest"]["path"])
        if initialization.get("initialization") != grant.proposal["scope"]["initialization"]:
            raise AuthorizationError("initialization manifest tensor identities differ from approved scope")
    return grant


def verify_authorization(*args, verify_inputs=True, **kwargs):
    grant = read_acknowledgment(*args, **kwargs)
    return verify_bound_files(grant) if verify_inputs else grant


def claim_authorization(grant):
    """Atomically consume one invocation permanently, including failed attempts."""
    from .lease import process_identity
    verify_bound_files(grant)
    grant.claim_path.parent.mkdir(parents=True, exist_ok=True)
    identity = process_identity(os.getpid())
    record = {"schema": "hift-wave6-oneuse-claim-v1", "consumed": True, "experiment_id": grant.experiment_id,
              "proposal_digest": grant.proposal_digest, "proposal_sha256": grant.proposal_sha256,
              "authorization_digest": grant.authorization_digest, "owner": identity,
              "claimed_utc": datetime.now(timezone.utc).isoformat(), "max_invocations": 1,
              "synthetic_only": grant.synthetic_only, "automatic_retry": False}
    try:
        descriptor = os.open(grant.claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise AuthorizationError("one-use authorization already consumed; no retry/resume") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # A failure after atomic creation intentionally leaves the invocation consumed.
    verify_bound_files(grant)
    return {"path": str(grant.claim_path), "sha256": file_digest(grant.claim_path), **record}


def assert_claim(grant, *, expected_parent_pid=None):
    from .lease import process_identity
    _same_envelope(grant)
    record = _read(grant.claim_path)
    if (record.get("schema") != "hift-wave6-oneuse-claim-v1" or record.get("consumed") is not True
            or record.get("authorization_digest") != grant.authorization_digest
            or record.get("proposal_digest") != grant.proposal_digest
            or record.get("proposal_sha256") != grant.proposal_sha256
            or record.get("experiment_id") != grant.experiment_id or type(record.get("max_invocations")) is not int
            or record.get("max_invocations") != 1 or record.get("synthetic_only") is not grant.synthetic_only
            or record.get("automatic_retry") is not False):
        raise AuthorizationError("exact consumed parent claim missing/mismatched")
    pid = expected_parent_pid if expected_parent_pid is not None else record["owner"]["pid"]
    if record["owner"] != process_identity(pid):
        raise AuthorizationError("one-use claim owner PID/creation identity mismatch")
    return record
