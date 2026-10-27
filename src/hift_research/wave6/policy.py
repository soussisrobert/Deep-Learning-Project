"""Explicit operational policies. Cooperative ownership is NOT physical exclusivity.

The scientific recipe and existing resource caps are unchanged. Missing policy
bindings always mean the historical strict policy, never cooperative permission.
"""
from pathlib import Path
import hashlib
import json

STRICT_PHYSICAL_EXCLUSIVITY = "STRICT_PHYSICAL_EXCLUSIVITY"
COOPERATIVE_GUARDED_GPU = "COOPERATIVE_GUARDED_GPU"
BASE_RESOURCES = {"host_process_peak_rss_bytes": 8 * 1024**3,
                  "minimum_available_host_bytes": 4 * 1024**3,
                  "max_cuda_reserved_fraction": .8}
COOPERATIVE_SPEC = {
    **BASE_RESOURCES,
    "minimum_launch_gpu_free_bytes": 4 * 1024**3,
    "minimum_runtime_gpu_free_bytes": 2 * 1024**3,
    "sample_interval_seconds": 1,
    "native_query_timeout_seconds": 2,
    "maximum_sample_age_seconds": 5,
    "worker_heartbeat_interval_seconds": .5,
    "worker_startup_grace_seconds": 30,
    "identity_and_lease_validation": "every_resource_sample_and_every_safe_boundary",
    "external_process_observation": "native_SMI_at_launch_and_release;raw_PID_type_name_memory;no_kind_inference",
    "unknown_external_contexts": "permitted_not_classified;not_hardware_wide_exclusivity",
    "untrusted_resource_measurement": "stop_fail_closed_no_retry",
}
STRICT_SPEC = {**BASE_RESOURCES, "context_classification": "original_strict_conditions_no_opaque_or_unowned_compute"}
IMPLEMENTATION_FILES = tuple("src/hift_research/wave6/" + name + ".py" for name in
    ("policy", "authorization", "lease", "entry", "worker", "runtime", "adapter", "monitor", "supervisor"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def policy_binding(name=STRICT_PHYSICAL_EXCLUSIVITY, root=None, *, coordination=None):
    if name not in (STRICT_PHYSICAL_EXCLUSIVITY, COOPERATIVE_GUARDED_GPU):
        raise ValueError("unknown operational GPU policy")
    if coordination not in (None, 'none') or (coordination == 'none' and name != COOPERATIVE_GUARDED_GPU):
        raise ValueError("unknown or incompatible GPU coordination mode")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    paths = IMPLEMENTATION_FILES + (('src/hift_research/wave6/direct_cuda.py',) if coordination == 'none' else ())
    files = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
    specification = dict(COOPERATIVE_SPEC if name == COOPERATIVE_GUARDED_GPU else STRICT_SPEC)
    version = 'cooperative-v1' if name == COOPERATIVE_GUARDED_GPU else 'strict-v1'
    if coordination == 'none':
        version = 'cooperative-direct-cuda-v1'
        specification.pop('identity_and_lease_validation')
        specification.update(gpu_coordination='none',
            identity_validation='one_use_claim_and_parent_capability_every_resource_sample_and_safe_boundary',
            external_process_observation='native_SMI_at_launch_and_final_observation;raw_PID_type_name_memory;no_kind_inference')
    binding = {"name": name, "version": "hift-gpu-policy-" + version,
               "specification": specification,
               "implementation_files_sha256": files, "implementation_sha256": _digest(files)}
    return {**binding, "policy_sha256": _digest(binding)}


def validate_policy_binding(proposal):
    binding = proposal.get("operational_policy")
    if binding is None:
        return STRICT_PHYSICAL_EXCLUSIVITY
    if not isinstance(binding, dict) or binding.get("name") not in (STRICT_PHYSICAL_EXCLUSIVITY, COOPERATIVE_GUARDED_GPU):
        raise ValueError("invalid operational policy binding")
    # Resolve against this executing checkout, not an attacker-selected cwd.
    expected = policy_binding(binding["name"], coordination=proposal.get('gpu_coordination'))
    if _digest(binding) != _digest(expected):
        raise ValueError("operational policy specification/version/implementation/hash mismatch")
    return binding["name"]
