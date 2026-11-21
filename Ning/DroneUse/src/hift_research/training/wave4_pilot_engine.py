"""One serial production controller: synthetic verification or separately authorized future real pilot.

Only committed effective-window boundaries can be saved. Mid-window faults taint
this instance: discard it, restore the last atomic boundary, replay the entire
indexed attempt window. No producer-position/prefetch replay claim.
"""
import copy
import math
import os
import time
import uuid
from pathlib import Path

import torch

from hift_research.provenance import sha256
from .graph import gradient_audit
from .lifecycle import EpochSchedule, actual_heads, apply_epoch_modes, lifecycle_optimizer, window_loss
from .recipe import ROOT, SOURCE_FILES, canonical_hash
from .wave4_lifecycle import cpu_tree, device_rng, device_tree, restore_device_rng
from .wave4_loader import SyntheticCandidateDataset, dataset_identity


def atomic_checkpoint(path, state):
    """Publish completed bytes atomically using no-overwrite hardlink, same filesystem."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    parent = path.parent.resolve()
    name = path.name+".partial-"+uuid.uuid4().hex
    temporary = parent/name
    with temporary.open("xb") as stream:
        torch.save(cpu_tree(state), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # atomic publication; raises rather than replaces an existing destination
    # Verify the exact absolute temporary target before removing our own staging link.
    if temporary.resolve() != parent/name or temporary.resolve().parent != parent:
        raise RuntimeError("unexpected temporary checkpoint target")
    temporary.unlink()
    return {"path": str(path), "sha256": sha256(path)}


class SerialPilotController:
    def __init__(self, model, dataset, *, config_identity, data_identity, device="cpu", gpu_token=None,
                 synthetic=True, physical=4, attempts_per_window=16, max_updates=2, max_attempts=32,
                 config_path=None, authorization_file=None, ack_real_sgd=False):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and not gpu_token):
            raise PermissionError("CPU or explicitly granted exclusive CUDA device required")
        if physical < 1 or attempts_per_window < physical or attempts_per_window % physical or max_attempts < 1:
            raise ValueError("invalid physical/window/attempt budget")
        if synthetic:
            if type(dataset) is not SyntheticCandidateDataset or data_identity.get("kind") != "synthetic" or not 1 <= max_updates <= 12:
                raise PermissionError("synthetic controller requires indexed SYNTHETIC dataset and bounded updates")
        else:
            from .wave4_pilot import future_authorization
            config, grant = future_authorization(config_path, authorization_file, ack_real_sgd=ack_real_sgd)
            from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
            if (not isinstance(dataset, TemporalCandidateDataset) or data_identity.get("kind") != "authorized_future_real"
                    or config_identity.get("config_sha256") != sha256(config_path) or gpu_token != grant["exclusive_gpu_token"]
                    or (physical, attempts_per_window, max_updates, max_attempts) != (4, 16, 100, 1600)):
                raise PermissionError("future real controller scope/identity/budget mismatch")
        if len(dataset) < max_attempts or any(data_identity.get(k) != v for k,v in dataset_identity(dataset).items()):
            raise ValueError("indexed dataset manifest/policy/reservations/budget mismatch")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("construct controller with fresh CPU model, then place before optimizer")
        self.model, self.dataset = model.to(self.device), dataset
        self.config_identity, self.data_identity = copy.deepcopy(config_identity), copy.deepcopy(data_identity)
        self.physical, self.attempts_per_window = physical, attempts_per_window
        self.max_updates, self.max_attempts, self.synthetic = max_updates, max_attempts, synthetic
        self.optimizer = lifecycle_optimizer(self.model)
        self.scheduler = EpochSchedule(self.optimizer)
        self.scheduler.set_epoch(0)
        apply_epoch_modes(self.model, 0)
        self.cursor = self.update = self.window = 0
        self.in_window, self.semantic_trace, self.measurements = False, [], []
        self.auxiliary_state = {}
        files = set(SOURCE_FILES)
        for folder in ("src/hift_research/training", "src/hift_research/corrections"):
            files.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT/folder).glob("*.py"))
        files.update(("src/hift_research/data_contract/torch_adapter.py", "third_party/HiFT/pysot/models/utile/tran.py",
                      "third_party/HiFT/pysot/models/utile/utile.py", "third_party/HiFT/pysot/models/backbone/newalexnet.py"))
        if not synthetic:
            files.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT/"src/hift_research/temporal_contract").glob("*.py"))
        self.code_identity = {name: sha256(ROOT/name) for name in sorted(files)}
        self.device_identity = {"device": str(self.device), "torch": str(torch.__version__), "tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
        if self.device.type == "cuda":
            self.device_identity.update({"gpu": torch.cuda.get_device_name(self.device), "cuda": torch.version.cuda})
            torch.cuda.reset_peak_memory_stats(self.device)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def timed(self, function):
        self.sync()
        start = time.perf_counter()
        value = function()
        self.sync()
        return value, time.perf_counter()-start

    def step(self, *, crash_after_micro=None, deadline_monotonic=None):
        if self.in_window:
            raise RuntimeError("uncommitted/tainted window: discard instance and restore prior boundary")
        if self.cursor >= self.max_attempts or self.update >= self.max_updates:
            return None
        from .wave4_pilot import buffered_attempt_window
        self.in_window = True  # no save/continued step permitted after any fault until a boundary is restored
        ids = list(range(self.cursor, min(self.cursor+self.attempts_per_window, self.max_attempts)))
        window_start = time.perf_counter()
        (batches, denominators), input_s = self.timed(lambda: buffered_attempt_window(
            self.dataset, ids, self.physical, kind="synthetic" if self.synthetic else "authorized_future_real"))
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, accepted = 0.0, sum(len(b["accepted_records"]) for b in batches)
        times = {"input_buffer_s": input_s, "h2d_s": 0.0, "forward_loss_s": 0.0, "backward_s": 0.0,
                 "gradient_audit_clip_s": 0.0, "optimizer_step_s": 0.0}
        for micro, cpu_batch in enumerate(batches, 1):
            if cpu_batch["accepted_records"]:
                batch, seconds = self.timed(lambda: device_tree(cpu_batch, self.device))
                times["h2d_s"] += seconds
                loss, seconds = self.timed(lambda: window_loss(actual_heads(self.model, batch), batch["targets"], denominators))
                times["forward_loss_s"] += seconds
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite window loss")
                _, seconds = self.timed(loss.backward)
                times["backward_s"] += seconds
                loss_sum += float(loss.detach())
            if deadline_monotonic is not None and time.perf_counter() > deadline_monotonic:
                raise TimeoutError("future pilot deadline reached; uncommitted window discarded")
            if crash_after_micro == micro:
                raise RuntimeError(f"injected midwindow interruption after micro={micro}; no effective SGD committed")
        audit, norm = None, None
        if accepted:
            def validate_and_clip():
                audit = gradient_audit(self.model)
                all_negative = all(r["candidate_negative"] for b in batches for r in b["accepted_records"])
                zero = [name for name in audit["unexpected_zero"] if not (all_negative and name.startswith("grader.convloc."))]
                if audit["unexpected_missing"] or audit["nonfinite"] or zero:
                    raise ValueError("unexpected active-gradient failure")
                norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient norm")
                return audit, norm
            (audit, norm), seconds = self.timed(validate_and_clip)
            times["gradient_audit_clip_s"] += seconds
            _, seconds = self.timed(self.optimizer.step)
            times["optimizer_step_s"] += seconds
            self.update += 1
        self.cursor += len(ids)  # COMMIT only after successful step or explicitly all-rejected complete window
        self.window += 1
        self.in_window = False
        row = {"attempt_ids": ids, "accepted": accepted, "rejected": len(ids)-accepted, "replacements": 0,
               "effective_accepted": accepted, "loss": loss_sum, "norm_before_clip": norm, "stepped": bool(accepted),
               "update": self.update, "window": self.window, "epoch": 0,
               "lr": [g["lr"] for g in self.optimizer.param_groups]}
        self.semantic_trace.append(row)
        times["window_total_s"] = time.perf_counter()-window_start
        memory = None
        if self.device.type == "cuda":
            memory = {"allocated": torch.cuda.memory_allocated(self.device), "reserved": torch.cuda.memory_reserved(self.device),
                      "peak_allocated": torch.cuda.max_memory_allocated(self.device), "peak_reserved": torch.cuda.max_memory_reserved(self.device),
                      "scope": "process CUDA allocator, may include other comparison models"}
        measurement = {"update": self.update, "window": self.window, "timings": times, "memory": memory, "gradient_audit": audit}
        self.measurements.append(measurement)
        return {"semantic": row, "measurement": measurement}

    def run_to_end(self):
        while self.step() is not None:
            pass
        return self

    def save_boundary(self, path):
        if self.in_window:
            raise RuntimeError("cannot checkpoint uncommitted/tainted window; restore previous boundary")
        self.sync()
        state = {"schema": "hift-wave4-serial-pilot-boundary-v1", "config_identity": self.config_identity,
                 "data_identity": self.data_identity, "code_identity": self.code_identity, "device_identity": self.device_identity,
                 "synthetic": self.synthetic, "physical": self.physical, "attempts_per_window": self.attempts_per_window,
                 "max_updates": self.max_updates, "max_attempts": self.max_attempts, "epoch": 0, "pending": None,
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                 "cursor": self.cursor, "update": self.update, "window": self.window, "semantic_trace": self.semantic_trace,
                 "rng": device_rng(self.device), "gradients": {n:p.grad for n,p in self.model.named_parameters()},
                 "auxiliary_state": copy.deepcopy(self.auxiliary_state)}
        return atomic_checkpoint(path, state)

    def restore_boundary(self, path, *, expected_sha256):
        if not expected_sha256 or sha256(path) != expected_sha256:
            raise ValueError("boundary checkpoint file hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"schema": "hift-wave4-serial-pilot-boundary-v1", "config_identity": self.config_identity,
                    "data_identity": self.data_identity, "code_identity": self.code_identity, "device_identity": self.device_identity,
                    "synthetic": self.synthetic, "physical": self.physical, "attempts_per_window": self.attempts_per_window,
                    "max_updates": self.max_updates, "max_attempts": self.max_attempts, "epoch": 0, "pending": None}
        if any(state.get(k) != v for k,v in expected.items()):
            raise ValueError("boundary config/data/source/device/clock/budget identity mismatch")
        target = self.model.state_dict()
        def valid(value, wanted):
            return (isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.shape == wanted.shape
                    and value.dtype == wanted.dtype and (not value.is_floating_point() or bool(torch.isfinite(value).all())))
        if state["model"].keys() != target.keys() or any(not valid(v,target[k]) for k,v in state["model"].items()):
            raise ValueError("boundary model key/shape/dtype/finite mismatch")
        groups = self.optimizer.state_dict()["param_groups"]
        if state["optimizer"]["param_groups"] != groups or state["scheduler"] != self.scheduler.state_dict():
            raise ValueError("boundary optimizer options/ordered IDs/LR or scheduler mismatch")
        params = {i:p for group,actual in zip(groups,self.optimizer.param_groups) for i,p in zip(group["params"],actual["params"])}
        names = {id(p): n for n,p in self.model.named_parameters()}
        needed_momentum = {i for i,p in params.items() if p.requires_grad and ".eles." not in names[id(p)]} if state["update"] else set()
        if set(state["optimizer"]["state"]) != needed_momentum:
            raise ValueError("boundary momentum coverage mismatch")
        for i, value in state["optimizer"]["state"].items():
            if set(value) != {"momentum_buffer"} or not valid(value["momentum_buffer"], params[i]):
                raise ValueError("boundary momentum mismatch")
        named = dict(self.model.named_parameters())
        if named.keys() != state["gradients"].keys():
            raise ValueError("boundary gradient keys mismatch")
        for n,v in state["gradients"].items():
            if v is not None and (not valid(v,named[n]) or not named[n].requires_grad or ".eles." in n):
                raise ValueError("boundary gradient mismatch")
        trace = state["semantic_trace"]
        consumed = [i for row in trace for i in row["attempt_ids"]]
        if (consumed != list(range(state["cursor"])) or not 0 <= state["cursor"] <= self.max_attempts
                or state["update"] != sum(row["stepped"] for row in trace) or not 0 <= state["update"] <= self.max_updates
                or state["window"] != len(trace) or state["window"] > math.ceil(self.max_attempts/self.attempts_per_window)):
            raise ValueError("boundary consumed cursor/window/update mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for name,p in named.items():
            gradient = state["gradients"][name]
            p.grad = gradient.to(self.device).clone() if gradient is not None else None
        self.cursor, self.update, self.window = state["cursor"], state["update"], state["window"]
        self.semantic_trace, self.in_window, self.measurements = trace, False, []
        self.auxiliary_state = copy.deepcopy(state["auxiliary_state"])
        apply_epoch_modes(self.model, 0)
        restore_device_rng(state["rng"], self.device)
        return self
