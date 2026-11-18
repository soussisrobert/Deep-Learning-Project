"""New device-aware lifecycle; W3 source remains immutable. No implicit real-SGD authority."""
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch

from hift_research.provenance import sha256
from .lifecycle import (SyntheticLifecycle, SyntheticSampler, actual_heads, apply_epoch_modes,
                        capture_rng, lifecycle_optimizer, restore_rng, window_denominators, window_loss, EpochSchedule)
from .recipe import ROOT, canonical_hash, source_settings


def seed_device(seed, device, *, gpu_token=None):
    device = torch.device(device)
    if device.type == "cuda" and not gpu_token:
        raise ValueError("explicit lead GPU token required before CUDA use")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("CPU/CUDA only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    return device


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {key: device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(device_tree(item, device) for item in value)
    return value


def device_rng(device):
    state = capture_rng()
    state["cuda_used"] = device.type == "cuda"
    state["device"] = str(device)
    if device.type == "cuda":
        state["torch_cuda"] = [torch.cuda.get_rng_state(device).cpu().clone()]
    return state


def restore_device_rng(state, device):
    if state["cuda_used"] != (device.type == "cuda") or state["device"] != str(device):
        raise ValueError("checkpoint RNG device policy mismatch")
    cpu = dict(state, cuda_used=False, torch_cuda=[])
    restore_rng(cpu)
    if device.type == "cuda":
        if len(state["torch_cuda"]) != 1:
            raise ValueError("exactly one used CUDA device RNG required")
        torch.cuda.set_rng_state(state["torch_cuda"][0], device)


class Wave4SyntheticLifecycle(SyntheticLifecycle):
    """Same partition/input ordering only; CPU pending inputs + explicit H2D each micro.

    Uses proven W3 mathematical helpers, but new checkpoint code handles CUDA RNG,
    momentum and pending gradients. Real image factories are unconditionally refused.
    The future real pilot uses a separate explicit authorization gate/runner.
    """
    def __init__(self, model, sampler, *, batch_factory, config_identity, data_identity,
                 microbatch=4, accumulation=4, epoch=0, device="cpu", gpu_token=None,
                 max_updates=2, heads=actual_heads):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not gpu_token:
            raise ValueError("lead GPU token required")
        if max_updates < 1 or max_updates > 12:
            raise ValueError("bounded synthetic path requires1..12 effective updates")
        super().__init__(model, sampler, batch_factory=batch_factory, config_identity=config_identity,
                         data_identity=data_identity, microbatch=microbatch, accumulation=accumulation,
                         epoch=epoch, heads=heads)
        self.model.to(self.device)
        # Construct AFTER device placement; never retain references to replaced CPU Parameters.
        self.optimizer = lifecycle_optimizer(self.model, self.settings)
        self.scheduler = EpochSchedule(self.optimizer, self.settings)
        self.scheduler.set_epoch(epoch)
        self.max_updates, self.gpu_token = max_updates, gpu_token
        for relative in ("src/hift_research/training/wave4_lifecycle.py", "src/hift_research/training/wave4_loader.py"):
            self.code_identity[relative] = sha256(ROOT/relative)
        self.device_identity = {"device": str(self.device), "torch": str(torch.__version__),
                                "dtype": str(next(self.model.parameters()).dtype), "tf32": False}
        if self.device.type == "cuda":
            self.device_identity["gpu"] = torch.cuda.get_device_name(self.device)

    def microstep(self):
        if self.pending is None:
            if self.update >= self.max_updates:
                if self.sampler.position != len(self.sampler.order):
                    raise ValueError("synthetic update budget exhausted before sampler end")
                return None
            if not self.begin_window():
                return None
        pending = self.pending
        cpu_batch = pending["batches"][pending["index"]]
        if cpu_batch.get("data_kind") != "synthetic":
            raise ValueError("real arrays may not enter Wave4 synthetic SGD")
        batch = device_tree(cpu_batch, self.device)
        loss = window_loss(self.heads(self.model, batch), batch["targets"], pending["denominators"])
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
            raise ValueError("nonfinite synthetic gradient")
        self.microsteps += 1
        pending["index"] += 1
        stepped = pending["index"] == len(pending["batches"])
        norm = None
        if stepped:
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings["TRAIN"]["GRAD_CLIP"]))
            if not np.isfinite(norm):
                raise ValueError("nonfinite clipping norm")
            self.optimizer.step()
            self.update += 1
        row = {"sample_ids": batch["sample_ids"], "loss_contribution": float(loss.detach()), "epoch": self.epoch,
               "update": self.update, "microstep": self.microsteps, "stepped": stepped, "preclip_norm": norm,
               "lr": [group["lr"] for group in self.optimizer.param_groups]}
        self.trace.append(row)
        if stepped:
            self.pending = None
        return row

    def save(self, path):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        state = {"schema": "hift-wave4-synthetic-device-lifecycle-v1", "config_identity": self.config_identity,
                 "data_identity": self.data_identity, "code_identity": self.code_identity,
                 "device_identity": self.device_identity, "settings_hash": canonical_hash(self.settings),
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "epoch": self.epoch, "update": self.update,
                 "microsteps": self.microsteps, "sampler": self.sampler.state_dict(), "pending": self.pending,
                 "trace": self.trace, "rng": device_rng(self.device), "max_updates": self.max_updates,
                 "microbatch": self.microbatch, "accumulation": self.accumulation,
                 "gradients": {name: p.grad if p.grad is not None else None for name, p in self.model.named_parameters()}}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(cpu_tree(state), stream)

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"schema": "hift-wave4-synthetic-device-lifecycle-v1", "config_identity": self.config_identity,
                    "data_identity": self.data_identity, "code_identity": self.code_identity,
                    "device_identity": self.device_identity, "settings_hash": canonical_hash(self.settings),
                    "microbatch": self.microbatch, "accumulation": self.accumulation, "max_updates": self.max_updates}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint source/config/data/device/budget identity mismatch")
        epoch = state["epoch"]
        if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < 70
                or state["scheduler"]["epoch"] != epoch or state["sampler"]["epoch"] != epoch
                or state["scheduler"]["table"] != self.scheduler.table
                or state["scheduler"]["settings_hash"] != canonical_hash(self.settings)):
            raise ValueError("checkpoint scheduler/epoch identity mismatch")
        model_state = self.model.state_dict()
        if state["model"].keys() != model_state.keys():
            raise ValueError("checkpoint model keys mismatch")
        def valid_tensor(value, target):
            return (isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.shape == target.shape
                    and value.dtype == target.dtype and (not value.is_floating_point() or bool(torch.isfinite(value).all())))
        if any(not valid_tensor(value, model_state[name]) for name, value in state["model"].items()):
            raise ValueError("checkpoint model tensor mismatch")
        current_groups = self.optimizer.state_dict()["param_groups"]
        saved_groups = state["optimizer"]["param_groups"]
        group_keys = ("name", "initial_lr", "momentum", "weight_decay", "dampening", "nesterov", "maximize",
                      "foreach", "differentiable", "fused", "params")
        lrs = [self.scheduler.table[epoch]*0.1, self.scheduler.table[epoch]]
        if len(saved_groups) != 2 or any(
            saved.keys() != current.keys() or any(saved.get(k) != current.get(k) for k in group_keys) or saved["lr"] != lr
            for saved, current, lr in zip(saved_groups, current_groups, lrs)):
            raise ValueError("checkpoint optimizer group mismatch")
        parameters = {index: parameter for group, actual in zip(current_groups, self.optimizer.param_groups)
                      for index, parameter in zip(group["params"], actual["params"])}
        for index, value in state["optimizer"]["state"].items():
            if index not in parameters or set(value) != {"momentum_buffer"} or not valid_tensor(value["momentum_buffer"], parameters[index]):
                raise ValueError("checkpoint optimizer momentum mismatch")
        named = dict(self.model.named_parameters())
        if state["gradients"].keys() != named.keys():
            raise ValueError("checkpoint gradient keys mismatch")
        for name, value in state["gradients"].items():
            if value is not None and (not valid_tensor(value, named[name]) or ".eles." in name):
                raise ValueError("checkpoint gradient tensor/dormant mismatch")
            if name.startswith("backbone.") and value is not None and (epoch < 10 or name.startswith(("backbone.layer1.", "backbone.layer2."))):
                raise ValueError("frozen backbone must have no gradient")
        sampler = SyntheticSampler.from_state(state["sampler"])
        pending = state["pending"]
        remaining = 0
        if pending is not None:
            if not 0 <= pending["index"] < len(pending["batches"]) or window_denominators(pending["batches"]) != pending["denominators"]:
                raise ValueError("pending index/denominators mismatch")
            ids = [i for batch in pending["batches"] for i in batch["sample_ids"]]
            if ids != sampler.order[sampler.position-len(ids):sampler.position] or any(b.get("data_kind") != "synthetic" for b in pending["batches"]):
                raise ValueError("pending synthetic input order mismatch")
            remaining = sum(len(b["sample_ids"]) for b in pending["batches"][pending["index"]:])
        trace = state["trace"]
        consumed = [i for row in trace if row["epoch"] == epoch for i in row["sample_ids"]]
        if (consumed != sampler.order[:sampler.position-remaining] or state["microsteps"] != len(trace)
                or state["update"] != sum(row["stepped"] for row in trace) or state["update"] > self.max_updates):
            raise ValueError("sampler/trace/update mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch, self.update, self.microsteps = epoch, state["update"], state["microsteps"]
        self.sampler, self.pending, self.trace = sampler, pending, trace
        apply_epoch_modes(self.model, epoch, self.settings)
        for name, parameter in named.items():
            gradient = state["gradients"][name]
            parameter.grad = gradient.to(self.device).clone() if gradient is not None else None
        restore_device_rng(state["rng"], self.device)
        return self
