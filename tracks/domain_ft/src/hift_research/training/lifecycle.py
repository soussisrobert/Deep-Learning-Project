"""CPU synthetic-only corrected lifecycle: exact resume, explicit epoch clock, global reductions.

No real-data SGD entry point and no implicit CUDA operations. Accumulation holds
one CPU synthetic window so true global class/mask denominators are known before
backward; pending inputs, gradients and RNG are checkpointed mid-window.
"""
import copy
import inspect
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hift_research.corrections.masked_losses import masked_bce_with_logits, masked_iou_loss
from .graph import masked_classification_log_probs, masked_decode_regression
from hift_research.provenance import sha256
from .recipe import ROOT, SOURCE_FILES, canonical_hash, source_lr_table, source_settings


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"algorithm": state[0], "keys": state[1].tolist(),
            "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch_cpu": torch.get_rng_state().clone(), "cuda_used": False, "torch_cuda": []}


def restore_rng(state):
    if state["cuda_used"] or state["torch_cuda"]:
        raise ValueError("CPU-only lifecycle cannot restore a CUDA-used checkpoint")
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["algorithm"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])


def apply_epoch_modes(model, epoch, settings=None):
    settings = source_settings() if settings is None else settings
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < settings["TRAIN"]["EPOCH"]:
        raise ValueError("epoch outside source schedule")
    model.train()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for module in model.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    if epoch >= settings["BACKBONE"]["TRAIN_EPOCH"]:
        for name in settings["BACKBONE"]["TRAIN_LAYERS"]:
            layer = getattr(model.backbone, name)
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
            for module in layer.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train()
    return model


def lifecycle_optimizer(model, settings=None):
    """Stable group membership avoids losing grader momentum at epoch10."""
    settings = source_settings() if settings is None else settings
    train = settings["TRAIN"]
    base = train["BASE_LR"]
    ratio = settings["BACKBONE"]["LAYERS_LR"]
    return torch.optim.SGD([
        {"name": "backbone", "params": list(model.backbone.parameters()), "lr": base*ratio,
         "initial_lr": base*ratio},
        {"name": "grader", "params": list(model.grader.parameters()), "lr": base, "initial_lr": base},
    ], momentum=train["MOMENTUM"], weight_decay=train["WEIGHT_DECAY"])


class EpochSchedule:
    """Explicit source-valued epoch clock. Optimizer updates NEVER advance it."""
    def __init__(self, optimizer, settings=None):
        self.settings = source_settings() if settings is None else copy.deepcopy(settings)
        self.optimizer = optimizer
        self.epoch = 0
        self.table = source_lr_table(self.settings)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < len(self.table):
            raise ValueError("invalid schedule epoch")
        self.epoch = epoch
        ratio = self.settings["BACKBONE"]["LAYERS_LR"]
        if [g.get("name") for g in self.optimizer.param_groups] != ["backbone", "grader"]:
            raise ValueError("optimizer groups must remain backbone,grader")
        self.optimizer.param_groups[0]["lr"] = self.table[epoch]*ratio
        self.optimizer.param_groups[1]["lr"] = self.table[epoch]

    def state_dict(self):
        return {"epoch": self.epoch, "table": self.table, "settings_hash": canonical_hash(self.settings)}

    def load_state_dict(self, state):
        if state["table"] != self.table or state["settings_hash"] != canonical_hash(self.settings):
            raise ValueError("scheduler source/config identity mismatch")
        self.set_epoch(state["epoch"])


def window_denominators(batches):
    totals = {"positive": 0, "negative": 0, "bce": 0, "localization": 0.0}
    for batch in batches:
        targets = batch["targets"]
        label = targets["label_cls1"]
        weight = targets["weightxff"]
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError("finite nonnegative weights required")
        totals["positive"] += int(label.eq(1).sum())
        totals["negative"] += int(label.eq(0).sum())
        totals["bce"] += targets["labelcls2"].numel()
        totals["localization"] += float(weight.double().sum())
    return totals


def window_loss(heads, targets, denominators):
    """One microbatch contribution to a globally normalized effective-window objective.

    NLL classes are normalized independently across the whole window, not by
    microbatch count. BCE uses all cells; IoU uses the total regression weight
    with ONE upstream denominator epsilon. Actual final partial windows work too.
    """
    loc, cls1, cls2 = heads
    labels = targets["label_cls1"]
    logp = masked_classification_log_probs(cls1, labels).reshape(-1, 2)
    label = labels.reshape(-1).long()
    nll = logp.reshape(-1)[:0].sum()
    for class_id, key in ((1, "positive"), (0, "negative")):
        selected = label.eq(class_id)
        count = denominators[key]
        if count:
            nll = nll + 0.5*F.nll_loss(logp[selected], label[selected], reduction="sum")/count
        elif bool(selected.any()):
            raise ValueError("global classification denominator excludes active observations")
    bce_count = denominators["bce"]
    if bce_count < cls2.numel() or bce_count <= 0:
        raise ValueError("invalid global BCE denominator")
    bce = masked_bce_with_logits(cls2, targets["labelcls2"])*cls2.numel()/bce_count
    weight = targets["weightxff"]
    local_weight = float(weight.double().sum())
    global_weight = denominators["localization"]
    if global_weight < local_weight or global_weight < 0:
        raise ValueError("invalid global localization denominator")
    iou = masked_iou_loss(masked_decode_regression(loc, weight),
                         masked_decode_regression(targets["labelxff"], weight), weight)
    iou = iou*((local_weight+1e-6)/(global_weight+1e-6)) if global_weight else iou
    total = 2.4*iou+nll+bce
    if not bool(torch.isfinite(total)):
        raise ValueError("nonfinite effective-window loss")
    return total


class SyntheticSampler:
    """Explicit single-worker synthetic order; no hidden prefetch/RNG ownership."""
    def __init__(self, sample_ids, *, epoch=0, position=0):
        self.order = list(sample_ids)
        if not self.order or not all(isinstance(v, int) and not isinstance(v, bool) for v in self.order):
            raise ValueError("nonempty integer synthetic sample order required")
        if len(set(self.order)) != len(self.order) or not 0 <= position <= len(self.order):
            raise ValueError("unique sample IDs and bounded position required")
        self.epoch, self.position = epoch, position

    def take(self, count):
        values = self.order[self.position:self.position+count]
        self.position += len(values)
        return values

    def state_dict(self):
        return {"order": self.order, "epoch": self.epoch, "position": self.position}

    @classmethod
    def from_state(cls, state):
        return cls(state["order"], epoch=state["epoch"], position=state["position"])


def _assert_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Wave3 lifecycle is CPU-only; GPU tensors forbidden")
    elif isinstance(value, dict):
        for item in value.values():
            _assert_cpu_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_cpu_tree(item)


def actual_heads(model, batch):
    z, x = model.features(batch["template"], batch["search"])
    return model.grader(x, z)


class SyntheticLifecycle:
    """Only synthetic small optimization is authorized; data identity enforced on restore.

    batch_factory receives IDs and MUST return synthetic tensors plus those IDs.
    Its implementation identity is caller-pinned in config_identity. Production
    loader worker queues/augmentation state are deliberately not claimed supported.
    """
    def __init__(self, model, sampler, *, batch_factory, config_identity, data_identity,
                 microbatch=1, accumulation=2, epoch=0, heads=actual_heads, settings=None):
        if data_identity.get("kind") != "synthetic" or not data_identity.get("sha256"):
            raise ValueError("only explicitly identified synthetic data is authorized")
        if min(microbatch, accumulation) < 1 or isinstance(microbatch, bool) or isinstance(accumulation, bool):
            raise ValueError("positive integer microbatch/accumulation required")
        if not isinstance(microbatch, int) or not isinstance(accumulation, int):
            raise ValueError("positive integer microbatch/accumulation required")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("CPU-only lifecycle")
        self.model, self.sampler = model, sampler
        self.batch_factory, self.heads = batch_factory, heads
        self.config_identity, self.data_identity = copy.deepcopy(config_identity), copy.deepcopy(data_identity)
        implementation_files = list(SOURCE_FILES) + [
            "src/hift_research/training/lifecycle.py", "src/hift_research/training/recipe.py",
            "src/hift_research/training/graph.py", "src/hift_research/training/upstream.py",
            "src/hift_research/corrections/classification_loss.py", "src/hift_research/corrections/masked_losses.py",
            "src/hift_research/corrections/positive_targets.py", "src/hift_research/corrections/pair_targets.py",
            "src/hift_research/corrections/tensor_layout.py", "third_party/HiFT/pysot/models/utile/tran.py",
            "third_party/HiFT/pysot/models/utile/utile.py", "third_party/HiFT/pysot/models/backbone/newalexnet.py",
            "third_party/HiFT/pysot/models/loss.py", "third_party/HiFT/pysot/datasets/anchortarget.py"]
        files = {str((ROOT/p).resolve()) for p in implementation_files}
        for value in (batch_factory, heads, type(model)):
            filename = inspect.getsourcefile(value)
            if filename is None:
                raise ValueError("synthetic model/factory/head source must be identifiable")
            files.add(str(Path(filename).resolve()))
        self.code_identity = {str(Path(filename).relative_to(ROOT)).replace("\\", "/")
                              if Path(filename).is_relative_to(ROOT) else filename: sha256(filename)
                              for filename in sorted(files)}
        self.settings = source_settings() if settings is None else copy.deepcopy(settings)
        self.microbatch, self.accumulation = microbatch, accumulation
        self.epoch, self.update, self.microsteps = epoch, 0, 0
        self.pending, self.trace = None, []
        self.optimizer = lifecycle_optimizer(model, self.settings)
        self.scheduler = EpochSchedule(self.optimizer, self.settings)
        self.set_epoch(epoch)
        if self.sampler.epoch != epoch:
            raise ValueError("sampler and lifecycle epoch mismatch")

    def set_epoch(self, epoch):
        if self.pending is not None:
            raise ValueError("cannot advance epoch with pending accumulation")
        apply_epoch_modes(self.model, epoch, self.settings)
        self.scheduler.set_epoch(epoch)
        self.epoch = epoch

    def start_next_epoch(self, sample_ids):
        if self.pending is not None or self.sampler.position != len(self.sampler.order):
            raise ValueError("finish current epoch and final partial accumulation before advancing")
        next_epoch = self.epoch+1
        self.set_epoch(next_epoch)
        self.sampler = SyntheticSampler(sample_ids, epoch=next_epoch)

    def begin_window(self):
        if self.pending is not None:
            raise ValueError("window already pending")
        ids = self.sampler.take(self.microbatch*self.accumulation)
        if not ids:
            return False
        batches = []
        for start in range(0, len(ids), self.microbatch):
            group = ids[start:start+self.microbatch]
            batch = self.batch_factory(group)
            if batch.get("sample_ids") != group or batch.get("data_kind") != "synthetic":
                raise ValueError("factory must return matching explicitly synthetic IDs")
            _assert_cpu_tree(batch)
            batches.append(batch)
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = {"batches": batches, "index": 0, "denominators": window_denominators(batches)}
        return True

    def microstep(self):
        if self.pending is None and not self.begin_window():
            return None
        state = self.pending
        batch = state["batches"][state["index"]]
        loss = window_loss(self.heads(self.model, batch), batch["targets"], state["denominators"])
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
            raise ValueError("nonfinite synthetic gradients")
        self.microsteps += 1
        state["index"] += 1
        stepped = state["index"] == len(state["batches"])
        norm = None
        if stepped:
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings["TRAIN"]["GRAD_CLIP"]))
            if not np.isfinite(norm):
                raise ValueError("nonfinite clipping norm")
            self.optimizer.step()
            self.update += 1
        record = {"sample_ids": batch["sample_ids"], "loss_contribution": float(loss.detach()),
                  "epoch": self.epoch, "update": self.update, "microstep": self.microsteps,
                  "stepped": stepped, "preclip_norm": norm,
                  "lr": [group["lr"] for group in self.optimizer.param_groups]}
        self.trace.append(record)
        if stepped:
            self.pending = None
        return record

    def run_to_end(self):
        while self.microstep() is not None:
            pass
        return self.trace

    def save(self, path):
        """Exclusive checkpoint: tensors/primitives only; never pickle live model objects."""
        path = Path(path)
        state = {"schema": "hift-synthetic-lifecycle-v1", "config_identity": self.config_identity,
                 "data_identity": self.data_identity, "code_identity": self.code_identity,
                 "settings_hash": canonical_hash(self.settings),
                 "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "epoch": self.epoch, "update": self.update,
                 "microsteps": self.microsteps, "sampler": self.sampler.state_dict(),
                 "pending": self.pending, "trace": self.trace, "rng": capture_rng(),
                 "gradients": {name: p.grad.detach().clone() if p.grad is not None else None
                               for name, p in self.model.named_parameters()},
                 "microbatch": self.microbatch, "accumulation": self.accumulation,
                 "device_policy": "cpu_only_synthetic"}
        _assert_cpu_tree(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(state, stream)

    def restore(self, path):
        """Restricted torch weights-only loader; identity/shape/modes checked before resume."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state.get("schema") != "hift-synthetic-lifecycle-v1"
                or state.get("device_policy") != "cpu_only_synthetic"
                or state.get("config_identity") != self.config_identity
                or state.get("data_identity") != self.data_identity or state.get("code_identity") != self.code_identity
                or state.get("settings_hash") != canonical_hash(self.settings)
                or state.get("microbatch") != self.microbatch or state.get("accumulation") != self.accumulation):
            raise ValueError("checkpoint config/data/source/device identity mismatch")
        _assert_cpu_tree(state)
        if state["scheduler"]["epoch"] != state["epoch"] or state["sampler"]["epoch"] != state["epoch"]:
            raise ValueError("checkpoint epoch clocks disagree")
        expected_model = self.model.state_dict()
        if state["model"].keys() != expected_model.keys():
            raise ValueError("checkpoint model keys mismatch")
        for name, value in state["model"].items():
            expected = expected_model[name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError("checkpoint model shape/dtype/finite mismatch")
        current_groups = self.optimizer.state_dict()["param_groups"]
        saved_groups = state["optimizer"]["param_groups"]
        group_options = ("name", "initial_lr", "momentum", "weight_decay", "dampening", "nesterov",
                         "maximize", "foreach", "differentiable", "fused")
        epoch = state["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 0 <= epoch < len(self.scheduler.table):
            raise ValueError("checkpoint epoch outside audited schedule")
        expected_lrs = [self.scheduler.table[epoch]*self.settings["BACKBONE"]["LAYERS_LR"], self.scheduler.table[epoch]]
        if len(saved_groups) != len(current_groups) or any(
            saved.keys() != current.keys() or saved["params"] != current["params"]
            or any(saved.get(key) != current.get(key) for key in group_options) or saved["lr"] != expected_lr
            for saved, current, expected_lr in zip(saved_groups, current_groups, expected_lrs)
        ):
            raise ValueError("checkpoint optimizer groups/recipe mismatch")
        id_to_parameter = {index: parameter for group, actual_group in zip(current_groups, self.optimizer.param_groups)
                           for index, parameter in zip(group["params"], actual_group["params"])}
        for index, optimizer_state in state["optimizer"]["state"].items():
            if index not in id_to_parameter or set(optimizer_state) not in (set(), {"momentum_buffer"}):
                raise ValueError("unexpected checkpoint optimizer state")
            parameter = id_to_parameter[index]
            buffer = optimizer_state.get("momentum_buffer")
            if buffer is not None and (not isinstance(buffer, torch.Tensor) or buffer.shape != parameter.shape
                    or buffer.dtype != parameter.dtype or not bool(torch.isfinite(buffer).all())):
                raise ValueError("checkpoint optimizer momentum shape/dtype/finite mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch, self.update, self.microsteps = state["epoch"], state["update"], state["microsteps"]
        self.pending, self.trace = state["pending"], state["trace"]
        self.sampler = SyntheticSampler.from_state(state["sampler"])
        if self.pending is not None:
            if not 0 <= self.pending["index"] < len(self.pending["batches"]):
                raise ValueError("invalid pending accumulation index")
            if window_denominators(self.pending["batches"]) != self.pending["denominators"]:
                raise ValueError("pending global denominators mismatch")
            pending_ids = [sample for batch in self.pending["batches"] for sample in batch["sample_ids"]]
            if (any(batch.get("data_kind") != "synthetic" for batch in self.pending["batches"])
                    or pending_ids != self.sampler.order[self.sampler.position-len(pending_ids):self.sampler.position]):
                raise ValueError("pending samples disagree with frozen sampler order")
            remaining = sum(len(batch["sample_ids"]) for batch in self.pending["batches"][self.pending["index"]:])
        else:
            remaining = 0
        consumed = [sample for row in self.trace if row["epoch"] == self.epoch for sample in row["sample_ids"]]
        if (consumed != self.sampler.order[:self.sampler.position-remaining]
                or self.microsteps != len(self.trace) or self.update != sum(row["stepped"] for row in self.trace)):
            raise ValueError("checkpoint sampler/trace/update clocks disagree")
        apply_epoch_modes(self.model, self.epoch, self.settings)
        named = dict(self.model.named_parameters())
        if named.keys() != state["gradients"].keys():
            raise ValueError("checkpoint gradient keys mismatch")
        for name, parameter in named.items():
            gradient = state["gradients"][name]
            if gradient is not None and (not isinstance(gradient, torch.Tensor) or gradient.shape != parameter.shape
                    or gradient.dtype != parameter.dtype or not bool(torch.isfinite(gradient).all())):
                raise ValueError("checkpoint gradient shape/dtype/finite mismatch")
            parameter.grad = gradient.clone() if gradient is not None else None
        restore_rng(state["rng"])
        return self
