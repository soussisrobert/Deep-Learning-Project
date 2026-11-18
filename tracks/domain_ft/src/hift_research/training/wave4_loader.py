"""Indexed attempt transport; serial/no-prefetch is the only exact-resume recommendation."""
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hift_research.data_contract.torch_adapter import reject_aware_collate
from hift_research.training.graph import synthetic_batch
from hift_research.training.recipe import canonical_hash


class SyntheticCandidateDataset(Dataset):
    """Deterministic synthetic test fixture matching the production tuple API; no real files."""
    def __init__(self, length=32, seed=20261023, rejected=(), faults=None, channels_last_arrays=False):
        self.length, self.seed = length, seed
        self.rejected, self.faults = tuple(rejected), dict(faults or {})
        self.channels_last_arrays = bool(channels_last_arrays)
        self.manifest_sha256 = canonical_hash({"kind": "synthetic", "length": length, "seed": seed,
                                              "rejected": self.rejected, "faults": self.faults,
                                              "channels_last_arrays": self.channels_last_arrays})
        self.policy_sha256 = canonical_hash({"negative": "every fourth synthetic ID", "seed": seed})
        self.reservations_sha256 = canonical_hash("synthetic-no-authentic-data")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        fault = self.faults.get(index)
        if fault:
            if fault == "missing":
                raise FileNotFoundError(f"synthetic missing image index={index}")
            if fault == "decode":
                raise ValueError(f"synthetic decode failure index={index}")
            raise RuntimeError(f"synthetic worker exception index={index}")
        record = {"candidate_index": index, "seed": self.seed+index,
                  "candidate_negative": bool(index % 4 == 3),
                  "status": "rejected" if index in self.rejected else "accepted",
                  "data_kind": "synthetic", "manifest_sha256": self.manifest_sha256,
                  "reasons": ["synthetic_geometry_reject"] if index in self.rejected else []}
        if record["status"] != "accepted":
            return record, None
        # RNG isolation is CPU-only and independent of worker/prefetch execution order.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                np.random.seed(self.seed+index)
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(self.seed+index).get_state())
                template, search, targets = synthetic_batch(1, negative=[record["candidate_negative"]])
        finally:
            np.random.set_state(numpy_state)
        factor = 0.95+random.Random(self.seed+index).random()*0.05
        arrays = {"template": template[0].numpy()*factor, "search": search[0].numpy(),
                  "bbox": np.asarray([95, 95, 223, 223], dtype=np.float32)}
        arrays.update({key: value[0].numpy() for key, value in targets.items()})
        if self.channels_last_arrays:
            for key in ("template", "search"):
                # Exact HWC-contiguous->CHW-view path used by authentic image crops.
                arrays[key] = np.ascontiguousarray(arrays[key].transpose(1,2,0)).transpose(2,0,1)
        return record, arrays

    candidate = __getitem__


class ReceiptDataset(Dataset):
    """Default errors PROPAGATE; diagnostic receipt mode logs faults with no replacement."""
    def __init__(self, dataset, *, errors="raise"):
        if errors not in ("raise", "receipt"):
            raise ValueError("errors must be raise or diagnostic receipt")
        self.dataset, self.errors = dataset, errors

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            record, arrays = self.dataset[index]
        except Exception as exc:
            if self.errors == "raise":
                raise RuntimeError(f"candidate index={index} failed; stop, no replacement") from exc
            record, arrays = {"candidate_index": index, "status": "rejected", "data_kind": "diagnostic_fault",
                              "reasons": [f"exception:{type(exc).__name__}"], "error": str(exc)}, None
        if record.get("candidate_index") != index:
            raise ValueError("candidate record index does not match requested index")
        if record.get("status") == "accepted" and arrays is None:
            raise ValueError("accepted attempt missing arrays")
        return record, arrays


def ordered_collate(candidates):
    result = reject_aware_collate(candidates)
    result["attempt_records"] = [record for record, arrays in candidates]
    result["attempt_indices"] = [record["candidate_index"] for record in result["attempt_records"]]
    if len(set(result["attempt_indices"])) != len(candidates):
        raise ValueError("duplicate attempt index in collator")
    return result


def indexed_loader(dataset, *, start=0, batch_size=4, num_workers=0, seed=20261023, errors="raise"):
    if num_workers not in (0, 2):
        raise ValueError("only serial0 or explicit tested spawn2 supported")
    if not 0 <= start <= len(dataset) or batch_size < 1:
        raise ValueError("invalid indexed loader bounds")
    kwargs = {"dataset": ReceiptDataset(dataset, errors=errors), "sampler": list(range(start, len(dataset))),
              "batch_size": batch_size, "num_workers": num_workers, "collate_fn": ordered_collate,
              "generator": torch.Generator().manual_seed(seed), "pin_memory": False}
    if num_workers:
        kwargs.update({"multiprocessing_context": "spawn", "prefetch_factor": 1, "persistent_workers": False})
    return DataLoader(**kwargs)


class ConsumedCursor:
    """Consumed receipts, not worker-dispatched indices. No async resume equivalence claim."""
    def __init__(self, dataset_identity, *, next_index=0):
        self.dataset_identity = copy.deepcopy(dataset_identity)
        self.next_index = next_index
        self.attempts = self.accepted = self.rejected = 0

    def commit(self, batch):
        indices = batch["attempt_indices"]
        if indices != list(range(self.next_index, self.next_index+len(indices))):
            raise ValueError("out-of-order, duplicate or missing consumed attempt")
        if batch["replacement_count"] != 0 or batch["candidate_count"] != len(indices):
            raise ValueError("replacement or attempt accounting mismatch")
        self.next_index += len(indices)
        self.attempts += len(indices)
        self.accepted += len(batch["accepted_records"])
        self.rejected += len(batch["rejected_records"])
        if self.attempts != self.accepted+self.rejected:
            raise ValueError("attempt accounting drift")

    def state_dict(self):
        return {"dataset_identity": self.dataset_identity, "next_index": self.next_index,
                "attempts": self.attempts, "accepted": self.accepted, "rejected": self.rejected}

    @classmethod
    def from_state(cls, state, dataset_identity):
        if state["dataset_identity"] != dataset_identity:
            raise ValueError("loader manifest/policy/reservations identity mismatch")
        value = cls(dataset_identity, next_index=state["next_index"])
        for key in ("attempts", "accepted", "rejected"):
            setattr(value, key, state[key])
        if value.attempts != value.accepted+value.rejected or value.next_index != value.attempts:
            raise ValueError("invalid consumed cursor state")
        return value


def dataset_identity(dataset):
    return {name: getattr(dataset, name) for name in ("manifest_sha256", "policy_sha256", "reservations_sha256")}


class SyntheticIndexedFactory:
    """Actual Dataset/collate CPU path into synthetic-only lifecycle (never real arrays)."""
    def __init__(self, dataset):
        if not isinstance(dataset, SyntheticCandidateDataset):
            raise ValueError("optimization fixture accepts only SyntheticCandidateDataset, never temporal real images")
        self.dataset = dataset

    def batch(self, ids):
        result = ordered_collate([ReceiptDataset(self.dataset)[index] for index in ids])
        if result["rejected_records"]:
            raise ValueError("synthetic SGD comparison fixture requires accepted IDs; rejection tested separately")
        inputs = result["model_inputs"]
        return {"data_kind": "synthetic", "sample_ids": ids, "template": inputs["template"], "search": inputs["search"],
                "targets": {key: inputs[key] for key in ("label_cls1", "labelxff", "labelcls2", "weightxff")},
                "attempt_records": result["attempt_records"]}
