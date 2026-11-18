"""CPU-only reject-aware Dataset/collate bridge, imported explicitly when needed.

Stores only manifest path/length across Windows spawn; pinned dynamic source
classes and caches are created lazily in each worker. No sampler replacement,
model forward, SGD, CUDA, or exact asynchronous-loader resume claim.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .schema import validate_training_manifest
from .pipeline import RealPairPipeline


class CandidateDataset(Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = str(Path(manifest_path).resolve())
        manifest = validate_training_manifest(self.manifest_path)
        self.length = manifest["policy"]["candidate_count"]
        self._pipeline = None
        self._pid = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pipeline"], state["_pid"] = None, None
        return state

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        if self._pipeline is None or self._pid != os.getpid():
            self._pipeline, self._pid = RealPairPipeline(self.manifest_path), os.getpid()
        return self._pipeline.candidate(int(index))


def reject_aware_collate(candidates):
    """Separate accepted/rejected candidates; never draw a replacement.

    Accepted target arrays are explicitly cast to FP32 for the corrected B2
    graph. Original audit-array dtype/hash remains recorded in each candidate.
    An all-rejected batch has model_inputs=None and must be logged/skipped by an
    authorized training lifecycle, not silently filled from future candidates.
    """
    accepted = [(record, arrays) for record, arrays in candidates if record["status"] == "accepted"]
    rejected = [record for record, arrays in candidates if record["status"] != "accepted"]
    if any(arrays is None for record, arrays in accepted):
        raise ValueError("accepted candidate missing model inputs")
    model_inputs = None
    if accepted:
        keys = ("template", "search", "label_cls1", "labelxff", "labelcls2", "weightxff", "bbox")
        model_inputs = {key: torch.from_numpy(np.stack([arrays[key] for record, arrays in accepted])).to(dtype=torch.float32)
                        for key in keys}
    return {"model_inputs": model_inputs, "accepted_records": [record for record, arrays in accepted],
            "rejected_records": rejected, "candidate_count": len(candidates), "replacement_count": 0}
