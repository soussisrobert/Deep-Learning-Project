"""Lazy per-process Windows-safe indexed Dataset; no rejection refill."""
from __future__ import annotations
import os
from numbers import Integral
from pathlib import Path
from torch.utils.data import Dataset
from hift_research.data_contract.torch_adapter import reject_aware_collate
from hift_research.data_contract.schema import file_hash
from .policy import load_policy
from .pipeline import TemporalPairPipeline


class TemporalCandidateDataset(Dataset):
    def __init__(self,manifest_path,policy_path,split="train",seed=None,length=None):
        self.manifest_path=str(Path(manifest_path).resolve());self.policy_path=str(Path(policy_path).resolve())
        self.split="development" if split=="dev" else split
        frozen=load_policy(self.policy_path);policy=frozen["policy"]
        self.seed=policy["seed"] if seed is None else seed
        self.length=policy["preflight_length"] if length is None else length
        self.manifest_sha256=file_hash(self.manifest_path);self.policy_sha256=file_hash(self.policy_path)
        self.reservations_sha256=frozen["reservations_sha256"]
        if (self.split!=policy["split"] or self.seed!=policy["seed"] or self.length not in policy["allowed_lengths"]
                or self.manifest_sha256!=frozen["manifest_sha256"]):raise ValueError("dataset differs from frozen source/split/seed/length policy")
        self._pipeline=None;self._pid=None
    def __len__(self):return self.length
    def __getstate__(self):
        state=self.__dict__.copy();state["_pipeline"]=None;state["_pid"]=None;return state
    def candidate(self,index):
        if isinstance(index,bool) or not isinstance(index,Integral) or not 0<=index<self.length:raise IndexError(index)
        try:
            if self._pipeline is None or self._pid!=os.getpid():
                self._pipeline=TemporalPairPipeline(self.manifest_path,self.policy_path,self.split,self.seed,self.length)
                self._pid=os.getpid()
            return self._pipeline.candidate(int(index))
        except Exception as exc:
            raise RuntimeError(f"TemporalCandidateDataset candidate_index={index} failed: {type(exc).__name__}: {exc}") from exc
    def __getitem__(self,index):return self.candidate(index)
