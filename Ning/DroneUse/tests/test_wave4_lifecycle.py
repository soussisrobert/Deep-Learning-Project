"""CPU first: Windows spawn/order/fault receipts and serial intended-input replay."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hift_research.training.graph import B2TrainingGraph
from hift_research.training.lifecycle import SyntheticSampler
from hift_research.training.recipe import ROOT, canonical_hash
from hift_research.training.wave4_loader import (
    ConsumedCursor, SyntheticCandidateDataset, SyntheticIndexedFactory, dataset_identity, indexed_loader,
)
from hift_research.training.wave4_lifecycle import Wave4SyntheticLifecycle, cpu_tree, device_rng, seed_device
from hift_research.training.wave4_recipes import pilot_recipe, reference_recipe, validate_frozen_pilot


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(autouse=True)
def cpu_seed():
    seed_device(20261023, "cpu")


def collect(dataset, workers, *, errors="raise", start=0, batch_size=2):
    return list(indexed_loader(dataset, start=start, batch_size=batch_size, num_workers=workers, errors=errors))


def test_serial_loader_iterator_and_indexing_preserve_parent_rng():
    dataset = SyntheticCandidateDataset(length=4)
    before = torch.get_rng_state().clone()
    first = next(iter(indexed_loader(dataset, num_workers=0, batch_size=2)))
    assert torch.equal(before, torch.get_rng_state())  # dedicated DataLoader Generator, not parent RNG
    assert first["attempt_indices"] == [0, 1]
    assert first["model_inputs"]["template"].dtype == torch.float32
    assert tree_equal(collect(dataset, 0), collect(dataset, 0))


def test_actual_windows_spawn2_order_seed_reject_and_fault_receipts_match_serial():
    dataset = SyntheticCandidateDataset(length=6, rejected=(1,), faults={3: "missing", 4: "decode", 5: "exception"})
    zero = collect(dataset, 0, errors="receipt")
    spawned = collect(dataset, 2, errors="receipt")
    assert tree_equal(zero, spawned)
    cursor = ConsumedCursor(dataset_identity(dataset))
    for batch in spawned:
        cursor.commit(batch)
    assert cursor.state_dict()["next_index"] == 6
    assert (cursor.attempts, cursor.accepted, cursor.rejected) == (6, 2, 4)
    assert [r["candidate_index"] for batch in spawned for r in batch["attempt_records"]] == list(range(6))
    assert all(batch["replacement_count"] == 0 for batch in spawned)
    assert spawned[-1]["model_inputs"] is None  # all-rejected batch is honest, not filled


@pytest.mark.parametrize("fault", ["missing", "decode", "exception"])
def test_default_fault_stops_with_unconsumed_index_and_explicit_reconstruction(fault):
    failed = SyntheticCandidateDataset(length=4, faults={2: fault})
    cursor = ConsumedCursor(dataset_identity(failed))
    iterator = iter(indexed_loader(failed, batch_size=1))
    cursor.commit(next(iterator)); cursor.commit(next(iterator))
    saved = cursor.state_dict()
    with pytest.raises(RuntimeError, match="index=2"):
        next(iterator)
    assert cursor.state_dict() == saved
    # Explicit diagnostic repair/reconstruction, NOT a retry/replacement inside a running pilot.
    repaired = SyntheticCandidateDataset(length=4)
    remaining = collect(repaired, 0, start=2, batch_size=1)
    assert [b["attempt_indices"] for b in remaining] == [[2], [3]]
    with pytest.raises(ValueError, match="identity"):
        ConsumedCursor.from_state(saved, dataset_identity(repaired))


def test_serial_consumed_cursor_resume_does_not_use_producer_dispatch_position():
    dataset = SyntheticCandidateDataset(length=6, rejected=(2, 5))
    cursor = ConsumedCursor(dataset_identity(dataset))
    full = collect(dataset, 0)
    cursor.commit(full[0])
    resumed = ConsumedCursor.from_state(cursor.state_dict(), dataset_identity(dataset))
    remainder = collect(dataset, 0, start=resumed.next_index)
    assert tree_equal(remainder, full[1:])
    for batch in remainder:
        resumed.commit(batch)
    assert resumed.attempts == 6 and resumed.accepted == 4 and resumed.rejected == 2
    with pytest.raises(ValueError, match="out-of-order"):
        resumed.commit(full[0])


def make_engine(epoch=10):
    dataset = SyntheticCandidateDataset(length=4)
    factory = SyntheticIndexedFactory(dataset)
    return Wave4SyntheticLifecycle(B2TrainingGraph(), SyntheticSampler(range(4), epoch=epoch),
                                   batch_factory=factory.batch, config_identity={"fixture": "Wave4CPU"},
                                   data_identity={"kind": "synthetic", "sha256": dataset.manifest_sha256},
                                   microbatch=1, accumulation=2, epoch=epoch, max_updates=2)


@pytest.mark.parametrize("epoch", [0, 10])
def test_actual_cpu_complete_indexed_input_h2d_identity_and_midaccum_resume(tmp_path, epoch):
    full = make_engine(epoch)
    full.run_to_end()
    expected_rng = device_rng(torch.device("cpu"))
    seed_device(20261023, "cpu")
    part = make_engine(epoch)
    part.microstep()
    checkpoint = tmp_path/f"mid-{epoch}.pt"
    part.save(checkpoint)
    restored = make_engine(epoch).restore(checkpoint)
    restored.run_to_end()
    for left, right in ((full.model.state_dict(), restored.model.state_dict()),
                        (full.optimizer.state_dict(), restored.optimizer.state_dict()),
                        (full.trace, restored.trace), (full.scheduler.state_dict(), restored.scheduler.state_dict()),
                        ({n:p.grad for n,p in full.model.named_parameters()}, {n:p.grad for n,p in restored.model.named_parameters()}),
                        (expected_rng, device_rng(torch.device("cpu")))):
        assert tree_equal(left, right)
    assert restored.update == 2 and restored.pending is None
    with pytest.raises(FileExistsError):
        part.save(checkpoint)


def test_cuda_and_real_data_are_fail_closed_without_permission():
    with pytest.raises(ValueError, match="GPU token"):
        seed_device(1, "cuda")
    model = B2TrainingGraph()
    factory = SyntheticIndexedFactory(SyntheticCandidateDataset(length=2))
    with pytest.raises(ValueError, match="GPU token"):
        Wave4SyntheticLifecycle(model, SyntheticSampler([0,1]), batch_factory=factory.batch,
                                config_identity={}, data_identity={"kind":"synthetic","sha256":"x"}, device="cuda")
    with pytest.raises(ValueError, match="synthetic"):
        Wave4SyntheticLifecycle(model, SyntheticSampler([0,1]), batch_factory=factory.batch,
                                config_identity={}, data_identity={"kind":"real","sha256":"x"})
    with pytest.raises(ValueError, match="SyntheticCandidateDataset"):
        SyntheticIndexedFactory(object())


def test_reference_and_pilot_predeclared_policies_are_not_220_equivalence():
    reference, pilot = reference_recipe(), pilot_recipe()
    assert reference["source_epochs"] == 70 and reference["source_physical_global_batch"] == 220
    assert pilot["batch"]["nominal_physical"] == 4 and pilot["batch"]["nominal_effective"] == 16
    assert pilot["clock"]["grader_lr"] == 0.0025 and not pilot["clock"]["backbone_trainable"]
    assert pilot["budget"]["max_attempts"] == 1600 and pilot["budget"]["max_effective_updates"] == 100
    assert pilot["development"]["positive_pairs"] == pilot["development"]["safe_negative_pairs"] == 64
    assert validate_frozen_pilot(pilot)
    pilot["optimizer"]["momentum"] = 0.8
    with pytest.raises(ValueError, match="policy/config drift"):
        validate_frozen_pilot(pilot)


def make_controller(dataset=None):
    from hift_research.training.wave4_pilot_engine import SerialPilotController
    dataset = dataset or SyntheticCandidateDataset(length=4)
    return SerialPilotController(B2TrainingGraph(), dataset, config_identity={"proof": "shared-production-controller"},
                                 data_identity={"kind": "synthetic", **dataset_identity(dataset)},
                                 physical=1, attempts_per_window=2, max_updates=2, max_attempts=4)


def test_shared_production_controller_midwindow_crash_replays_from_atomic_boundary(tmp_path):
    full = make_controller().run_to_end()
    expected_rng = device_rng(torch.device("cpu"))
    seed_device(20261023, "cpu")
    part = make_controller()
    part.step()
    saved = part.save_boundary(tmp_path/"boundary.pt")
    assert not list(tmp_path.glob("*.partial-*"))
    with pytest.raises(RuntimeError, match="midwindow"):
        part.step(crash_after_micro=1)
    assert part.cursor == 2 and part.update == 1 and part.in_window
    with pytest.raises(RuntimeError, match="uncommitted"):
        part.save_boundary(tmp_path/"forbidden.pt")
    with pytest.raises(RuntimeError, match="tainted"):
        part.step()
    restored = make_controller().restore_boundary(saved["path"], expected_sha256=saved["sha256"])
    restored.run_to_end()
    assert tree_equal(full.model.state_dict(), restored.model.state_dict())
    assert tree_equal(full.optimizer.state_dict(), restored.optimizer.state_dict())
    assert tree_equal(full.semantic_trace, restored.semantic_trace)
    assert tree_equal(expected_rng, device_rng(torch.device("cpu")))
    assert restored.cursor == 4 and restored.update == 2
    assert "timings" not in restored.semantic_trace[-1]  # volatile observations never affect exact replay
    with pytest.raises(ValueError, match="file hash"):
        make_controller().restore_boundary(saved["path"], expected_sha256="wrong")


def test_shared_controller_all_rejected_window_commits_attempts_without_step_or_refill():
    controller = make_controller(SyntheticCandidateDataset(length=4, rejected=(0,1,2,3)))
    controller.run_to_end()
    assert controller.cursor == 4 and controller.window == 2 and controller.update == 0
    assert all(row["accepted"] == 0 and not row["stepped"] and row["replacements"] == 0 for row in controller.semantic_trace)


def test_future_real_gate_missing_authorization_denies_before_any_output_or_cuda(tmp_path):
    from hift_research.training.wave4_pilot import future_authorization, launch_future_pilot
    with pytest.raises(PermissionError, match="NOT AUTHORIZED"):
        future_authorization(ROOT/"configs/B2-Pilot-W4.json")
    with pytest.raises(PermissionError, match="NOT AUTHORIZED"):
        launch_future_pilot(ROOT/"configs/B2-Pilot-W4.json", output=tmp_path/"must-not-exist")
    assert not (tmp_path/"must-not-exist").exists()


AUTHENTIC_SOURCE = ROOT/"artifacts/wave4-source/SOURCE-001/temporal_manifest.json"
AUTHENTIC_POLICY = ROOT/"artifacts/wave4-preflight/FROZEN-002/train_policy.json"


@pytest.mark.skipif(not AUTHENTIC_POLICY.exists(), reason="ignored authentic temporal assets are optional; never download from tests")
def test_authentic_windows_spawn2_and_serial_have_identical_first_four_attempts():
    from torch.utils.data import Subset
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
    dataset = TemporalCandidateDataset(AUTHENTIC_SOURCE, AUTHENTIC_POLICY, split="train", seed=20261023, length=1024)
    subset = Subset(dataset, range(4))  # bound workers to four authentic attempts, not all1024
    serial = collect(subset, 0, batch_size=2)
    spawn2 = collect(subset, 2, batch_size=2)
    assert tree_equal(serial, spawn2)
    assert [r["candidate_index"] for b in serial for r in b["attempt_records"]] == [0,1,2,3]
    assert all(b["replacement_count"] == 0 for b in serial)
    cursor = ConsumedCursor(dataset_identity(dataset))
    cursor.commit(serial[0])
    restored = ConsumedCursor.from_state(cursor.state_dict(), dataset_identity(dataset))
    remainder = collect(subset, 0, start=restored.next_index, batch_size=2)
    assert tree_equal(remainder, serial[1:])


@pytest.mark.skipif(not AUTHENTIC_POLICY.exists(), reason="ignored authentic temporal assets are optional")
def test_authentic_readonly_fault_injection_stops_and_recovers_same_index_without_media_mutation():
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
    dataset = TemporalCandidateDataset(AUTHENTIC_SOURCE, AUTHENTIC_POLICY, split="train", seed=20261023, length=1024)
    class InjectedFault:
        def __len__(self):
            return 4
        def __getitem__(self, index):
            if index == 1:
                raise OSError("injected read-only transport failure index=1; media unchanged")
            return dataset[index]
    cursor = ConsumedCursor(dataset_identity(dataset))
    iterator = iter(indexed_loader(InjectedFault(), batch_size=1, num_workers=0))
    cursor.commit(next(iterator))
    boundary = cursor.state_dict()
    with pytest.raises(RuntimeError, match="index=1"):
        next(iterator)
    assert cursor.state_dict() == boundary
    restored = ConsumedCursor.from_state(boundary, dataset_identity(dataset))
    recovered = next(iter(indexed_loader(dataset, start=restored.next_index, batch_size=1, num_workers=0)))
    assert recovered["attempt_indices"] == [1] and recovered["replacement_count"] == 0


def test_real_hwc_transpose_stack_layout_is_repaired_without_pixel_or_target_changes():
    from hift_research.training.wave4_layout import canonicalize_model_inputs
    dataset = SyntheticCandidateDataset(length=4, channels_last_arrays=True)
    raw = collect(dataset, 0, batch_size=4)[0]["model_inputs"]
    assert not raw["search"].is_contiguous() and raw["search"].is_contiguous(memory_format=torch.channels_last)
    with pytest.raises(RuntimeError):
        raw["search"].view(1, 12, 287, 287)  # reproduces immutable upstream xcorr failure
    fixed = canonicalize_model_inputs(raw)
    for key in ("template", "search"):
        assert fixed[key].is_contiguous() and torch.equal(fixed[key], raw[key])
    assert fixed["labelxff"] is raw["labelxff"]
    assert fixed["search"].view(1,12,287,287).shape == (1,12,287,287)
    assert canonicalize_model_inputs(None) is None


@pytest.mark.parametrize("corruption", ["dtype", "shape", "batch", "sparse", "nonfinite", "device"])
def test_layout_bridge_rejects_unexpected_contracts(corruption):
    from hift_research.training.wave4_layout import canonicalize_model_inputs
    inputs = {"template": torch.zeros(2,3,127,127), "search": torch.zeros(2,3,287,287)}
    if corruption == "dtype":
        inputs["template"] = inputs["template"].double()
    elif corruption == "shape":
        inputs["template"] = torch.zeros(2,127,127,3)
    elif corruption == "batch":
        inputs["template"] = inputs["template"][:1]
    elif corruption == "sparse":
        inputs["template"] = inputs["template"].to_sparse()
    elif corruption == "nonfinite":
        inputs["template"].fill_(float("inf"))
    else:
        inputs["template"] = torch.empty(2,3,127,127,device="meta")
    with pytest.raises(ValueError):
        canonicalize_model_inputs(inputs)


def test_shared_noncontiguous_numpy_input_controller_boundary_replay(tmp_path):
    from hift_research.training.wave4_pilot_engine import SerialPilotController
    dataset = SyntheticCandidateDataset(length=4, channels_last_arrays=True)
    def build():
        return SerialPilotController(B2TrainingGraph(), dataset, config_identity={"fixture": "authentic-layout-regression"},
                    data_identity={"kind":"synthetic",**dataset_identity(dataset)}, physical=2, attempts_per_window=2,
                    max_updates=2, max_attempts=4)
    full = build().run_to_end()
    rng = device_rng(torch.device("cpu"))
    seed_device(20261023,"cpu")
    part = build()
    part.step()
    checkpoint = part.save_boundary(tmp_path/"layout-boundary.pt")
    with pytest.raises(RuntimeError, match="midwindow"):
        part.step(crash_after_micro=1)
    restored = build().restore_boundary(checkpoint["path"], expected_sha256=checkpoint["sha256"]).run_to_end()
    assert tree_equal(full.model.state_dict(), restored.model.state_dict())
    assert tree_equal(full.optimizer.state_dict(), restored.optimizer.state_dict())
    assert tree_equal(full.semantic_trace, restored.semantic_trace)
    assert tree_equal(rng,device_rng(torch.device("cpu")))


def test_shared_final_partial_window18_attempts_uses_actual_two_sample_denominators_and_single_clip_step(monkeypatch):
    from hift_research.training import wave4_pilot
    from hift_research.training.lifecycle import actual_heads, capture_rng, restore_rng, window_loss
    from hift_research.training.wave4_pilot_engine import SerialPilotController
    dataset = SyntheticCandidateDataset(length=18, channels_last_arrays=True)
    controller = SerialPilotController(B2TrainingGraph(), dataset, config_identity={"fixture":"18-attempt-tail"},
                   data_identity={"kind":"synthetic",**dataset_identity(dataset)}, physical=4,
                   attempts_per_window=16,max_updates=2,max_attempts=18)
    counts = {"zero":0,"step":0,"clip":0}
    zero_original, step_original = controller.optimizer.zero_grad, controller.optimizer.step
    clip_original = torch.nn.utils.clip_grad_norm_
    buffered_original = wave4_pilot.buffered_attempt_window
    observed = []
    def zero(*args,**kwargs):
        counts["zero"] += 1
        return zero_original(*args,**kwargs)
    def step(*args,**kwargs):
        counts["step"] += 1
        return step_original(*args,**kwargs)
    def clip(*args,**kwargs):
        counts["clip"] += 1
        return clip_original(*args,**kwargs)
    def buffer(*args,**kwargs):
        batches,denominators = buffered_original(*args,**kwargs)
        observed.append((copy.deepcopy(denominators),[len(b["accepted_records"]) for b in batches]))
        return batches,denominators
    monkeypatch.setattr(controller.optimizer,"zero_grad",zero)
    monkeypatch.setattr(controller.optimizer,"step",step)
    monkeypatch.setattr(torch.nn.utils,"clip_grad_norm_",clip)
    monkeypatch.setattr(wave4_pilot,"buffered_attempt_window",buffer)
    first = controller.step()
    # Reference is the same two-element physical tail, not a fictitious four-microbatch divisor.
    rng = capture_rng()
    reference = copy.deepcopy(controller.model)
    reference.zero_grad(set_to_none=True)
    tail,denominators = buffered_original(dataset,[16,17],4,kind="synthetic")
    loss = window_loss(actual_heads(reference,tail[0]),tail[0]["targets"],denominators)
    loss.backward()
    clip_original(reference.parameters(),10.0)
    restore_rng(rng)
    second = controller.step()
    assert first["semantic"]["effective_accepted"] == 16
    assert second["semantic"]["effective_accepted"] == 2 and controller.cursor == 18 and controller.update == 2
    assert observed[0][0]["bce"] == 16*121 and observed[1][0]["bce"] == 2*121
    assert observed[0][1] == [4,4,4,4] and observed[1][1] == [2]
    assert second["semantic"]["loss"] == float(loss.detach())
    assert tree_equal({n:p.grad for n,p in reference.named_parameters()}, {n:p.grad for n,p in controller.model.named_parameters()})
    assert counts == {"zero":2,"step":2,"clip":2}
    assert controller.step() is None


def test_checkpoint_rejects_momentum_rng_gradient_source_identity_corruption(tmp_path):
    engine = make_engine()
    for _ in range(3):
        engine.microstep()
    original = tmp_path/"original.pt"
    engine.save(original)
    state = torch.load(original, weights_only=True)
    first = next(iter(state["optimizer"]["state"].values()))
    first["momentum_buffer"].fill_(float("nan"))
    bad = tmp_path/"bad.pt"
    torch.save(state, bad)
    with pytest.raises(ValueError, match="momentum"):
        make_engine().restore(bad)
