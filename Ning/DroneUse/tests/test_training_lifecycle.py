"""Synthetic CPU lifecycle verification, never real-data optimization or CUDA."""
import ast
import copy
import math
import random
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from hift_research.training.graph import B2TrainingGraph, gradient_audit, synthetic_batch
from hift_research.training.lifecycle import (
    EpochSchedule, SyntheticLifecycle, SyntheticSampler, apply_epoch_modes, capture_rng,
    lifecycle_optimizer, restore_rng, seed_cpu, window_denominators, window_loss,
)
from hift_research.training.recipe import (
    ROOT, bind_training_manifest, canonical_hash, make_proposed_recipe, source_lr_table,
    source_settings, validate_recipe_binding, validate_proposed_config, load_legacy_backbone_strict,
)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        for index in range(1, 6):
            setattr(self, f"layer{index}", nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.ReLU()))

    def forward(self, x):
        for index in range(1, 6):
            x = getattr(self, f"layer{index}")(x)
        return x


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()
        self.grader = nn.Sequential(nn.Dropout2d(0.25), nn.Conv2d(3, 7, 1))


def toy_heads(model, batch):
    value = model.grader(model.backbone(batch["template"]))
    return torch.tanh(value[:, :4]+0.6), value[:, 4:6], value[:, 6:7]


def toy_factory(ids):
    # All three RNGs are intentionally consumed, including before prefetch completes.
    value = torch.rand(len(ids), 3, 2, 2) + random.random() + float(np.random.random())
    labels = torch.tensor([[[[1., 0.], [-1., 0.]]] if i % 2 == 0 else [[[0., 0.], [-2., 0.]]]
                           for i in ids])
    weights = torch.tensor([[[[64., 0.], [0., 1.5]]] if i % 2 == 0 else [[[0., 0.], [0., 0.]]]
                            for i in ids])
    return {"data_kind": "synthetic", "sample_ids": ids, "template": value,
            "targets": {"label_cls1": labels, "labelxff": torch.full((len(ids), 4, 2, 2), 0.25),
                        "labelcls2": (labels == 1).float(), "weightxff": weights}}


def actual_factory(ids):
    template, search, targets = synthetic_batch(len(ids), negative=[bool(i % 3 == 0) for i in ids])
    scale = 0.9+random.random()*0.05+float(np.random.random())*0.05
    return {"data_kind": "synthetic", "sample_ids": ids, "template": template*scale,
            "search": search, "targets": targets}


def runner(model=None, *, epoch=9, ids=(4, 0, 3, 2, 1), factory=toy_factory, heads=toy_heads,
           accumulation=2, microbatch=1, config_identity=None, data_identity=None):
    return SyntheticLifecycle(model or Toy(), SyntheticSampler(ids, epoch=epoch),
                              batch_factory=factory, heads=heads, epoch=epoch,
                              microbatch=microbatch, accumulation=accumulation,
                              config_identity=config_identity or {"recipe": "synthetic-test-v1"},
                              data_identity=data_identity or {"kind": "synthetic", "sha256": canonical_hash(list(ids))})


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.fixture(autouse=True)
def deterministic_cpu():
    seed_cpu(20261023)


def test_actual_source_settings_and_lr_normalization():
    settings = source_settings()
    assert settings["TRAIN"]["EPOCH"] == 70
    assert settings["TRAIN"]["BATCH_SIZE"] == 220
    assert settings["TRAIN"]["MOMENTUM"] == 0.9
    assert settings["TRAIN"]["GRAD_CLIP"] == 10
    assert settings["DATASET"]["VIDEOS_PER_EPOCH"] == 600000
    assert settings["DATASET"]["NEG"] == 0.05
    assert settings["DATASET"]["SEARCH"]["BLUR"] == 0.2
    table = source_lr_table(settings)
    assert len(table) == 70
    assert table[0] == 0.0025 and table[5] == 0.005
    assert table[4] == pytest.approx(0.0025*2**(4/5))
    assert table[-1] == pytest.approx(0.00005)
    # Execute actual pinned scheduler module minus only its cfg import/main block.
    path = ROOT/"third_party/HiFT/pysot/utils/lr_scheduler.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "pysot.core.config")
                 and not isinstance(node, ast.If)]
    def namespace(value):
        return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) and k != "KWARGS" else v
                                  for k, v in value.items()})
    env = {"cfg": namespace(settings)}
    exec(compile(tree, str(path), "exec"), env)
    model = Toy()
    optimizer = lifecycle_optimizer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schedule = env["build_lr_scheduler"](optimizer, epochs=70)
        for epoch, expected in enumerate(table):
            schedule.step(epoch)
            assert optimizer.param_groups[1]["lr"] == expected
            assert optimizer.param_groups[0]["lr"] == pytest.approx(expected*0.1)


def test_actual_source_epoch10_rebuild_discards_momentum_and_bn_entry_path_differs():
    settings = source_settings()
    def namespace(value):
        return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) and k != "KWARGS" else v
                                  for k, v in value.items()})
    cfg = namespace(settings)
    schedule_path = ROOT/"third_party/HiFT/pysot/utils/lr_scheduler.py"
    schedule_tree = ast.parse(schedule_path.read_text())
    schedule_tree.body = [n for n in schedule_tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "pysot.core.config")
                          and not isinstance(n, ast.If)]
    env = {"cfg": cfg, "nn": nn, "torch": torch}
    exec(compile(schedule_tree, str(schedule_path), "exec"), env)
    path = ROOT/"third_party/HiFT/tools/train.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "build_opt_lr")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    model = Toy().train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer, _ = env["build_opt_lr"](model, 0)
        assert len(optimizer.param_groups[0]["params"]) == 0
        sum(p.sum() for p in model.grader.parameters()).backward()
        optimizer.step()
        assert optimizer.state
        rebuilt, schedule = env["build_opt_lr"](model, 10)
        schedule.step(10)
        assert not rebuilt.state
        assert len(rebuilt.param_groups[0]["params"]) == 12
        assert not model.backbone.layer1[1].training
        fresh = Toy().train()
        env["build_opt_lr"](fresh, 10)
        assert fresh.backbone.layer1[1].training  # fresh source resume differs from uninterrupted mode
    apply_epoch_modes(fresh, 10)
    assert not fresh.backbone.layer1[1].training


def test_epoch_not_update_schedule_and_unfreeze_preserves_momentum():
    run = runner(ids=(1, 2, 3))
    before_bn = run.model.backbone.layer3[1].running_mean.clone()
    run.run_to_end()
    assert all(not p.requires_grad for p in run.model.backbone.parameters())
    assert torch.equal(before_bn, run.model.backbone.layer3[1].running_mean)
    momentum = run.optimizer.state[run.model.grader[1].weight]["momentum_buffer"].clone()
    assert run.scheduler.epoch == 9 and run.update == 2
    before_lr = run.optimizer.param_groups[1]["lr"]
    run.start_next_epoch([4, 5])
    assert run.epoch == run.scheduler.epoch == run.sampler.epoch == 10
    assert run.optimizer.param_groups[1]["lr"] != before_lr
    assert torch.equal(momentum, run.optimizer.state[run.model.grader[1].weight]["momentum_buffer"])
    assert all(p.requires_grad for p in run.model.backbone.layer3.parameters())
    assert all(not p.requires_grad for p in run.model.backbone.layer1.parameters())
    assert not run.model.backbone.layer1[1].training and run.model.backbone.layer3[1].training
    run.run_to_end()
    assert not torch.equal(before_bn, run.model.backbone.layer3[1].running_mean)


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
@pytest.mark.parametrize("epoch", [0, 4, 5, 9, 10, 69])
def test_bitexact_interrupted_resume_with_pending_accumulation_rng_sampler_and_partial(tmp_path, cut, epoch):
    seed_cpu(20261023)
    full = runner(epoch=epoch)
    full.run_to_end()
    expected_rng = capture_rng()
    seed_cpu(20261023)
    interrupted = runner(epoch=epoch)
    for _ in range(cut):
        interrupted.microstep()
    path = tmp_path/f"resume-{cut}.pt"
    interrupted.save(path)
    with pytest.raises(FileExistsError):
        interrupted.save(path)
    restored = runner(epoch=epoch)
    restored.restore(path)
    restored.run_to_end()
    assert_tree_equal(full.trace, restored.trace)
    assert_tree_equal(full.model.state_dict(), restored.model.state_dict())
    assert_tree_equal(full.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_tree_equal(full.scheduler.state_dict(), restored.scheduler.state_dict())
    assert_tree_equal(full.sampler.state_dict(), restored.sampler.state_dict())
    assert_tree_equal(expected_rng, capture_rng())
    assert_tree_equal({name: p.grad for name, p in full.model.named_parameters()},
                      {name: p.grad for name, p in restored.model.named_parameters()})
    assert restored.update == 3 and restored.microsteps == 5  # final singleton steps once
    assert [i for row in restored.trace for i in row["sample_ids"]] == [4, 0, 3, 2, 1]
    assert restored.trace[-1]["stepped"]


def test_global_class_mask_denominators_not_average_of_microbatch_means():
    batches = [toy_factory([2]), toy_factory([1, 3])]
    batches[1]["targets"]["weightxff"][0, 0, 1, 1] = 3.0  # unequal active weights in BOTH micros
    totals = window_denominators(batches)
    assert totals == {"positive": 1, "negative": 8, "bce": 12, "localization": 68.5}
    heads = []
    for b in (1, 2):
        heads.append((torch.full((b, 4, 2, 2), 0.3, dtype=torch.float64, requires_grad=True),
                      torch.randn(b, 2, 2, 2, dtype=torch.float64, requires_grad=True),
                      torch.randn(b, 1, 2, 2, dtype=torch.float64, requires_grad=True)))
    for batch in batches:
        batch["targets"] = {k: v.double() for k, v in batch["targets"].items()}
    terms = [window_loss(h, batch["targets"], totals) for h, batch in zip(heads, batches)]
    merged_heads = tuple(torch.cat([h[i] for h in heads]) for i in range(3))
    merged_targets = {k: torch.cat([batch["targets"][k] for batch in batches]) for k in batches[0]["targets"]}
    merged = window_loss(merged_heads, merged_targets, window_denominators([{"targets": merged_targets}]))
    torch.testing.assert_close(sum(terms), merged, rtol=1e-12, atol=1e-12)
    wrong = sum(window_loss(h, batch["targets"], window_denominators([batch])) for h, batch in zip(heads, batches))/2
    assert abs(float((wrong-merged).detach())) > 1e-3
    summed_grads = torch.autograd.grad(sum(terms), [v for h in heads for v in h], retain_graph=True)
    merged_grads = torch.autograd.grad(merged, [v for h in heads for v in h])
    for a, b in zip(summed_grads, merged_grads):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_float32_global_denominator_rounding_is_toleranced_not_bitexact_batch_claim():
    batches = [toy_factory([2]), toy_factory([0, 3])]
    batches[1]["targets"]["weightxff"][0, 0, 0, 0] = 1e-4
    totals = window_denominators(batches)
    heads = [(torch.full((b, 4, 2, 2), 0.3, requires_grad=True),
              torch.randn(b, 2, 2, 2, requires_grad=True), torch.randn(b, 1, 2, 2, requires_grad=True))
             for b in (1, 2)]
    accumulated = sum(window_loss(h, batch["targets"], totals) for h, batch in zip(heads, batches))
    merged_heads = tuple(torch.cat([h[i] for h in heads]) for i in range(3))
    merged_targets = {k: torch.cat([batch["targets"][k] for batch in batches]) for k in batches[0]["targets"]}
    merged = window_loss(merged_heads, merged_targets, window_denominators([{"targets": merged_targets}]))
    torch.testing.assert_close(accumulated, merged, rtol=1e-5, atol=1e-6)
    a = torch.autograd.grad(accumulated, [v for h in heads for v in h], retain_graph=True)
    b = torch.autograd.grad(merged, [v for h in heads for v in h])
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_bn_and_dropout_make_microbatch_equivalence_false():
    layer = nn.BatchNorm2d(2).train()
    other = copy.deepcopy(layer)
    values = torch.arange(32, dtype=torch.float32).reshape(4, 2, 2, 2)
    whole = layer(values)
    split = torch.cat([other(values[:1]), other(values[1:])])
    assert not torch.allclose(whole, split)
    assert not torch.equal(layer.running_mean, other.running_mean)
    drop = nn.Dropout(0.5).train()
    seed_cpu(17)
    one = drop(torch.ones(4, 3))
    seed_cpu(17)
    two = torch.cat([drop(torch.ones(1, 3)), drop(torch.ones(3, 3))])
    # Outputs may agree on some implementations; RNG partition equivalence is NOT promised.
    assert one.shape == two.shape


def test_actual_b2_graph_freeze_unfreeze_gamma_and_midaccumulation_resume(tmp_path):
    def actual_runner():
        return runner(B2TrainingGraph(), epoch=10, ids=(1, 2, 4), factory=actual_factory,
                      heads=lambda model, batch: model.grader(*reversed(model.features(batch["template"], batch["search"]))))
    seed_cpu(20261023)
    full = actual_runner()
    assert full.model.grader.transformer.encoder.layers[0].cross_attn.gamma.item() == 0
    full.microstep()
    audit = gradient_audit(full.model)
    assert len(audit["expected_initial_gamma_zero"]) == 4 and len(audit["expected_dormant"]) == 4
    full.run_to_end()
    seed_cpu(20261023)
    interrupted = actual_runner()
    interrupted.microstep()
    checkpoint = tmp_path/"actual-mid.pt"
    interrupted.save(checkpoint)
    restored = actual_runner().restore(checkpoint)
    restored.run_to_end()
    assert_tree_equal(full.trace, restored.trace)
    assert_tree_equal(full.model.state_dict(), restored.model.state_dict())
    assert_tree_equal(full.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_tree_equal({name: p.grad for name, p in full.model.named_parameters()},
                      {name: p.grad for name, p in restored.model.named_parameters()})
    assert not gradient_audit(restored.model)["nonfinite"]
    assert restored.model.grader.transformer.encoder.layers[0].cross_attn.gamma.item() != 0
    apply_epoch_modes(restored.model, 0)
    assert all(not p.requires_grad for p in restored.model.backbone.parameters())
    assert all(not m.training for m in restored.model.backbone.modules() if isinstance(m, nn.BatchNorm2d))


def test_checkpoint_rejects_wrong_identity_and_invalid_model_before_resume(tmp_path):
    run = runner()
    run.microstep()
    path = tmp_path/"original.pt"
    run.save(path)
    with pytest.raises(ValueError, match="identity"):
        runner(config_identity={"recipe": "different"}).restore(path)
    with pytest.raises(ValueError, match="identity"):
        runner(data_identity={"kind": "synthetic", "sha256": "different"}).restore(path)
    state = torch.load(path, weights_only=True)
    state["model"]["grader.1.weight"].fill_(float("nan"))
    bad = tmp_path/"nan.pt"
    torch.save(state, bad)
    with pytest.raises(ValueError, match="model shape/dtype/finite"):
        runner().restore(bad)
    state = torch.load(path, weights_only=True)
    state["pending"]["denominators"]["positive"] += 1
    bad2 = tmp_path/"counts.pt"
    torch.save(state, bad2)
    with pytest.raises(ValueError, match="denominators"):
        runner().restore(bad2)


@pytest.mark.parametrize("corruption", ["momentum_nan", "momentum_shape", "momentum_dtype", "dampening", "nesterov", "maximize", "gradient_dtype", "pending_ids", "source_identity"])
def test_malformed_optimizer_and_pending_checkpoints_fail_before_accepted(tmp_path, corruption):
    run = runner()
    for _ in range(3):
        run.microstep()
    original = tmp_path/"original.pt"
    run.save(original)
    state = torch.load(original, weights_only=True)
    if corruption.startswith("momentum_"):
        first = next(iter(state["optimizer"]["state"].values()))
        value = first["momentum_buffer"]
        first["momentum_buffer"] = (torch.full_like(value, float("nan")) if corruption == "momentum_nan"
                                    else value.reshape(-1)[:1] if corruption == "momentum_shape" else value.double())
    elif corruption in ("dampening", "nesterov", "maximize"):
        state["optimizer"]["param_groups"][0][corruption] = 0.1 if corruption == "dampening" else True
    elif corruption == "gradient_dtype":
        name = next(name for name, value in state["gradients"].items() if value is not None)
        state["gradients"][name] = state["gradients"][name].double()
    elif corruption == "pending_ids":
        state["pending"]["batches"][0]["sample_ids"] = [999]
    else:
        state["code_identity"]["src/hift_research/training/lifecycle.py"] = "changed"
    bad = tmp_path/"bad.pt"
    torch.save(state, bad)
    with pytest.raises(ValueError):
        runner().restore(bad)


def test_legacy_backbone_counter_only_compatibility_is_strict_and_preserves_all_source_tensors():
    path = ROOT/"third_party/HiFT/pretrained_models/alexnet-bn.pth"
    backbone = B2TrainingGraph().backbone
    result = load_legacy_backbone_strict(backbone, path)
    assert result["sha256"] == "933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8"
    assert len(result["inserted_zero_counters"]) == 5
    assert not result["strict_raw_keys_match"] and result["strict_normalized_load"]
    assert result["all_original_tensors_bitexact"] and result["checkpoint_file_unchanged"]
    assert result["loaded_for_optimization"] is False


@pytest.mark.parametrize("corruption", ["learned_missing", "unexpected", "shape", "dtype", "partial_counters", "cumulative_bn"])
def test_counter_compatibility_refuses_any_nonapproved_state(tmp_path, corruption):
    path = ROOT/"third_party/HiFT/pretrained_models/alexnet-bn.pth"
    payload = torch.load(path, weights_only=True)
    weights = payload.get("state_dict", payload)
    if corruption == "learned_missing":
        weights.pop("layer1.0.weight")
    elif corruption == "unexpected":
        weights["unapproved.weight"] = torch.zeros(1)
    elif corruption == "shape":
        weights["layer1.0.weight"] = weights["layer1.0.weight"][:1]
    elif corruption == "dtype":
        weights["layer1.0.weight"] = weights["layer1.0.weight"].double()
    elif corruption == "partial_counters":
        weights["layer1.1.num_batches_tracked"] = torch.zeros((), dtype=torch.int64)
    candidate = tmp_path/"candidate.pt"
    torch.save(weights, candidate)
    backbone = B2TrainingGraph().backbone
    if corruption == "cumulative_bn":
        backbone.layer1[1].momentum = None
    with pytest.raises(ValueError):
        load_legacy_backbone_strict(backbone, candidate)


def test_config_validation_rejects_changed_numeric_recipe():
    import json
    config = json.loads((ROOT/"configs/B2-wave3-proposed.json").read_text())
    # Unit tests must remain runnable without ignored authentic data/artifacts.
    # Actual frozen-file binding is exercised separately by the verification CLI.
    config["real_training_manifest"] = config["real_training_manifest_sha256"] = None
    validate_proposed_config(config)
    config["proposed_source_preserving_settings"]["momentum"] = 0.1
    with pytest.raises(ValueError, match="numeric"):
        validate_proposed_config(config)


def test_real_data_and_unfrozen_recipe_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        runner(data_identity={"kind": "real_training", "sha256": "anything"})
    recipe = make_proposed_recipe()
    assert recipe["launch_allowed"] is False and recipe["input_contract"]["candidate_negative_probability"] == 0.05
    with pytest.raises(ValueError, match="provisional"):
        validate_recipe_binding(recipe)
    wrong = tmp_path/"eval.json"
    wrong.write_text('{"schema":"tracking-evaluation","sequences":["one"]}')
    with pytest.raises(ValueError, match="canonical"):
        bind_training_manifest(recipe, wrong)


def test_clip_and_step_only_at_window_boundary(monkeypatch):
    run = runner()
    calls = {"clip": 0, "step": 0}
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = run.optimizer.step
    def clip(*args, **kwargs):
        calls["clip"] += 1
        return original_clip(*args, **kwargs)
    def step(*args, **kwargs):
        calls["step"] += 1
        return original_step(*args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(run.optimizer, "step", step)
    run.microstep()
    assert calls == {"clip": 0, "step": 0}
    run.run_to_end()
    assert calls == {"clip": 3, "step": 3}
    assert run.scheduler.epoch == 9


def test_restricted_loader_refuses_pickled_python_objects(tmp_path):
    import pickle
    path = tmp_path/"object.pt"
    torch.save({"untrusted_object": SimpleNamespace(code="not executed")}, path)
    with pytest.raises(pickle.UnpicklingError):
        runner().restore(path)


def test_reject_epoch_switch_midwindow_and_partial_not_dropped():
    run = runner()
    run.microstep()
    with pytest.raises(ValueError, match="pending"):
        run.set_epoch(10)
    with pytest.raises(ValueError, match="finish"):
        run.start_next_epoch([6])
    run.run_to_end()
    assert run.update == 3
    run.start_next_epoch([6])
    assert run.scheduler.epoch == 10
