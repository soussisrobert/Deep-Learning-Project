from pathlib import Path

import pytest
import torch

from hift_research.backbones.resnet import (B2ResNetGraph, GraphTracker, ResNet18Trunk, apply_resnet_epoch_modes,
                                            load_imagenet_resnet18, resnet_optimizer)
from hift_research.training.graph import synthetic_batch

WEIGHTS = Path(__file__).resolve().parents[1]/"artifacts/pretrained/resnet18-f37072fd.pth"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="official ImageNet weights not present")
def test_official_weights_load_strictly():
    trunk = load_imagenet_resnet18(ResNet18Trunk(), WEIGHTS)
    assert float(trunk.conv1.weight.abs().sum()) > 0


def test_features_match_alexnet_contract():
    torch.manual_seed(0)
    graph = B2ResNetGraph().eval()
    z, x = graph.features(torch.rand(2, 3, 127, 127)*255, torch.rand(2, 3, 287, 287)*255)
    assert [tuple(t.shape) for t in z] == [(2, 384, 10, 10), (2, 384, 8, 8), (2, 256, 6, 6)]
    assert [tuple(t.shape) for t in x] == [(2, 384, 30, 30), (2, 384, 28, 28), (2, 256, 26, 26)]
    loc, cls1, cls2 = graph.grader(x, z)
    assert loc.shape == (2, 4, 11, 11) and cls1.shape == (2, 2, 11, 11) and cls2.shape == (2, 1, 11, 11)


def test_trains_neck_and_grader_with_trunk_frozen_until_unfreeze():
    torch.manual_seed(0)
    graph = B2ResNetGraph()
    apply_resnet_epoch_modes(graph, 0)
    template, search, targets = synthetic_batch(2)
    graph.losses(graph.grader(*reversed(graph.features(template, search))), targets)["total_loss"].backward()
    assert all(p.grad is None for p in graph.backbone.parameters())
    assert all(p.grad is not None for p in graph.neck.parameters())
    apply_resnet_epoch_modes(graph, 10)
    assert all(p.requires_grad for p in graph.backbone.layer2.parameters())
    assert not any(p.requires_grad for p in graph.backbone.layer1.parameters())
    groups = resnet_optimizer(graph, 0.005).param_groups
    assert groups[0]["lr"] == pytest.approx(0.0005) and groups[1]["lr"] == pytest.approx(0.005)


def test_graph_tracker_matches_training_graph():
    torch.manual_seed(1)
    graph = B2ResNetGraph().eval()
    z, x = torch.rand(1, 3, 127, 127)*255, torch.rand(1, 3, 287, 287)*255
    with torch.no_grad():
        expected = graph.grader(*reversed(graph.features(z, x)))
    wrapper = GraphTracker(graph).eval()
    wrapper.template(z)
    out = wrapper.track(x)
    for name, value in zip(("loc", "cls1", "cls2"), expected):
        assert torch.equal(out[name], value)
