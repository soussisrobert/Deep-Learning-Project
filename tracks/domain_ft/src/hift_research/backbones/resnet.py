"""ResNet-18 backbone for the B2 HiFT head (a new architecture variant, not HiFT/B0/B2-AlexNet).

Trunk keys match torchvision's resnet18 so the official ImageNet file loads strictly. SiamRPN++
style: layer3/layer4 strides replaced by dilation 2/4, so layer2..4 all have stride 8. A trainable
1x1 neck maps 128/256/512 channels to the 384/384/256 the HiFT fusion expects, and maps are
center-cropped to AlexNet's sizes (template 10/8/6, search 30/28/26) so every depthwise
correlation stays 21x21 with cell centers aligned to the label grid (8j+63.5 vs 8j+63 px).
Inputs are the pipeline's BGR 0..255 crops; ImageNet RGB normalization happens inside.
"""
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from hift_research.corrections.tensor_layout import encoder_forward_corrected
from hift_research.training.graph import B2TrainingGraph
from hift_research.training.upstream import classes

TEMPLATE_CROPS, SEARCH_CROPS = (10, 8, 6), (30, 28, 26)
OUT_CHANNELS = (384, 384, 256)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + identity)


def _layer(inplanes, planes, stride, dilation):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    first_dilation = max(1, dilation//2)
    return nn.Sequential(BasicBlock(inplanes, planes, stride, first_dilation, downsample), BasicBlock(planes, planes, 1, dilation))


class ResNet18Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _layer(64, 64, 1, 1)
        self.layer2 = _layer(64, 128, 2, 1)
        self.layer3 = _layer(128, 256, 1, 2)
        self.layer4 = _layer(256, 512, 1, 4)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)*255, persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)*255, persistent=False)

    def forward(self, bgr):
        x = (bgr.flip(1)-self.mean)/self.std
        x = self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        return c2, c3, self.layer4(c3)


def load_imagenet_resnet18(trunk, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    trunk.load_state_dict(state, strict=True)
    return trunk


def center_crop(feature, size):
    h, w = feature.shape[-2:]
    top, left = (h-size)//2, (w-size)//2
    if top < 0 or left < 0:
        raise ValueError("feature map smaller than crop")
    return feature[..., top:top+size, left:left+size]


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList(nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
                                  for c_in, c_out in zip((128, 256, 512), OUT_CHANNELS))

    def forward(self, features, crops):
        return [center_crop(p(f), s) for p, f, s in zip(self.proj, features, crops)]


class B2ResNetGraph(B2TrainingGraph):
    """Same corrected HiFT grader/losses as B2TrainingGraph; ResNet-18 trunk + neck instead of AlexNet."""
    def __init__(self):
        nn.Module.__init__(self)
        _, hift, _ = classes()
        self.backbone = ResNet18Trunk()
        self.neck = Neck()
        self.grader = hift(SimpleNamespace(TRAIN=SimpleNamespace(groupchannel=32)))
        for layer in self.grader.transformer.encoder.layers:
            layer.forward = MethodType(encoder_forward_corrected, layer)
        self.train()

    def train(self, mode=True):
        nn.Module.train(self, mode)
        return self

    def features(self, template, search):
        return self.neck(self.backbone(template), TEMPLATE_CROPS), self.neck(self.backbone(search), SEARCH_CROPS)


def apply_resnet_epoch_modes(model, epoch, unfreeze_epoch=10):
    """Mirror of the source policy: trunk frozen (BN eval) before unfreeze_epoch, then layer2-4 train.
    The new neck and the grader always train."""
    model.train()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if epoch >= unfreeze_epoch:
        for layer in (model.backbone.layer2, model.backbone.layer3, model.backbone.layer4):
            for p in layer.parameters():
                p.requires_grad_(True)
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    return model


def resnet_optimizer(model, base_lr, backbone_ratio=0.1):
    return torch.optim.SGD([
        {"name": "backbone", "params": list(model.backbone.parameters()), "lr": base_lr*backbone_ratio, "initial_lr": base_lr*backbone_ratio},
        {"name": "grader", "params": list(model.neck.parameters())+list(model.grader.parameters()), "lr": base_lr, "initial_lr": base_lr},
    ], lr=base_lr, momentum=0.9, weight_decay=0.0001)


class GraphTracker:
    """template()/track() interface the upstream HiFTTracker expects, for any B2 graph."""
    def __init__(self, graph):
        self.graph = graph

    def eval(self):
        self.graph.eval()
        return self

    def template(self, z):
        with torch.no_grad():
            self.zf = self.graph.neck(self.graph.backbone(z), TEMPLATE_CROPS) if hasattr(self.graph, "neck") else self.graph.backbone(z)

    def track(self, x):
        with torch.no_grad():
            g = self.graph
            xf = g.neck(g.backbone(x), SEARCH_CROPS) if hasattr(g, "neck") else g.backbone(x)
            loc, cls1, cls2 = g.grader(xf, self.zf)
            return {"cls1": cls1, "cls2": cls2, "loc": loc}
