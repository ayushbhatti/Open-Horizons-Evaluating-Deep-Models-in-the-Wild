# Frozen ImageNet Backbone Loaders
import torch.nn as nn
import torchvision

def build_backbone(name):
    name = name.lower()
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    for p in m.parameters(): p.requires_grad = False
    return m, feat_dim