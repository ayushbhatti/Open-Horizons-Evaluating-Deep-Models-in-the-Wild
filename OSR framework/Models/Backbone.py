import torch.nn as nn
import torchvision

try:
    import open_clip
except ImportError:
    open_clip = None


def build_backbone(name: str):
    name = name.lower()

    if name == "resnet50":
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()

    elif name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        feat_dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()

    elif name in ["clip_vit_b16", "clip-vit-b16", "clip"]:
        assert open_clip is not None, (
            "open_clip_torch is not installed. "
            "Install it with `pip install open_clip_torch`."
        )
        # OpenAI CLIP ViT-B/16
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        m = model.visual           # visual encoder only
        feat_dim = m.output_dim    # CLIP visual feature dimension

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # freeze backbone
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m, feat_dim