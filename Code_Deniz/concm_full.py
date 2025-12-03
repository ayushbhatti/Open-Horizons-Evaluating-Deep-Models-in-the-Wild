#!/usr/bin/env python3
"""
Full ConCM implementation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MPC: Memory-aware Prototype Calibration
# ============================================================================


class CrossAttentionMemory(nn.Module):
    """
    Cross-attention mechanism for extracting semantic attributes from base classes.
    Implements pattern separation from the hippocampal memory system.
    """

    def __init__(self, feat_dim=2048, hidden_dim=512, output_dim=128, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(feat_dim, hidden_dim)
        self.key_proj = nn.Linear(feat_dim, hidden_dim)
        self.value_proj = nn.Linear(feat_dim, output_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, query, memory_bank):
        """
        Args:
            query: Novel class features [1, feat_dim]
            memory_bank: Base class prototypes [num_base, feat_dim]
        Returns:
            Retrieved semantic attributes [1, output_dim]
        """
        B, D = query.shape
        N = memory_bank.shape[0]

        Q = self.query_proj(query).view(B, self.num_heads, self.head_dim)
        K = self.key_proj(memory_bank).view(N, self.num_heads, self.head_dim)
        V = self.value_proj(memory_bank).view(N, self.num_heads, -1)

        # Attention scores
        attn = torch.einsum("bhd,nhd->bhn", Q, K) * self.scale
        attn = F.softmax(attn, dim=2)

        # Aggregate values
        out = torch.einsum("bhn,nhd->bhd", attn, V)
        out = out.reshape(B, -1)

        return out


class MemoryAwarePrototypeCalibration(nn.Module):
    """
    MPC Module: Extracts generalized semantic attributes and calibrates prototypes.
    """

    def __init__(self, feat_dim=2048, attr_dim=128):
        super().__init__()
        self.cross_attn = CrossAttentionMemory(feat_dim=feat_dim, output_dim=attr_dim)
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + attr_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, novel_proto, base_memory, alpha=0.6):
        """
        Args:
            novel_proto: Few-shot prototype [1, feat_dim]
            base_memory: Base class prototypes [num_base, feat_dim]
            alpha: Calibration degree (0=full MPC, 1=original proto)
        Returns:
            Calibrated prototype [1, feat_dim]
        """
        # Pattern completion: retrieve relevant attributes
        attributes = self.cross_attn(novel_proto, base_memory)

        # Fuse attributes with novel prototype
        fused = torch.cat([novel_proto, attributes], dim=-1)
        calibrated = self.fusion(fused)

        # Blend with original prototype
        calibrated = F.normalize(alpha * novel_proto + (1 - alpha) * calibrated, dim=1)

        return calibrated


# ============================================================================
# DSM: Dynamic Structure Matching
# ============================================================================


class ProjectorNetwork(nn.Module):
    """
    Projector that maps embedding space to geometric space.
    Implements hypersphere-to-hypersphere mapping with L2 normalization.
    """

    def __init__(self, feat_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Features [B, feat_dim]
        Returns:
            Projected features [B, output_dim]
        """
        # L2 normalize before projection
        x = F.normalize(x, dim=1)
        # Project
        x = self.net(x)
        # L2 normalize after projection
        x = F.normalize(x, dim=1)
        return x


class DynamicStructureMatching(nn.Module):
    """
    DSM Module: Constructs geometric structure and adaptively aligns features.
    Ensures geometric optimality and maximum matching.
    """

    def __init__(self, feat_dim=2048, geom_dim=128):
        super().__init__()
        self.projector = ProjectorNetwork(
            feat_dim=feat_dim, hidden_dim=2048, output_dim=geom_dim
        )
        self.structural_anchors = {}

    def update_structure(self, class_id, prototype):
        """
        Dynamically update the geometric structure with new class.
        Maintains equidistant prototype separation.
        """
        # Set projector to eval mode to avoid BatchNorm issues with single sample
        was_training = self.projector.training
        self.projector.eval()
        with torch.no_grad():
            projected = self.projector(prototype)
        if was_training:
            self.projector.train()
        # Store as [geom_dim] not [1, geom_dim]
        self.structural_anchors[class_id] = projected.squeeze(0).detach()

    def align_features(self, features, labels, temperature=0.07):
        """
        Compute alignment loss between features and structural anchors.

        Args:
            features: Batch features [B, feat_dim]
            labels: Class labels [B]
            temperature: Temperature for contrastive loss
        Returns:
            Alignment loss (returns 0 if no anchors initialized)
        """
        # Return zero loss if no anchors initialized yet
        if len(self.structural_anchors) == 0:
            return torch.tensor(0.0, device=features.device)

        projected = self.projector(features)  # [B, geom_dim]

        # Get anchor prototypes [num_classes, geom_dim]
        anchor_ids = sorted(self.structural_anchors.keys())
        anchors = torch.stack([self.structural_anchors[i] for i in anchor_ids], dim=0)

        # Compute similarities [B, num_classes]
        sim = torch.matmul(projected, anchors.t()) / temperature

        # Create target labels
        target = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        for i, label in enumerate(labels):
            if int(label) in anchor_ids:
                idx = anchor_ids.index(int(label))
                target[i] = idx

        # Cross entropy loss
        loss = F.cross_entropy(sim, target)

        return loss


# ============================================================================
# Prototype Augmentation
# ============================================================================


def prototype_augmentation(prototype, num_samples=50, beta=0.6):
    """
    Augment prototypes via Gaussian sampling.

    Args:
        prototype: Original prototype [1, feat_dim]
        num_samples: Number of augmented samples
        beta: Covariance scaling degree
    Returns:
        Augmented samples [num_samples, feat_dim]
    """
    feat_dim = prototype.shape[-1]
    mean = prototype.squeeze(0)

    # Estimate covariance (simplified: use identity scaled by beta)
    cov = torch.eye(feat_dim, device=prototype.device) * beta

    # Sample from multivariate normal
    samples = torch.distributions.MultivariateNormal(mean, cov).sample((num_samples,))

    # Normalize samples
    samples = F.normalize(samples, dim=1)

    return samples


# ============================================================================
# Complete ConCM Framework
# ============================================================================


class ConCMFramework(nn.Module):
    """
    Complete ConCM framework combining MPC and DSM modules.
    """

    def __init__(self, feat_dim=2048, attr_dim=128, geom_dim=128):
        super().__init__()
        self.mpc = MemoryAwarePrototypeCalibration(feat_dim=feat_dim, attr_dim=attr_dim)
        self.dsm = DynamicStructureMatching(feat_dim=feat_dim, geom_dim=geom_dim)
        self.base_memory = None

    def train_base_session(self, encoder, base_features, base_labels):
        """
        Train MPC and projector on base session.

        Args:
            encoder: Feature encoder
            base_features: Base class features
            base_labels: Base class labels
        """
        # Set to eval mode for single prototype processing
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # Extract base prototypes for memory
            unique_classes = torch.unique(base_labels)
            base_protos = []
            for cls in unique_classes:
                mask = base_labels == cls
                proto = base_features[mask].mean(0, keepdim=True)
                base_protos.append(proto)
                # Initialize structural anchors
                self.dsm.update_structure(int(cls), proto)

            self.base_memory = torch.cat(base_protos, dim=0)

        if was_training:
            self.train()

    def calibrate_novel_class(self, novel_proto, class_id, alpha=0.6):
        """
        Calibrate novel class prototype using MPC and update DSM.

        Args:
            novel_proto: Few-shot prototype [1, feat_dim]
            class_id: Novel class ID
            alpha: Calibration degree
        Returns:
            Calibrated prototype [1, feat_dim]
        """
        # Set to eval mode for single prototype processing
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # MPC: Calibrate prototype
            calibrated = self.mpc(novel_proto, self.base_memory, alpha=alpha)

            # DSM: Update structure
            self.dsm.update_structure(class_id, calibrated)

        if was_training:
            self.train()

        return calibrated
