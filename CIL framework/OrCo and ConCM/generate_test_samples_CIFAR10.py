#!/usr/bin/env python3
"""
Generate fixed test samples for consistent evaluation across experiments.
Creates a CSV file with 1000 samples per class from CIFAR-10 test set.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from collections import defaultdict

# Configuration
SEED = 42
SAMPLES_PER_CLASS = 1000
OUTPUT_FILE = "fixed_test_samples.csv"
DATA_DIR = "../../data"


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_fixed_test_samples():
    """Generate and save fixed test sample indices"""
    set_seed()

    print("=" * 70)
    print("GENERATING FIXED TEST SAMPLES")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    # Load CIFAR-10 test set (transform doesn't matter, we only need labels)
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(
        DATA_DIR, train=False, download=True, transform=transform
    )

    print(f"✓ Loaded CIFAR-10 test set: {len(testset)} samples")

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(testset):
        class_indices[int(label)].append(idx)

    # Verify we have enough samples per class
    print("\nSamples available per class:")
    for cls in sorted(class_indices.keys()):
        print(f"  Class {cls}: {len(class_indices[cls])} samples")

    # Check if we can get 1000 samples per class
    min_samples = min(len(indices) for indices in class_indices.values())
    if min_samples < SAMPLES_PER_CLASS:
        print(f"\n⚠️  Warning: Only {min_samples} samples available per class")
        print(f"   Adjusting to {min_samples} samples per class")
        samples_to_select = min_samples
    else:
        samples_to_select = SAMPLES_PER_CLASS

    # Sample fixed indices for each class
    selected_samples = []
    for cls in sorted(class_indices.keys()):
        # Shuffle and select
        indices = class_indices[cls].copy()
        random.shuffle(indices)
        selected = indices[:samples_to_select]

        for idx in selected:
            selected_samples.append({"class_id": cls, "sample_idx": idx})

    # Save to CSV using pandas
    print(f"\n💾 Saving {len(selected_samples)} samples to {OUTPUT_FILE}...")
    df = pd.DataFrame(selected_samples)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved successfully!")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    class_counts = df.groupby("class_id").size()

    print(f"Total samples: {len(df)}")
    print(f"Samples per class:")
    for cls in sorted(class_counts.index):
        print(f"  Class {cls}: {class_counts[cls]} samples")

    print("\n✅ Generation complete!")
    print(f"\nTo use these samples in test.py, add the flag:")
    print(f"  --fixed_samples {OUTPUT_FILE}")
    print("=" * 70)

    return OUTPUT_FILE


if __name__ == "__main__":
    generate_fixed_test_samples()
