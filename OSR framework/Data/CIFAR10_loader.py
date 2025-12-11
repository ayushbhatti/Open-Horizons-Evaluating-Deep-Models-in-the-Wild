import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# ----------------------------
# Safe relabeled subset class
# ----------------------------
class RelabeledSubset(Dataset):
    """
    Wraps a subset of a dataset and relabels class indices
    according to a given mapping (e.g., {0:0, 1:1, 3:2, ...}).
    This avoids using local lambda functions that can't be pickled.
    """
    def __init__(self, dataset, indices, class_map):
        self.dataset = dataset
        self.indices = indices
        self.class_map = class_map

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        return x, self.class_map[y]

    def __len__(self):
        return len(self.indices)


# ----------------------------
# CIFAR-10 Loader for OSR setup
# ----------------------------
def get_cifar10_loaders(
    known_classes: int = 6,
    batch_size: int = 256,
    num_workers: int = 4,
    backbone: str = "resnet50",
):
    """
    Load CIFAR-10 dataset and simulate an open-set split:
    - First 'known_classes' classes are treated as known.
    - Remaining are treated as unknown.

    The preprocessing pipeline is adapted based on the backbone:
    - ResNet/ConvNeXt: standard CIFAR-10 normalization at 32x32.
    - CLIP ViT: CLIP-style resize to 224x224 and CLIP mean/std.
    """
    assert 1 <= known_classes <= 9, "Keep at least one unknown class."

    backbone = backbone.lower()

    # ----------------------------
    # Transforms based on backbone
    # ----------------------------
    if "clip" in backbone:
        # CLIP ViT-B/16 style preprocessing (224x224 + CLIP mean/std)
        img_size = 224
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        T_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        T_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Standard CIFAR-10 preprocessing (32x32), good for ResNet / ConvNeXt
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )

        T_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        T_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # ----------------------------
    # Load CIFAR-10 from torchvision
    # ----------------------------
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=T_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=T_test
    )

    # Class split
    known = list(range(known_classes))
    unknown = list(range(known_classes, 10))

    # Indices for known and unknown
    tr_idx = [i for i, (_, y) in enumerate(trainset) if y in known]
    te_id_idx = [i for i, (_, y) in enumerate(testset) if y in known]
    te_ood_idx = [i for i, (_, y) in enumerate(testset) if y in unknown]

    # Relabel known classes to 0..K-1
    class_map = {c: i for i, c in enumerate(known)}

    # Create datasets
    train_known = RelabeledSubset(trainset, tr_idx, class_map)
    test_known = RelabeledSubset(testset, te_id_idx, class_map)
    test_ood = Subset(testset, te_ood_idx)  # labels unused for OOD

    # DataLoaders
    tr_loader = DataLoader(
        train_known,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    te_id_loader = DataLoader(
        test_known,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    te_ood_loader = DataLoader(
        test_ood,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return tr_loader, te_id_loader, te_ood_loader, known, unknown


# ----------------------------
# Optional quick test
# ----------------------------
if __name__ == "__main__":
    tr, te_id, te_ood, known, unknown = get_cifar10_loaders(backbone="clip_vit_b16")
    print(f"Train batches: {len(tr)}, Test (ID): {len(te_id)}, Test (OOD): {len(te_ood)}")
    print(f"Known classes: {known}, Unknown classes: {unknown}")