import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

def get_cifar10_loaders(known_classes=6, batch_size=256, num_workers=4):
    normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                                     std=[0.2470,0.2435,0.2616])
    T_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(), normalize])
    T_test = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=T_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=T_test)

    known = list(range(known_classes))
    unknown = list(range(known_classes, 10))

    tr_idx = [i for i,(_,y) in enumerate(trainset) if y in known]
    te_id_idx = [i for i,(_,y) in enumerate(testset) if y in known]
    te_ood_idx = [i for i,(_,y) in enumerate(testset) if y in unknown]

    class_map = {c:i for i,c in enumerate(known)}

    def relabel(dataset, idxs):
        ds = Subset(dataset, idxs)
        old_get = ds.dataset.__getitem__
        def _get(i):
            x,y = old_get(ds.indices[i])
            return x, class_map[y]
        ds.dataset.__getitem__ = lambda i: _get(i)
        return ds

    train_known = relabel(trainset, tr_idx)
    test_known  = relabel(testset, te_id_idx)
    test_ood    = Subset(testset, te_ood_idx)

    tr_loader = DataLoader(train_known, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    te_id_loader = DataLoader(test_known, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_ood_loader = DataLoader(test_ood, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr_loader, te_id_loader, te_ood_loader, known, unknown