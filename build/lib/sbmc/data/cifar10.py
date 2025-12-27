from typing import Optional, Tuple, List
from dataclasses import dataclass
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from .base import DatasetBundle


@dataclass
class CIFAR10DataConfig:
    # CIFAR-10 root
    root: str = "./sbmc/data"

    # In-distribution labels (0–9)
    allowed_labels: List[int] = None

    # Cache paths for ResNet-50 embeddings
    train_cache_path: str = "cifar10_train_embeddings.pt"
    test_cache_path: str = "cifar10_test_embeddings.pt"

    # MAP/DE splits (on embedded train data)
    map_train_size: int = 40000
    map_val_size: int = 10000

    # Batch size for MAP/DE loaders
    batch_size: int = 128

    # Batch size used when *embedding* images with ResNet-50
    embed_batch_size: int = 128

    # Device for embedding and full-batch tensors (SMC/HMC)
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.allowed_labels is None:
            self.allowed_labels = list(range(10))


class FilteredCIFAR10(Dataset):
    """
    CIFAR-10 wrapper that keeps only samples whose label is in allowed_labels.
    Mirrors the FilteredCIFAR10 in CIFAR_psmc.py.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        transform,
        download: bool,
        allowed_labels: List[int],
    ):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )
        self.allowed_labels = set(int(a) for a in allowed_labels)
        self.indices = [
            i for i, (_, label) in enumerate(self.dataset)
            if int(label) in self.allowed_labels
        ]

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        return self.dataset[base_idx]

    def __len__(self):
        return len(self.indices)


def create_resnet50_embedded_cifar10_dataset(
    config: CIFAR10DataConfig,
):
    """
    Load CIFAR-10 and embed each image using a pretrained ResNet-50,
    as in CIFAR_psmc.py. Embeddings are cached to disk.
    """
    os.makedirs(config.root, exist_ok=True)

    train_cache_path = os.path.join(config.root, config.train_cache_path)
    test_cache_path  = os.path.join(config.root, config.test_cache_path)
    if (
        os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading cached ResNet-50 embeddings for CIFAR-10...")
        X_train, y_train = torch.load(train_cache_path)
        X_test, y_test = torch.load(test_cache_path)
        return X_train, y_train, X_test, y_test

    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cached embeddings not found. Computing ResNet-50 embeddings for CIFAR-10...")

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = FilteredCIFAR10(
        root=config.root,
        train=True,
        transform=transform,
        download=True,
        allowed_labels=config.allowed_labels,
    )
    test_dataset = FilteredCIFAR10(
        root=config.root,
        train=False,
        transform=transform,
        download=True,
        allowed_labels=config.allowed_labels,
    )

    train_dataset = Subset(train_dataset, list(range(len(train_dataset))))
    test_dataset = Subset(test_dataset, list(range(len(test_dataset))))

    train_loader = DataLoader(
        train_dataset, batch_size=config.embed_batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.embed_batch_size, shuffle=False, num_workers=2
    )

    resnet50 = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # drop final fc
    feature_extractor.eval()
    feature_extractor.to(device)

    X_train_list, y_train_list = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)           # (B, 2048, 1, 1)
            features = features.view(features.size(0), -1) # (B, 2048)
            X_train_list.append(features.cpu())
            y_train_list.append(targets)
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)

    X_test_list, y_test_list = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            features = features.view(features.size(0), -1)
            X_test_list.append(features.cpu())
            y_test_list.append(targets)
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)

    torch.save((X_train, y_train), config.train_cache_path)
    torch.save((X_test, y_test), config.test_cache_path)
    print("ResNet-50 embeddings computed and saved.")

    return X_train, y_train, X_test, y_test


def build_cifar10_dataset(
    config: CIFAR10DataConfig = CIFAR10DataConfig(),
) -> DatasetBundle:
    """
    Build a DatasetBundle for CIFAR-10 using ResNet-50 embeddings.
    """
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test = create_resnet50_embedded_cifar10_dataset(config)

    N_train = X_train.shape[0]
    if config.map_train_size + config.map_val_size > N_train:
        raise ValueError(
            f"map_train_size + map_val_size = {config.map_train_size + config.map_val_size} "
            f"exceeds number of train samples {N_train}."
        )

    # MAP/DE splits
    x_map_train = X_train[:config.map_train_size]
    y_map_train = y_train[:config.map_train_size]
    x_map_val = X_train[config.map_train_size:config.map_train_size + config.map_val_size]
    y_map_val = y_train[config.map_train_size:config.map_train_size + config.map_val_size]

    train_ds = TensorDataset(x_map_train, y_map_train)
    val_ds = TensorDataset(x_map_val, y_map_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # Full-batch tensors for SMC/HMC
    x_train_full = X_train.to(device)
    y_train_full = y_train.to(device)
    x_test_full = X_test.to(device)
    y_test_full = y_test.to(device)

    input_shape: Tuple[int, ...] = (X_train.shape[1],)  # 2048
    num_classes = len(set(int(l) for l in config.allowed_labels))

    return DatasetBundle(
        name="cifar10",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        x_train_full=x_train_full,
        y_train_full=y_train_full,
        x_test_full=x_test_full,
        y_test_full=y_test_full,
        input_shape=input_shape,
        num_classes=num_classes,
    )
