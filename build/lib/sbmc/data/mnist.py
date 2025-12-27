from typing import Sequence, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

from .base import DatasetBundle


@dataclass
class MNISTDataConfig:
    # Where to store/download MNIST
    root: str = "./data"

    # Which labels are considered in-distribution
    allowed_labels: Sequence[int] = tuple(range(8))  # 0–7

    # Train/val split (in terms of *filtered* train points)
    n_total_train: int = 1200   # total (train+val)
    n_train: int = 1000         # MAP/DE train
    n_val: int = 200            # MAP/DE val

    # ID test set size
    n_test_id: int = 1000       # first 1000 filtered test points (like PSMC)

    # Batch size for MAP/DE loaders
    batch_size: int = 64

    # Device for full-batch tensors (SMC/HMC)
    device: Optional[torch.device] = None


class FilteredDataset(Dataset):
    """Wrap a dataset and keep only samples whose label is in allowed_labels."""

    def __init__(self, base_dataset: Dataset, allowed_labels: Sequence[int]):
        self.base = base_dataset
        self.allowed = set(int(a) for a in allowed_labels)
        self.indices = [
            i for i, (_, y) in enumerate(self.base)
            if int(y) in self.allowed
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, y = self.base[base_idx]
        return x, y


def build_mnist_dataset(
    config: MNISTDataConfig = MNISTDataConfig(),
) -> DatasetBundle:
    """Build a DatasetBundle for MNIST with filtered labels and fixed splits."""
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    full_train_dataset = torchvision.datasets.MNIST(
        root=config.root, train=True, download=True, transform=transform
    )
    full_test_dataset = torchvision.datasets.MNIST(
        root=config.root, train=False, download=True, transform=transform
    )

    # Filter labels
    filtered_train_pool = FilteredDataset(full_train_dataset, config.allowed_labels)
    filtered_test_pool = FilteredDataset(full_test_dataset, config.allowed_labels)

    # Train/val split
    assert config.n_train + config.n_val <= config.n_total_train
    filtered_train_total = Subset(filtered_train_pool, list(range(config.n_total_train)))
    map_train_dataset = Subset(filtered_train_total, list(range(config.n_train)))
    map_val_dataset = Subset(
        filtered_train_total,
        list(range(config.n_train, config.n_train + config.n_val)),
    )

    train_loader = DataLoader(
        map_train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        map_val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Full-batch train for SMC/HMC: use all (train+val) 1200 points
    train_loader_full = DataLoader(
        filtered_train_total,
        batch_size=len(filtered_train_total),
        shuffle=False,
    )
    x_train_full, y_train_full = next(iter(train_loader_full))
    x_train_full = x_train_full.to(device)
    y_train_full = y_train_full.to(device)

    # ID test subset
    test_id_dataset = Subset(filtered_test_pool, list(range(config.n_test_id)))
    test_loader_full = DataLoader(
        test_id_dataset,
        batch_size=len(test_id_dataset),
        shuffle=False,
    )
    x_test_full, y_test_full = next(iter(test_loader_full))
    x_test_full = x_test_full.to(device)
    y_test_full = y_test_full.to(device)

    input_shape: Tuple[int, ...] = tuple(x_train_full.shape[1:])
    num_classes = len(set(int(y) for y in config.allowed_labels))

    return DatasetBundle(
        name="mnist",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        x_train_full=x_train_full,
        y_train_full=y_train_full,
        x_test_full=x_test_full,
        y_test_full=y_test_full,
        input_shape=input_shape,
        num_classes=num_classes,
    )
