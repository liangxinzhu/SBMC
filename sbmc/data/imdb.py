from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
import os
import warnings

import warnings
import torchtext
torchtext.disable_torchtext_deprecation_warning()  # 👈 silences that big banner

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchtext.datasets import IMDB
from sentence_transformers import SentenceTransformer

from .base import DatasetBundle


@dataclass
class IMDBDataConfig:
    # Where to store/download MNIST
    root: str = "./sbmc/data"

    # SBERT model
    model_name: str = "all-mpnet-base-v2"

    # Cache paths for embeddings
    train_cache_path: str = "imdb_embeddings_trainBig.pt"
    test_cache_path: str = "imdb_embeddings_testBig.pt"

    # MAP/DE splits
    map_train_size: int = 20000
    map_val_size: int = 5000

    # Batch size for MAP/DE loaders
    batch_size: int = 128

    # Device for full-batch tensors (SMC/HMC)
    device: Optional[torch.device] = None


def _label_to_int(label: Union[int, str]) -> int:
    """Map torchtext IMDB labels to {0,1}, robust to different formats."""
    if isinstance(label, int):
        # Sometimes 0/1, sometimes 1/2
        if label in (0, 1):
            return int(label)
        if label in (1, 2):
            return {1: 0, 2: 1}[label]
    if isinstance(label, str):
        l = label.lower()
        if "neg" in l:
            return 0
        if "pos" in l:
            return 1
    raise ValueError(f"Unexpected IMDB label format: {label!r}")


def create_sbert_embedded_imdb_dataset(config: IMDBDataConfig):
    """
    Load IMDB and embed each review using SBERT (with caching).
    """
    os.makedirs(config.root, exist_ok=True)

    train_cache_path = os.path.join(config.root, config.train_cache_path)
    test_cache_path  = os.path.join(config.root, config.test_cache_path)

    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading cached SBERT embeddings for IMDB from {config.root}...")
        X_train, y_train = torch.load(train_cache_path)
        X_test, y_test = torch.load(test_cache_path)
        return X_train, y_train, X_test, y_test

    print("Cached IMDB embeddings not found. Computing SBERT embeddings...")

    sbert = SentenceTransformer(config.model_name)
    sbert.eval()

    train_data = list(IMDB(root=config.root, split="train"))
    test_data  = list(IMDB(root=config.root, split="test"))

    X_train_list: List[List[float]] = []
    y_train_list: List[int] = []
    for (label, text) in train_data:
        label_int = _label_to_int(label)
        emb = sbert.encode(text, convert_to_numpy=True)
        X_train_list.append(emb)
        y_train_list.append(label_int)

    X_test_list: List[List[float]] = []
    y_test_list: List[int] = []
    for (label, text) in test_data:
        label_int = _label_to_int(label)
        emb = sbert.encode(text, convert_to_numpy=True)
        X_test_list.append(emb)
        y_test_list.append(label_int)

    X_train = torch.tensor(X_train_list, dtype=torch.float32)
    y_train = torch.tensor(y_train_list, dtype=torch.long)
    X_test = torch.tensor(X_test_list, dtype=torch.float32)
    y_test = torch.tensor(y_test_list, dtype=torch.long)

    torch.save((X_train, y_train), train_cache_path)
    torch.save((X_test, y_test), test_cache_path)
    print(f"SBERT IMDB embeddings computed and saved to {config.root}.")

    return X_train, y_train, X_test, y_test


def build_imdb_dataset(
    config: IMDBDataConfig = IMDBDataConfig(),
) -> DatasetBundle:
    """
    Build a DatasetBundle for IMDB using SBERT embeddings, matching IMDB_psmc.py.
    """
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test = create_sbert_embedded_imdb_dataset(config)

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

    input_shape: Tuple[int, ...] = (X_train.shape[1],)  # e.g. (768,)
    num_classes = int(y_train.max().item() + 1)  # assumes labels are {0,1}

    return DatasetBundle(
        name="imdb",
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
