from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader


@dataclass
class DatasetBundle:
    """Container for dataset-related objects.

    This is the API boundary between dataset-specific code and
    generic inference methods.
    """

    name: str
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader] = None

    # Full-batch tensors (for methods like SMC / HMC that want all data at once)
    x_train_full: Optional[torch.Tensor] = None
    y_train_full: Optional[torch.Tensor] = None
    x_test_full: Optional[torch.Tensor] = None
    y_test_full: Optional[torch.Tensor] = None

    # Meta information, useful for building models
    input_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
