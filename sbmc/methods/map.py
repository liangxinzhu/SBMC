# sbmc/methods/map.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..data.base import DatasetBundle


@dataclass
class MAPConfig:
    # Optimizer
    lr: float = 1e-3

    # Max epochs (your script uses "for epoch in range(1000)")
    max_epochs: int = 1000

    # Early stopping (moving average over val loss)
    moving_avg_window: int = 10
    patience: int = 5

    # Regularization hyperparameters (match v=0.1, sdb = sqrt(v))
    sigma_w: float = math.sqrt(0.1)    # std for conv and fc weights
    sigma_b: float = math.sqrt(0.1)     # std for biases

    # Device and seed
    device: Optional[torch.device] = None
    seed: Optional[int] = None


@dataclass
class MAPResult:
    net: nn.Module
    history: Dict[str, List[float]]
    stopped_epoch: int


def _set_global_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    import numpy as np
    import pyro
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


def _compute_reg_loss(
    net: nn.Module,
    sigma_w: float,
    sigma_b: float,
) -> torch.Tensor:
    """
    Match your MAP/DE reg:
      - conv and fc weights   -> sigma_w
      - biases       -> sigma_b
    reg = sum( param^2 / (2*sigma^2) ) over all params (prior mean 0).
    """
    reg = torch.tensor(0.0, device=next(net.parameters()).device)
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("bias"):
            sigma = sigma_b
        elif "conv" in name:
            sigma = sigma_w
        elif "fc" in name:
            sigma = sigma_w
        else:
            # default to fc-style for any other weight
            sigma = sigma_w
        reg = reg + (p ** 2).sum() / (2.0 * sigma ** 2)
    return reg


def train_map(
    net: nn.Module,
    dataset: DatasetBundle,
    config: MAPConfig,
) -> MAPResult:
    """
    MAP training matching your MNIST_MAP.py behavior:

      loss = CE + reg_loss / N_train
      reg_loss = sum(param^2 / (2 * sigma^2)) with layer-specific sigmas
      early stopping on moving average of val loss.
    """
    _set_global_seed(config.seed)

    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    assert dataset.train_loader is not None, "MAP requires a train_loader."
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader  # can be None, but you use it in MNIST

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)

    N_train = len(train_loader.dataset)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    best_moving_avg = float("inf")
    no_improve_count = 0
    stopped_epoch = 0

    for epoch in range(config.max_epochs):
        # ---- Training ----
        net.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            ce_loss = criterion(logits, y)

            reg_loss = _compute_reg_loss(
                net,
                sigma_w=config.sigma_w,
                sigma_b=config.sigma_b,
            )

            loss = ce_loss + reg_loss / float(N_train)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # ---- Validation ----
        if val_loader is not None:
            net.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits_val = net(x_val)
                    ce_loss_val = criterion(logits_val, y_val)
                    reg_loss_val = _compute_reg_loss(
                        net,
                        sigma_w=config.sigma_w,
                        sigma_b=config.sigma_b,
                    )
                    val_loss_batch = ce_loss_val + reg_loss_val / float(N_train)
                    val_running_loss += val_loss_batch.item()
            val_loss = val_running_loss / len(val_loader)
            history["val_loss"].append(val_loss)

            print(
                f"[MAP] Epoch {epoch+1}: "
                f"Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}"
            )

            # Moving-average early stopping (same as your code)
            if epoch >= config.moving_avg_window - 1:
                window_losses = history["val_loss"][-config.moving_avg_window:]
                moving_avg = sum(window_losses) / config.moving_avg_window
                if moving_avg < best_moving_avg:
                    best_moving_avg = moving_avg
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= config.patience:
                    print(f"[MAP] Early stopping at epoch {epoch+1}")
                    stopped_epoch = epoch + 1
                    break
        else:
            # no val_loader -> just train for max_epochs
            print(f"[MAP] Epoch {epoch+1}: Train Loss = {train_loss:.8f}")

        stopped_epoch = epoch + 1

    return MAPResult(net=net, history=history, stopped_epoch=stopped_epoch)
