# sbmc/methods/de.py

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..data.base import DatasetBundle

try:
    import pyro
except ImportError:  # pragma: no cover
    pyro = None


ModelBuilder = Callable[[], nn.Module]


@dataclass
class DEConfig:
    # Ensemble settings
    ensemble_size: int = 10  # match your ensemble_size = 10

    # Optimizer
    lr: float = 1e-3

    # Training schedule
    max_epochs: int = 1000
    moving_avg_window: int = 10
    patience: int = 5

    # Regularization hyperparameters (same as MAP)
    sigma_w: float = math.sqrt(0.1)
    sigma_b: float = math.sqrt(0.1)

    # Optional override for batch size (if you want different from dataset config)
    batch_size_override: Optional[int] = None

    # Device and base seed
    device: Optional[torch.device] = None
    base_seed: int = 0  # we'll derive member seeds from this


@dataclass
class MemberTrainingStats:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    epochs_trained: int = 0
    train_time_sec: float = 0.0
    seed: Optional[int] = None


@dataclass
class DEResult:
    members: List[nn.Module]
    stats: List[MemberTrainingStats]


def _set_member_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if pyro is not None:
        pyro.set_rng_seed(seed)


def _maybe_override_loader(
    loader: DataLoader,
    batch_size_override: Optional[int],
) -> DataLoader:
    if batch_size_override is None:
        return loader
    return DataLoader(
        loader.dataset,
        batch_size=batch_size_override,
        shuffle=True,
    )


def _compute_reg_loss(
    net: nn.Module,
    sigma_w: float,
    sigma_b: float,
) -> torch.Tensor:
    """
    Same regularization as in MAP and original DE code:
      reg = sum(param^2 / (2*sigma^2)) with layer-dependent sigma.
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
            sigma = sigma_w
        reg = reg + (p ** 2).sum() / (2.0 * sigma ** 2)
    return reg


def train_de(
    model_fn: ModelBuilder,
    dataset: DatasetBundle,
    config: DEConfig,
) -> DEResult:
    """
    Deep Ensemble training matching the logic of MNIST_DE.py:

      - Adam lr=1e-3
      - Same Gaussian prior regularization as MAP
      - Moving-average early stopping (window=10, patience=5)
      - Up to 1000 epochs per member
      - ensemble_size=10 by default
    """
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert dataset.train_loader is not None, "DE requires train_loader."
    assert dataset.val_loader is not None, "DE requires val_loader."
    train_loader = _maybe_override_loader(dataset.train_loader, config.batch_size_override)
    val_loader = _maybe_override_loader(dataset.val_loader, config.batch_size_override)

    N_train = len(train_loader.dataset)

    members: List[nn.Module] = []
    stats_list: List[MemberTrainingStats] = []

    for m in range(config.ensemble_size):
        # Member-specific seed, analogous to member_seed = r*1000 + m
        member_seed = config.base_seed * 1000 + m
        _set_member_seed(member_seed)

        net = model_fn().to(device)
        optimizer = optim.Adam(net.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_moving_avg = float("inf")
        no_improve_count = 0

        start_time = time.time()
        stopped_epoch = 0

        for epoch in range(config.max_epochs):
            # ---- Training ----
            net.train()
            running_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = net(x_batch)
                ce_loss = criterion(logits, y_batch)
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
            train_losses.append(train_loss)

            # ---- Validation ----
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
            val_losses.append(val_loss)

            print(
                f"[DE] Member {m} (seed={member_seed}) "
                f"Epoch {epoch+1}: Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}"
            )

            # Moving-average early stopping
            if epoch >= config.moving_avg_window - 1:
                window_losses = val_losses[-config.moving_avg_window:]
                moving_avg = sum(window_losses) / config.moving_avg_window
                if moving_avg < best_moving_avg:
                    best_moving_avg = moving_avg
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= config.patience:
                    print(
                        f"[DE] Member {m} (seed={member_seed}) "
                        f"Early stopping at epoch {epoch+1}"
                    )
                    stopped_epoch = epoch + 1
                    break

            stopped_epoch = epoch + 1

        total_time = time.time() - start_time
        member_stats = MemberTrainingStats(
            train_losses=train_losses,
            val_losses=val_losses,
            epochs_trained=stopped_epoch,
            train_time_sec=total_time,
            seed=member_seed,
        )
        print(
            f"[DE] Member {m} (seed={member_seed}) "
            f"trained in {stopped_epoch} epochs, time={total_time:.2f}s"
        )

        members.append(net)
        stats_list.append(member_stats)

    return DEResult(members=members, stats=stats_list)
