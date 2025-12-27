# sbmc/models/simple_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Unified classifier for any embedding dataset (IMDb, CIFAR10, AG-News, etc.).

    - When hidden_dim == 0 or None:
        Logistic regression: parameters:
            fc.weight, fc.bias

    - When hidden_dim > 0:
        1-hidden-layer MLP:
            fc1.weight, fc1.bias
            fc2.weight, fc2.bias

    This matches the structure required by your MAP/DE/PSMC/PHMC algorithms.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        sigma_w: float = 0.0,
        sigma_b: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        if hidden_dim is None or hidden_dim == 0:
            # Logistic regression
            self.fc = nn.Linear(input_dim, num_classes)

            if sigma_w > 0:
                nn.init.normal_(self.fc.weight, mean=0.0, std=sigma_w)
            if sigma_b > 0 and self.fc.bias is not None:
                nn.init.normal_(self.fc.bias, mean=0.0, std=sigma_b)

        else:
            # MLP
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

            if sigma_w > 0:
                nn.init.normal_(self.fc1.weight, mean=0.0, std=sigma_w)
                nn.init.normal_(self.fc2.weight, mean=0.0, std=sigma_w)

            if sigma_b > 0:
                if self.fc1.bias is not None:
                    nn.init.normal_(self.fc1.bias, mean=0.0, std=sigma_b)
                if self.fc2.bias is not None:
                    nn.init.normal_(self.fc2.bias, mean=0.0, std=sigma_b)

    def forward(self, x):
        if hasattr(self, "fc"):
            return self.fc(x)
        else:
            x = F.relu(self.fc1(x))
            return self.fc2(x)
