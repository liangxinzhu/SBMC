import math
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Small CNN for MNIST (0–7) with configurable prior stds for init."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 8,
        sigma_w: float | None = None,
        sigma_b: float | None = None,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

        # If sigmas are given, initialise like your MAP script
        if sigma_w is not None and sigma_b is not None:
            # weights: std = sigma_w; biases: std = sigma_b
            nn.init.normal_(self.conv.weight, mean=0.0, std=sigma_w)
            if self.conv.bias is not None:
                nn.init.normal_(self.conv.bias, mean=0.0, std=sigma_b)

            nn.init.normal_(self.fc.weight, mean=0.0, std=sigma_w)
            if self.fc.bias is not None:
                nn.init.normal_(self.fc.bias, mean=0.0, std=sigma_b)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
