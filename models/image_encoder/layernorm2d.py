import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D feature maps.

    Input Shape:
        (B, C, H, W)

    Output Shape:
        (B, C, H, W)
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):

        # Input (B,C,H,W)

        # Mean over channel dimension (B,C,H,W) -> (B,1,H,W)
        mean = x.mean(dim=1, keepdim=True)

        # Variance over channel dimension (B,C,H,W) -> (B,1,H,W)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)

        # Normalize (B,C,H,W) -> (B,C,H,W)
        x = (x - mean) / torch.sqrt(variance + self.eps)

        # Scale and Shift (B,C,H,W) -> (B,C,H,W)
        x = (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )

        return x