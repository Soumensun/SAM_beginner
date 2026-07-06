import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        image_size=1024,
        in_channels=3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):  # (B, 3, 1024, 1024)
        x = self.proj(x)  # (B, 768, 64, 64)
        x = x.permute(0, 2, 3, 1)  # (B, 64, 64, 768)
        return x
