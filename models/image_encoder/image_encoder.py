import torch
import torch.nn as nn

from .patch_embedding import PatchEmbedding
from .block import Block
from .layernorm2d import LayerNorm2d


class ImageEncoder(nn.Module):

    def __init__(
        self,
        image_size=1024,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_channels=256,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
    ):
        super().__init__()

        # Number of patches
        self.grid_size = image_size // patch_size

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Absolute Position Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.grid_size,
                self.grid_size,
                embed_dim,
            )
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList()

        for i in range(depth):

            self.blocks.append(

                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_size=0 if i in global_attn_indexes else window_size,
                    input_size=(self.grid_size, self.grid_size),
                )

            )

        # Neck
        self.neck = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            LayerNorm2d(
                out_channels,
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            LayerNorm2d(
                out_channels,
            ),
        )

    def forward(self, x):

        # Patch Embedding (B,3,1024,1024) -> (B,64,64,768)
        x = self.patch_embed(x)

        # Add Absolute Position Embedding (B,64,64,768) -> (B,64,64,768)
        x = x + self.pos_embed

        # Transformer Blocks (B,64,64,768) -> (B,64,64,768)
        for block in self.blocks:
            x = block(x)

        # BHWC -> BCHW (B,64,64,768) -> (B,768,64,64)
        x = x.permute(
            0,
            3,
            1,
            2,
        )

        # Neck (B,768,64,64) -> (B,256,64,64)
        x = self.neck(x)

        return x