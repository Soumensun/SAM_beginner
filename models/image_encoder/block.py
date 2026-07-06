import torch
import torch.nn as nn

from .attention import Attention
from .mlp import MLPBlock
from .window import window_partition, window_unpartition


class Block(nn.Module):

    def __init__(
        self,
        dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        window_size=14,
        input_size=(14, 14),
    ):
        super().__init__()

        self.window_size = window_size

        norm_layer=nn.LayerNorm

        # LayerNorm before Attention
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            use_rel_pos=True,
            input_size=(
                input_size if window_size == 0
                else (window_size, window_size)
            ),
)

        # LayerNorm before MLP
       

        self.norm2 = norm_layer(dim)

        # Feed Forward Network
        self.mlp = MLPBlock(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
        )

    def forward(self, x):

        # Input
        # (B,H,W,C)

        shortcut = x

        # LayerNorm
        # (B,H,W,C)
        x = self.norm1(x)

        # Save original spatial size
        H, W = x.shape[1], x.shape[2]

        # Window Partition (B,H,W,C) ->(B*num_windows,ws,ws,C)

        if self.window_size > 0:
            x, pad_hw = window_partition(
                x,
                self.window_size,
            )

            # Flatten each window (B*num_windows,ws*ws,C)
            B_, Wh, Ww, C = x.shape
            x = x.reshape(
                B_,
                Wh * Ww,
                C,
            )
        else:
            B_, H_, W_, C = x.shape
            x = x.reshape(
                B_,
                H_ * W_,
                C,
            )

        # Shape
        # (B*num_windows,196,768)


        # Window Attention
        #
        # (B*num_windows,196,768)
        #
        # ->
        #
        # (B*num_windows,196,768)


        x = self.attn(x)


        # Restore Window
        #
        # (B*num_windows,196,768)
        #
        # ->
        #
        # (B*num_windows,14,14,768)
  
        if self.window_size > 0:
            x = x.reshape(
                B_,
                Wh,
                Ww,
                C,
            )

            # Merge Windows
            #
            # (B*num_windows,14,14,C)
            #
            # ->
            #
            # (B,H,W,C)

            x = window_unpartition(
                x,
                self.window_size,
                pad_hw,
                (H, W),
            )
        else:
            x = x.reshape(B_, H, W, C)

        # First Residual Connection
        # (B,H,W,C)
        x = shortcut + x

        # Save Residual
        shortcut = x

        # LayerNorm
        # (B,H,W,C)
        x = self.norm2(x)

        # Feed Forward Network
        # (B,H,W,C)
        x = self.mlp(x)

        # Second Residual Connection
        # (B,H,W,C)
        x = shortcut + x

        return x