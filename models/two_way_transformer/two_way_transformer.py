

import torch
import torch.nn as nn

from .attention import Attention
from .attention_block import TwoWayAttentionBlock


class TwoWayTransformer(nn.Module):

    def __init__(
        self,
        depth=2,
        embedding_dim=256,
        num_heads=8,
        mlp_dim=2048,
        attention_downsample_rate=2,
        activation=nn.ReLU,
    ):
        super().__init__()

        self.depth = depth
        self.embedding_dim = embedding_dim

        # Two-Way Attention Blocks
        self.layers = nn.ModuleList()

        for i in range(depth):

            self.layers.append(

                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )

            )

        # Final Prompt → Image Attention
        self.final_attn_token_to_image = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
        )

        # Final LayerNorm
        self.norm_final_attn = nn.LayerNorm(
            embedding_dim,
        )

    def forward(
        self,
        image_embedding,
        image_pe,
        point_embedding,
    ):
        """
        image_embedding : (B,256,H,W)

        image_pe : (B,256,H,W)

        point_embedding : (B,N,256)
        """

        # ----------------------------------------------------
        # Flatten Image Embedding
        # (B,256,H,W) -> (B,H*W,256)
        # ----------------------------------------------------

        B, C, H, W = image_embedding.shape

        image_embedding = image_embedding.flatten(2).permute(
            0,
            2,
            1,
        )

        # (B,256,H,W) -> (B,H*W,256)
        image_pe = image_pe.flatten(2).permute(
            0,
            2,
            1,
        )

        # ----------------------------------------------------
        # Initialize Queries and Keys
        # ----------------------------------------------------

        # Prompt Tokens (B,N,256)
        queries = point_embedding

        # Image Tokens (B,H*W,256)
        keys = image_embedding

        # ----------------------------------------------------
        # Two-Way Attention Blocks
        # ----------------------------------------------------

        for layer in self.layers:

            queries, keys = layer(

                queries=queries,

                keys=keys,

                query_pe=point_embedding,

                key_pe=image_pe,

            )

        # ----------------------------------------------------
        # Final Prompt → Image Attention
        # ----------------------------------------------------

        # Add Prompt Position Encoding (B,N,256) -> (B,N,256)
        q = queries + point_embedding

        # Add Image Position Encoding (B,H*W,256) -> (B,H*W,256)
        k = keys + image_pe

        # Cross Attention (B,N,256) -> (B,N,256)
        attn_out = self.final_attn_token_to_image(
            q=q,
            k=k,
            v=keys,
        )

        # Residual Connection (B,N,256) -> (B,N,256)
        queries = queries + attn_out

        # LayerNorm (B,N,256) -> (B,N,256)
        queries = self.norm_final_attn(
            queries,
        )

        return queries, keys