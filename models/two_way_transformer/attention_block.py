import torch
import torch.nn as nn

from .attention import Attention
from .mlp import MLPBlock


class TwoWayAttentionBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_heads=8,
        mlp_dim=2048,
        activation=nn.ReLU,
        attention_downsample_rate=2,
        skip_first_layer_pe=False,
    ):
        super().__init__()

        self.skip_first_layer_pe = skip_first_layer_pe

        # Self Attention
        self.self_attn = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        # Query -> Image
        self.cross_attn_token_to_image = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
        )

        # MLP
        self.mlp = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            activation=activation,
        )

        # Image -> Query
        self.cross_attn_image_to_token = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        queries,
        keys,
        query_pe,
        key_pe,
    ):
        """
        queries  : Prompt Tokens (B,Nq,256)

        keys     : Image Tokens (B,Nk,256)

        query_pe : Prompt Position Encoding (B,Nq,256)

        key_pe   : Image Position Encoding (B,Nk,256)
        """

        # --------------------------------------------------------
        # 1. Self Attention on Prompt Tokens
        # --------------------------------------------------------

        if self.skip_first_layer_pe:

            # Self Attention (B,Nq,256) -> (B,Nq,256)
            queries = self.self_attn(
                q=queries,
                k=queries,
                v=queries,
            )

        else:

            # Add Prompt Position Encoding (B,Nq,256) -> (B,Nq,256)
            q = queries + query_pe

            # Self Attention (B,Nq,256) -> (B,Nq,256)
            attn_out = self.self_attn(
                q=q,
                k=q,
                v=queries,
            )

            # Residual Connection (B,Nq,256) -> (B,Nq,256)
            queries = queries + attn_out

        # LayerNorm (B,Nq,256) -> (B,Nq,256)
        queries = self.norm1(queries)

        # --------------------------------------------------------
        # 2. Prompt → Image Cross Attention
        # --------------------------------------------------------

        # Add Prompt Position Encoding (B,Nq,256) -> (B,Nq,256)
        q = queries + query_pe

        # Add Image Position Encoding (B,Nk,256) -> (B,Nk,256)
        k = keys + key_pe

        # Cross Attention (B,Nq,256) -> (B,Nq,256)
        attn_out = self.cross_attn_token_to_image(
            q=q,
            k=k,
            v=keys,
        )

        # Residual Connection (B,Nq,256) -> (B,Nq,256)
        queries = queries + attn_out

        # LayerNorm (B,Nq,256) -> (B,Nq,256)
        queries = self.norm2(queries)

        # --------------------------------------------------------
        # 3. MLP Block
        # --------------------------------------------------------

        # MLP (B,Nq,256) -> (B,Nq,256)
        mlp_out = self.mlp(queries)

        # Residual Connection (B,Nq,256) -> (B,Nq,256)
        queries = queries + mlp_out

        # LayerNorm (B,Nq,256) -> (B,Nq,256)
        queries = self.norm3(queries)

        # --------------------------------------------------------
        # 4. Image → Prompt Cross Attention
        # --------------------------------------------------------

        # Add Prompt Position Encoding (B,Nq,256) -> (B,Nq,256)
        q = queries + query_pe

        # Add Image Position Encoding (B,Nk,256) -> (B,Nk,256)
        k = keys + key_pe

        # Reverse Cross Attention (B,Nk,256) -> (B,Nk,256)
        attn_out = self.cross_attn_image_to_token(
            q=k,
            k=q,
            v=queries,
        )

        # Residual Connection (B,Nk,256) -> (B,Nk,256)
        keys = keys + attn_out

        # LayerNorm (B,Nk,256) -> (B,Nk,256)
        keys = self.norm4(keys)

        return queries, keys
        
        