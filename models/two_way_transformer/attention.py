import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_heads=8,
        downsample_rate=1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Internal Dimension
        self.internal_dim = embedding_dim // downsample_rate

        assert (
            self.internal_dim % num_heads == 0
        ), "internal_dim must be divisible by num_heads."

        self.head_dim = self.internal_dim // num_heads

        self.scale = self.head_dim ** -0.5

        # Query Projection
        self.q_proj = nn.Linear(
            embedding_dim,
            self.internal_dim,
        )

        # Key Projection
        self.k_proj = nn.Linear(
            embedding_dim,
            self.internal_dim,
        )

        # Value Projection
        self.v_proj = nn.Linear(
            embedding_dim,
            self.internal_dim,
        )

        # Output Projection
        self.out_proj = nn.Linear(
            self.internal_dim,
            embedding_dim,
        )

    def forward(
        self,
        q,
        k,
        v,
    ):

        # Query Projection (B,Nq,256) -> (B,Nq,128)
        Q = self.q_proj(q)

        # Key Projection (B,Nk,256) -> (B,Nk,128)
        K = self.k_proj(k)

        # Value Projection (B,Nk,256) -> (B,Nk,128)
        V = self.v_proj(v)

        B = Q.shape[0]

        Nq = Q.shape[1]

        Nk = K.shape[1]

        # -------------------------------------------------
        # Split Heads
        # -------------------------------------------------

        # Query (B,Nq,128) -> (B,8,Nq,16)
        Q = Q.reshape(
            B,
            Nq,
            self.num_heads,
            self.head_dim,
        ).permute(
            0,
            2,
            1,
            3,
        )

        # Key (B,Nk,128) -> (B,8,Nk,16)
        K = K.reshape(
            B,
            Nk,
            self.num_heads,
            self.head_dim,
        ).permute(
            0,
            2,
            1,
            3,
        )

        # Value (B,Nk,128) -> (B,8,Nk,16)
        V = V.reshape(
            B,
            Nk,
            self.num_heads,
            self.head_dim,
        ).permute(
            0,
            2,
            1,
            3,
        )

        # -------------------------------------------------
        # Attention Scores (B,8,Nq,16) @ (B,8,16,Nk) -> (B,8,Nq,Nk)
        # -------------------------------------------------

        attn = (Q @ K.transpose(-2, -1)) * self.scale

        # Softmax (B,8,Nq,Nk) -> (B,8,Nq,Nk)
        attn = torch.softmax(
            attn,
            dim=-1,
        )

        # Weighted Sum (B,8,Nq,Nk) @ (B,8,Nk,16) -> (B,8,Nq,16)
        out = attn @ V

        # Merge Heads (B,8,Nq,16) -> (B,Nq,128)
        out = out.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            B,
            Nq,
            self.internal_dim,
        )

        # Output Projection (B,Nq,128) -> (B,Nq,256)
        out = self.out_proj(out)

        return out