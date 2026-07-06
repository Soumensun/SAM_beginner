import torch
import torch.nn as nn


class MLPBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        mlp_dim=2048,
        activation=nn.ReLU,
    ):
        super().__init__()

        # First Linear Layer (256 -> 2048)
        self.lin1 = nn.Linear(
            embedding_dim,
            mlp_dim,
        )

        # Activation
        self.act = activation()

        # Second Linear Layer (2048 -> 256)
        self.lin2 = nn.Linear(
            mlp_dim,
            embedding_dim,
        )

    def forward(self, x):

        # First Linear (B,N,256) -> (B,N,2048)
        x = self.lin1(x)

        # Activation (B,N,2048) -> (B,N,2048)
        x = self.act(x)

        # Second Linear (B,N,2048) -> (B,N,256)
        x = self.lin2(x)

        return x