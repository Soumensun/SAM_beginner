import torch
import torch.nn as nn


class MLPBlock(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        mlp_dim=2048,
        activation=nn.GELU,
    ):
        super().__init__()

        # First Linear Layer
        self.lin1 = nn.Linear(
            embedding_dim,
            mlp_dim,
        )

        # Activation
        self.act = activation()

        # Second Linear Layer
        self.lin2 = nn.Linear(
            mlp_dim,
            embedding_dim,
        )

    def forward(self, x):

        # First Linear
        x = self.lin1(x)

        # Activation
        x = self.act(x)

        # Second Linear
        x = self.lin2(x)

        return x
