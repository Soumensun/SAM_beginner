import torch
import torch.nn as nn

class AbsPositionEmbedding(nn.Module):
    def __init__(self,embed_dim=768,image_size=1024,patch_size=16):
        super().__init__()
        self.embed_dim=embed_dim
        self.grid_size = image_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1,self.grid_size,self.grid_size,embed_dim))


    def forward(self,x):

        x = x + self.pos_embed
        return x
