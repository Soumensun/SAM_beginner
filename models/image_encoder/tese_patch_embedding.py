import torch

from sam.image_encoder.patch_embedding import PatchEmbedding


model = PatchEmbedding()

x = torch.randn(2, 3, 1024, 1024)

out = model(x)

print("Input :", x.shape)
print("Output:", out.shape)