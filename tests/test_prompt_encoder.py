import torch

from segment_anything import sam_model_registry
from models import SAM


CHECKPOINT = "checkpoint/sam_vit_b_01ec64.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


official_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT,
).to(DEVICE)

official_model.eval()


our_model = SAM().to(DEVICE)

# Map qkv to separate query, key, value projections and proj to out_proj
official_state = official_model.state_dict()
our_state = {}
for k, v in official_state.items():
    if "image_encoder.blocks." in k and ".attn.qkv." in k:
        prefix = k.split(".attn.qkv.")[0]
        suffix = k.split(".attn.qkv.")[1]
        q, k_t, v_t = torch.chunk(v, 3, dim=0)
        our_state[f"{prefix}.attn.q_proj.{suffix}"] = q
        our_state[f"{prefix}.attn.k_proj.{suffix}"] = k_t
        our_state[f"{prefix}.attn.v_proj.{suffix}"] = v_t
    elif "image_encoder.blocks." in k and ".attn.proj." in k:
        prefix = k.split(".attn.proj.")[0]
        suffix = k.split(".attn.proj.")[1]
        our_state[f"{prefix}.attn.out_proj.{suffix}"] = v
    else:
        our_state[k] = v

our_model.load_state_dict(
    our_state,
    strict=True,
)

our_model.eval()


point_coords = torch.tensor(
    [[[512.0, 512.0]]],
    device=DEVICE,
)

point_labels = torch.tensor(
    [[1]],
    device=DEVICE,
)

points = (
    point_coords,
    point_labels,
)

boxes = None

masks = None


with torch.no_grad():

    official_sparse, official_dense = (
        official_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
    )

    our_sparse, our_dense = (
        our_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
    )


print("=" * 80)
print("Sparse Prompt Shape")
print("=" * 80)

print("Official :", official_sparse.shape)

print("Ours     :", our_sparse.shape)

print()


print("=" * 80)
print("Dense Prompt Shape")
print("=" * 80)

print("Official :", official_dense.shape)

print("Ours     :", our_dense.shape)

print()


print("=" * 80)
print("Sparse Prompt Comparison")
print("=" * 80)

print(
    torch.allclose(
        official_sparse,
        our_sparse,
        atol=1e-6,
    )
)

difference = (
    official_sparse -
    our_sparse
).abs()

print(
    "Maximum Error :",
    difference.max().item(),
)

print(
    "Mean Error :",
    difference.mean().item(),
)

print()

print("=" * 80)
print("Dense Prompt Comparison")
print("=" * 80)

print(
    torch.allclose(
        official_dense,
        our_dense,
        atol=1e-6,
    )
)

difference = (
    official_dense -
    our_dense
).abs()

print(
    "Maximum Error :",
    difference.max().item(),
)

print(
    "Mean Error :",
    difference.mean().item(),
)

print()


print("=" * 80)
print("Sample Sparse")
print("=" * 80)

print()

print("Official")

print(
    official_sparse[0]
)

print()

print("Ours")

print(
    our_sparse[0]
)

print("=" * 80)
print("Sample Dense")
print("=" * 80)

print()

print("Official")

print(
    official_dense[
        0,
        :5,
        :5,
        :5,
    ]
)

print()

print("Ours")

print(
    our_dense[
        0,
        :5,
        :5,
        :5,
    ]
)