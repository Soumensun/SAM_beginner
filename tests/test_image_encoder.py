import torch

from segment_anything import sam_model_registry
from models import SAM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = "checkpoint/sam_vit_b_01ec64.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Load Official Model
# --------------------------------------------------

official_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT,
).to(DEVICE)

official_model.eval()


# --------------------------------------------------
# Load Our Model
# --------------------------------------------------

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


# --------------------------------------------------
# Dummy Input
# --------------------------------------------------

image = torch.randn(
    1,
    3,
    1024,
    1024,
).to(DEVICE)


# --------------------------------------------------
# Forward
# --------------------------------------------------

with torch.no_grad():

    official_output = official_model.image_encoder(
        image
    )

    our_output = our_model.image_encoder(
        image
    )


# --------------------------------------------------
# Shape
# --------------------------------------------------

print("=" * 80)
print("Image Encoder Shape")
print("=" * 80)

print("Official :", official_output.shape)

print("Ours     :", our_output.shape)

print()


# --------------------------------------------------
# Numerical Comparison
# --------------------------------------------------

print("=" * 80)
print("Image Encoder Numerical Comparison")
print("=" * 80)

print(
    "Allclose :",
    torch.allclose(
        official_output,
        our_output,
        atol=1e-6,
    ),
)

print()

difference = (
    official_output -
    our_output
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


# --------------------------------------------------
# Sample Tensor
# --------------------------------------------------

print("=" * 80)
print("Sample Tensor")
print("=" * 80)

print()

print("Official")

print(
    official_output[
        0,
        :5,
        :5,
        :5,
    ]
)

print()

print("Ours")

print(
    our_output[
        0,
        :5,
        :5,
        :5,
    ]
)