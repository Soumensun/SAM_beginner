import torch

from segment_anything import sam_model_registry
from models import SAM


# --------------------------------------------------
# Paths
# --------------------------------------------------

CHECKPOINT_PATH = "checkpoint/sam_vit_b_01ec64.pth"


# --------------------------------------------------
# Load Official Model
# --------------------------------------------------

print("=" * 80)
print("Loading Official SAM...")
print("=" * 80)

official_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT_PATH,
)

official_model.eval()


# --------------------------------------------------
# Create Our Model
# --------------------------------------------------

print("=" * 80)
print("Creating Our SAM...")
print("=" * 80)

our_model = SAM()

our_model.eval()


# --------------------------------------------------
# Load Official State Dict
# --------------------------------------------------

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

print("=" * 80)
print("Loading Weights...")
print("=" * 80)

missing_keys, unexpected_keys = our_model.load_state_dict(
    our_state,
    strict=True,
)

print()

print("Missing Keys :", len(missing_keys))
for key in missing_keys:
    print(key)

print()

print("Unexpected Keys :", len(unexpected_keys))
for key in unexpected_keys:
    print(key)

print()

if len(missing_keys) == 0 and len(unexpected_keys) == 0:
    print("All weights loaded successfully.")
else:
    print("Some parameters could not be loaded.")


# --------------------------------------------------
# Save Our Model
# --------------------------------------------------

torch.save(
    our_model.state_dict(),
    "our_sam_loaded.pth",
)

print()

print("Saved model as our_sam_loaded.pth")