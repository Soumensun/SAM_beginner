import torch

from SAM_official.segment_anything import sam_model_registry
from models import SAM


# --------------------------------------------------
# Load Official Model
# --------------------------------------------------

official_model = sam_model_registry["vit_b"](
    checkpoint="checkpoint/sam_vit_b_01ec64.pth",
)

official_model.eval()


# --------------------------------------------------
# Load Our Model
# --------------------------------------------------

our_model = SAM()

our_model.eval()


# --------------------------------------------------
# Get State Dicts
# --------------------------------------------------

official_state = official_model.state_dict()
our_state = our_model.state_dict()


# --------------------------------------------------
# Compare Number of Parameters
# --------------------------------------------------

print("=" * 80)
print("Official Parameters :", len(official_state))
print("Our Parameters      :", len(our_state))
print("=" * 80)


# --------------------------------------------------
# Missing Keys
# --------------------------------------------------

missing_keys = sorted(
    set(official_state.keys()) - set(our_state.keys())
)

print("\nMissing Keys :", len(missing_keys))

for key in missing_keys:
    print(key)


# --------------------------------------------------
# Unexpected Keys
# --------------------------------------------------

unexpected_keys = sorted(
    set(our_state.keys()) - set(official_state.keys())
)

print("\nUnexpected Keys :", len(unexpected_keys))

for key in unexpected_keys:
    print(key)


# --------------------------------------------------
# Compare Shapes
# --------------------------------------------------

print("\nChecking Parameter Shapes...\n")

shape_error = False

for key in official_state.keys():

    if key in our_state:

        if official_state[key].shape != our_state[key].shape:

            shape_error = True

            print(f"{key}")

            print("Official :", official_state[key].shape)

            print("Ours     :", our_state[key].shape)

            print()

if not shape_error:
    print("All parameter shapes match.")


# --------------------------------------------------
# Try Strict Loading
# --------------------------------------------------

print("\nTrying strict=True loading...\n")

try:

    our_model.load_state_dict(
        official_state,
        strict=True,
    )

    print("SUCCESS : strict=True passed.")

except Exception as e:

    print("FAILED")

    print(e)