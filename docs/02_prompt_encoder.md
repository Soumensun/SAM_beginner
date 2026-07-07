# SAM — Prompt Encoder

**File:** `models/prompt_encoder/prompt_encoder.py`

The Prompt Encoder is what makes SAM interactive. It converts whatever the user
provides (a click, a box, or a rough mask) into numerical vectors that the Mask
Decoder can understand.

SAM supports **3 types of prompts**. They are not mutually exclusive — you can
combine them.

---

## The Two Output Types

No matter what prompts you provide, the encoder always outputs two things:

| Output | Shape | Meaning |
|---|---|---|
| `sparse_embeddings` | `(B, N, 256)` | Tokens for points + boxes. N depends on how many prompts. |
| `dense_embeddings` | `(B, 256, 64, 64)` | A spatial map for mask prompts (or a blank if no mask). |

---

## Prompt Type 1 — Points

A point prompt is a click on the image. The user says: *"the object is here"* (foreground)
or *"the object is NOT here"* (background).

**Input:**
- `coords`: `(B, N, 2)` — pixel coordinates of the clicks  
- `labels`: `(B, N)` — `1` = foreground, `0` = background

**Encoding steps:**

### A. Sinusoidal Position Encoding
The raw pixel coordinate `(x, y)` is first normalized to the range `[0, 1]` relative
to the full 1024×1024 image, then passed through **Random Fourier Features (sinusoidal)**
to turn it into a 256-dimensional vector:
```
(B, N, 2)  →  (B, N, 256)
```

This gives the model a rich, high-frequency description of location, not just a
raw number.

### B. Label Embedding (the "type" label)
Just knowing the location is not enough — the model also needs to know **what that
point means**. Is it a foreground click or a background click?

SAM has 4 learnable type embeddings stored in `self.point_embeddings`:
- Index 0: background point
- Index 1: foreground point
- Index 2: top-left corner of a box
- Index 3: bottom-right corner of a box

These learned vectors are **added on top** of the position encoding:
```python
point_embedding[labels == 0] += self.point_embeddings[0].weight  # background
point_embedding[labels == 1] += self.point_embeddings[1].weight  # foreground
```

### C. Padding Point (when no box is used)
If the user provided only points (no box), SAM pads the token sequence with a
dummy "not-a-point" token at position `(0, 0)` with label `-1`. This ensures the
decoder always gets a fixed-structure input.

```python
point_embedding[labels == -1] = 0.0
point_embedding[labels == -1] += self.not_a_point_embed.weight
```

---

## Prompt Type 2 — Bounding Boxes

A box prompt tells SAM: *"the object is somewhere inside this rectangle."*

**Input:**
- `boxes`: `(B, 4)` — `[x_min, y_min, x_max, y_max]`

**Encoding steps:**

The box is reshaped into 2 corner points:
```
(B, 4) → (B, 2, 2)
```

These 2 corners are each encoded with the same sinusoidal position encoding as
points, resulting in `(B, 2, 256)`.

Then the **top-left corner** gets `self.point_embeddings[2]` added, and the
**bottom-right corner** gets `self.point_embeddings[3]` added. This teaches the
model to distinguish which corner is which.

```python
corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # top-left
corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # bottom-right
```

These 2 tokens are then concatenated into `sparse_embeddings`.

---

## Prompt Type 3 — Masks (Dense Prompt)

A mask prompt is a rough segmentation map — for example, the output from a
previous SAM forward pass that the user wants to refine.

**Input:** `(B, 1, 256, 256)` — a coarse binary or soft mask

Because this is a full 2D map (not a small list of points), it is encoded **differently**
from points and boxes. It is NOT added to `sparse_embeddings`. Instead it becomes
the `dense_embeddings` that gets **added directly onto the image features** before
the decoder.

**Encoding steps (mask downscaling network):**

```
(B, 1, 256, 256)
    │ Conv2d(1 → 4, k=2, s=2)      → (B, 4, 128, 128)
    │ LayerNorm2d(4) + GELU
    │ Conv2d(4 → 16, k=2, s=2)     → (B, 16, 64, 64)
    │ LayerNorm2d(16) + GELU
    │ Conv2d(16 → 256, k=1)        → (B, 256, 64, 64)
    ▼
(B, 256, 64, 64)
```

This matches the spatial size of the image embedding `(B, 256, 64, 64)` so it can
be added directly.

### When no mask is provided:
```python
dense_embeddings = self.no_mask_embed.weight.reshape(1, 256, 1, 1).expand(B, -1, 64, 64)
```
A single learned "no-mask" vector is broadcast across the entire 64×64 spatial grid.
This acts as a learned default: *"no mask was provided."*

---

## Combined Output

After processing all prompts, the outputs are:

```
sparse_embeddings: (B, N, 256)
   ├── [0..N_points-1] : point tokens
   └── [N_points..N_points+1] : box corner tokens (if box provided)

dense_embeddings: (B, 256, 64, 64)
   └── Encoded mask (or no_mask_embed if no mask)
```

`N` is variable — it depends on how many points and whether a box was given.
This is why the Two-Way Transformer must be able to handle a **variable-length**
token sequence.
