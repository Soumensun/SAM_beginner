# SAM — Mask Decoder

**File:** `models/mask_decoder/mask_decoder.py`

The Mask Decoder takes the outputs of the Two-Way Transformer and converts them
into actual pixel-level segmentation masks.

---

## Inputs

```python
def predict_masks(
    image_embeddings,         # (B, 256, 64, 64) - raw image features
    image_pe,                 # (B, 256, 64, 64) - positional encoding for image
    sparse_prompt_embeddings, # (B, N, 256)       - point/box prompt tokens
    dense_prompt_embeddings,  # (B, 256, 64, 64)  - mask prompt (or no_mask_embed)
):
```

---

## Step 1 — Build the N+5 Token Sequence

```python
output_tokens = torch.cat([iou_token.weight, mask_tokens.weight], dim=0)
# (1, 256) + (4, 256) → (5, 256)

output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
# (5, 256) → (B, 5, 256)

tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
# (B, 5, 256) + (B, N, 256) → (B, 5+N, 256)
```

The dense prompt is added to the image embedding (not tokens):
```python
image_embeddings = image_embeddings + dense_prompt_embeddings
# (B, 256, 64, 64) + (B, 256, 64, 64) → (B, 256, 64, 64)
```

---

## Step 2 — Two-Way Transformer

```python
queries, keys = self.transformer(
    image_embeddings,    # (B, 256, 64, 64)
    image_pe,            # (B, 256, 64, 64)
    tokens,              # (B, 5+N, 256)
)
# queries : (B, 5+N, 256)  — updated prompt tokens
# keys    : (B, 4096, 256) — updated image tokens (flattened 64*64)
```

---

## Step 3 — Slice the Outputs

```python
iou_token_out   = queries[:, 0, :]          # (B, 256)  — IoU predictor input
mask_tokens_out = queries[:, 1:1+4, :]      # (B, 4, 256) — one vector per mask

# Reshape image tokens back to spatial
keys = keys.reshape(B, 64, 64, 256).permute(0, 3, 1, 2)   # (B, 256, 64, 64)
```

---

## Step 4 — Output Upscaling

The image features are at `64×64` resolution. The final masks must be at
`256×256`. Two transposed convolutions (stride=2 each) give a 4× upscale:

```python
self.output_upscaling = nn.Sequential(
    ConvTranspose2d(256, 64, kernel_size=2, stride=2),  # 64×64 → 128×128
    LayerNorm2d(64),
    GELU(),
    ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 128×128 → 256×256
    GELU(),
)
upscaled_embedding = self.output_upscaling(keys)
# (B, 256, 64, 64) → (B, 32, 256, 256)
```

Now we have a rich high-resolution feature map, but we don't yet know
**which object to highlight** — that depends on the prompt. This is where
the Hypernetwork comes in.

---

## Step 5 — Hypernetwork MLP (Variable-Weight MLP)

**File:** `models/mask_decoder/mlp.py`

```python
self.output_hypernetworks_mlps = nn.ModuleList([
    MLP(input_dim=256, hidden_dim=256, output_dim=32, num_layers=3)
    for _ in range(self.num_mask_tokens)   # 4 MLPs, one per mask token
])
```

Each of the 4 mask tokens is passed through its own dedicated MLP:

```python
hyper_in_list = []
for i in range(4):
    hyper_in_list.append(
        self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
    )
    # Input:  (B, 256)
    # Output: (B, 32)

hyper_in = torch.stack(hyper_in_list, dim=1)
# (B, 4, 32)
```

### What are these 32 numbers?

These are **dynamic weights**. They encode the answer to:
*"Given this specific prompt, what spatial pattern should I look for in the image?"*

The MLP itself has **static, trained weights** — they were learned during training
over millions of examples. However, because the *input* to the MLP changes with
every new prompt, the *output* (the 32 numbers) also changes. Hence they are
called dynamic weights.

This is the **Hypernetwork** pattern: a network that generates the weights
(or parameters) for another computation.

---

## Step 6 — Mask Generation via Matrix Multiplication

This is the most elegant step. The dynamic weights `hyper_in` are
matrix-multiplied against the high-resolution image features:

```python
# Flatten spatial dims of upscaled_embedding
upscaled_embedding = upscaled_embedding.view(B, 32, 256*256)
# (B, 32, 65536)

masks = hyper_in @ upscaled_embedding
# (B, 4, 32) @ (B, 32, 65536) → (B, 4, 65536)

masks = masks.view(B, 4, 256, 256)
# (B, 4, 256, 256)
```

### Why does this work?

Each row of `hyper_in[b, i, :]` is a 32-dimensional vector. When you
multiply it against the 32×65536 image feature matrix, you get a dot product
at every single pixel. This dot product is high where the pixel's features
match the prompt's expectations, and low where they don't.

The result is naturally a **probability map** that highlights the object
the user pointed to — a segmentation mask.

This matrix multiplication is mathematically equivalent to applying a
**dynamic 1×1 convolution** to the image features, where the convolution
kernel is generated fresh by the MLP for each prompt.

---

## Step 7 — IoU Prediction

```python
iou_pred = self.iou_prediction_head(iou_token_out)
# (B, 256) → (B, 4)
```

The IoU prediction head is a simple 3-layer MLP that takes the IoU token
and predicts how good each of the 4 masks is (Intersection over Union score).
This lets SAM rank its own masks.

---

## Step 8 — Selecting the Final Masks

```python
def forward(self, ..., multimask_output):
    masks, iou_pred = self.predict_masks(...)   # (B, 4, 256, 256)

    if multimask_output:
        mask_slice = slice(1, None)   # Return masks 1, 2, 3
    else:
        mask_slice = slice(0, 1)      # Return only mask 0

    masks   = masks[:, mask_slice, :, :]   # (B, 3, 256, 256) or (B, 1, 256, 256)
    iou_pred = iou_pred[:, mask_slice]      # (B, 3)           or (B, 1)
```

| `multimask_output` | Masks returned | Use case |
|---|---|---|
| `True` | Masks 1, 2, 3 — three levels of granularity | Ambiguous prompt (one click) |
| `False` | Mask 0 — the single best overall mask | Clear/specific prompt (box) |

---

## Complete Data Flow Summary

```
image_embeddings (B,256,64,64)
dense_prompt     (B,256,64,64)   → add together → (B,256,64,64)
                                                        │
tokens (B,5+N,256) ─────────────────────────────────── │
                                                        ▼
                                          Two-Way Transformer
                                                        │
                        ┌───────────────────────────────┘
                        │
              queries (B,5+N,256)       keys (B,4096,256)
                        │                       │
         ┌──────────────┘                       │ reshape
         │                                      ▼
  iou_token_out (B,256)          (B,256,64,64)
  mask_tokens_out (B,4,256)               │
         │                                │ output_upscaling
         │ Hypernetwork MLPs              ▼
         ▼                       (B,32,256,256)
  hyper_in (B,4,32)                       │
         │                                │ flatten to (B,32,65536)
         └──────────── @ ─────────────────┘
                        │
                (B,4,65536) → view → (B,4,256,256)  ← masks

  iou_token_out (B,256) → iou_prediction_head → (B,4) ← iou_pred

                        │ slice by multimask_output
                        ▼
  Final: masks (B,1or3,256,256) + iou_pred (B,1or3)
```
