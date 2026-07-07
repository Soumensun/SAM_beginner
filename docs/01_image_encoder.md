# SAM — Image Encoder

**File:** `models/image_encoder/image_encoder.py`

The image encoder is the heaviest part of SAM. Its job is to compress a
`(B, 3, 1024, 1024)` RGB image all the way down to a compact feature map
`(B, 256, 64, 64)` that the rest of SAM will use.

It is a **Vision Transformer (ViT)** with three important upgrades specific to SAM:
1. Window Partition / Unpartition  
2. Mixed local + global attention  
3. A CNN "Neck" at the end

---

## Step 1 — Patch Embedding

**File:** `models/image_encoder/patch_embedding.py`

```
Input:  (B, 3, 1024, 1024)
Output: (B, 64, 64, 768)
```

SAM cannot process 1024×1024 individual pixels — that is 1,048,576 tokens, which
would be impossibly expensive for self-attention.

Instead, a `Conv2d` with `kernel_size=16, stride=16` slides over the image and
collapses each non-overlapping 16×16 pixel square into a single 768-dimensional
vector. Because 1024 / 16 = 64, the image becomes a **64×64 grid of patches**.

Each patch is a compressed summary of its 16×16 region.

---

## Step 2 — Absolute Position Embedding

**File:** `models/image_encoder/abs_position_embedding.py`  
**Also implemented inline:** `image_encoder.py` line 38-45

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, 768))
```

```
x = x + self.pos_embed   # (B,64,64,768) + (1,64,64,768) -> (B,64,64,768)
```

### The Problem
A Transformer has no inherent sense of space. If you shuffled all the 64×64 patches
into a random order, the transformer would produce the same result — it cannot tell
where each patch came from.

### The Solution
We create a learnable "name tag" tensor with the exact same shape as the patch grid
`(1, 64, 64, 768)`. Each spatial position `[row, col]` gets its own unique 768-dimensional
learnable vector that is trained to represent "I am at location (row, col)".

By **adding** this tensor onto the patch embeddings, every patch now carries both:
- **What it looks like** (from patch embedding)
- **Where it is** (from position embedding)

> **Why `nn.Parameter` and not a regular tensor?**  
> `nn.Parameter` tells PyTorch: "This tensor's values should be updated by the optimizer
> during training." A regular tensor would be ignored by backprop.

---

## Step 3 — Transformer Blocks with Window Attention

**File:** `models/image_encoder/block.py`

The 12 transformer blocks use **windowed self-attention** by default, switching to
**full global attention** only at 4 specific layers (`global_attn_indexes = (2, 5, 8, 11)`).

```python
window_size=0 if i in global_attn_indexes else window_size
```

This `0` is a flag: if `window_size == 0`, the block skips windowing and runs full attention.

---

## Trick: Window Partition and Unpartition

**File:** `models/image_encoder/window.py`

### Why windows?
In a standard ViT, **every patch attends to every other patch**. With 64×64 = 4,096
patches, the attention matrix is 4096 × 4096 = **16 million elements** per layer.
This is extremely slow and memory-hungry.

SAM's solution: **divide the image into small non-overlapping windows** and run
attention only *within* each window. With a window size of 14×14, each window has
only 196 tokens — much more manageable.

### How `window_partition` works

```
Input:  (B, 64, 64, 768)
Output: (B * num_windows, 14, 14, 768)
        pad_hw = (Hp, Wp)  # padded H and W if not divisible
```

**Step-by-step (with window_size=14, grid=64×64):**

```
1. Padding:
   64 is not perfectly divisible by 14 (64/14 = 4.57)
   So we pad to the next multiple: 70 (14 * 5 = 70)
   Shape: (B, 70, 70, 768)

2. Reshape into grid of windows:
   (B, 70, 70, 768) -> (B, 5, 14, 5, 14, 768)
   Reading: B images, 5 windows tall, each 14 pixels, 5 windows wide, each 14 pixels

3. Permute axes to group windows:
   (B, 5, 14, 5, 14, 768) -> (B, 5, 5, 14, 14, 768)

4. Merge batch and window dims:
   (B, 5, 5, 14, 14, 768) -> (B*25, 14, 14, 768)
```

Now we run self-attention on each of the 25 mini-images (windows) independently.

### How `window_unpartition` works

After attention, this is the exact reverse:

```
Input:  (B*25, 14, 14, 768)
Output: (B, 64, 64, 768)   (padding is removed at the end)
```

The code reverse-engineers the batch size `B` using:
```python
B = windows.shape[0] // ((Hp // window_size) * (Wp // window_size))
```

Then it undoes the permutation and reshaping steps, and crops away the padding:
```python
x = x[:, :H, :W, :]
```

---

## Step 4 — The Neck (Dimension Reduction)

```
Input:  (B, 768, 64, 64)
Output: (B, 256, 64, 64)
```

After all 12 transformer blocks, the feature map still has 768 channels (the full
embed dim). A small CNN neck reduces this to 256 channels, which is the standard
"SAM channel size" used by all other modules.

The neck contains:
- `Conv2d(768 -> 256, kernel=1)` — pointwise conv to reduce channels
- `LayerNorm2d(256)` — normalize the 2D feature map
- `Conv2d(256 -> 256, kernel=3, padding=1)` — small spatial mixing
- `LayerNorm2d(256)` — normalize again

---

## LayerNorm2d — A Custom Normalization

**File:** `models/image_encoder/layernorm2d.py`

```python
class LayerNorm2d(nn.Module):
    # Input:  (B, C, H, W)
    # Output: (B, C, H, W)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)        # average over C
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
```

### Why not just use PyTorch's `nn.LayerNorm`?
PyTorch's built-in `nn.LayerNorm` expects the last dimension to be the channel.
That works perfectly for sequences (shape `B, T, C`) but is awkward for 2D images
in `(B, C, H, W)` format.

`LayerNorm2d` computes the mean and variance **across the channel dimension** (`dim=1`)
for each spatial location, keeping the `(H, W)` spatial structure intact.

The `[:, None, None]` trick broadcasts the 1D weight and bias vectors so they
can be applied across all spatial positions:
```python
# weight shape: (C,)
# weight[:, None, None] shape: (C, 1, 1)
# This broadcasts correctly against (B, C, H, W)
```

---

## Full Forward Pass Summary

```
(B, 3, 1024, 1024)
    │ PatchEmbedding (Conv2d 16x16 stride 16)
    ▼
(B, 64, 64, 768)
    │ + abs_pos_embed
    ▼
(B, 64, 64, 768)
    │ 12x Block (8 windowed, 4 global)
    │   each block: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
    ▼
(B, 64, 64, 768)
    │ permute BHWC -> BCHW
    ▼
(B, 768, 64, 64)
    │ Neck (Conv 1x1 + LayerNorm2d + Conv 3x3 + LayerNorm2d)
    ▼
(B, 256, 64, 64)   ← Final image embedding passed to Mask Decoder
```
