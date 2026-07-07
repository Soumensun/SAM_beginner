# SAM — Two-Way Transformer

**Files:**  
- `models/two_way_transformer/two_way_transformer.py`  
- `models/two_way_transformer/attention_block.py`

The Two-Way Transformer is the most unique and creative part of SAM.  
Standard transformers only flow information in one direction.  
SAM's transformer flows information in **both directions simultaneously**:  
- **Prompts → Image** (the prompt learns *what* in the image to look at)  
- **Image → Prompts** (the image features update based on the prompt's location)

---

## The N+5 Token Sequence

Before the Two-Way Transformer even starts, the Mask Decoder builds a special
combined token sequence.

### The 5 "special" output tokens

The decoder creates 5 learnable tokens from scratch:

```python
# 1 IoU token
self.iou_token = nn.Embedding(1, 256)          # shape: (1, 256)

# 4 mask tokens
self.num_mask_tokens = num_multimask_outputs + 1   # 3 + 1 = 4
self.mask_tokens = nn.Embedding(4, 256)             # shape: (4, 256)
```

These are stacked together:
```python
output_tokens = torch.cat([iou_token.weight, mask_tokens.weight], dim=0)
# Shape: (5, 256)
```

| Token Index | Name | Purpose |
|---|---|---|
| 0 | IoU Token | Will become the quality score for each mask |
| 1 | Mask Token 0 | Will generate the single "best" mask |
| 2 | Mask Token 1 | Will generate mask (ambiguity level 1) |
| 3 | Mask Token 2 | Will generate mask (ambiguity level 2) |
| 4 | Mask Token 3 | Will generate mask (ambiguity level 3) |

### Concatenating with prompt tokens

The output tokens and the sparse prompt embeddings are concatenated into one sequence:

```python
tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
# (B, 5, 256) + (B, N, 256) → (B, 5+N, 256)
```

This is the **N+5 sequence**. The transformer processes all of these together.
`N` is variable (depends on how many prompts the user gave).

---

## Inside the Two-Way Transformer

**File:** `models/two_way_transformer/two_way_transformer.py`

The transformer has 2 identical `TwoWayAttentionBlock` layers.

Before entering, the image features are flattened:
```
(B, 256, 64, 64) → (B, 4096, 256)   (64*64 = 4096 image tokens)
```

Now we have two parallel token streams:
- **queries** = `(B, 5+N, 256)` — our prompt tokens + output tokens
- **keys** = `(B, 4096, 256)` — the image feature tokens

---

## Inside One TwoWayAttentionBlock

**File:** `models/two_way_transformer/attention_block.py`

Each block performs **4 sub-operations** in sequence:

### Step 1 — Self-Attention on Prompt Tokens

The prompt tokens (queries) attend to each other.  
This lets the tokens "talk to each other" — e.g., the IoU token can look at
what the mask tokens are doing.

```
queries (B, 5+N, 256) → self-attention → queries (B, 5+N, 256)
+ LayerNorm
```

> **Note on `skip_first_layer_pe`:**  
> In the very first block, position encodings are NOT added before self-attention.
> This is because the tokens haven't seen the image yet — adding position info
> before the first cross-attention step would be premature.

### Step 2 — Cross-Attention: Prompt → Image

The prompt tokens **look at the image** to gather relevant visual features.

```
q = queries + query_pe    # Add prompt position encoding
k = keys + key_pe         # Add image position encoding

cross_attn_token_to_image(q, k, v=keys)
    → each prompt token aggregates info from the relevant image regions

queries = queries + attn_out    # Residual connection
+ LayerNorm
```

This is where the model learns: *"I am a foreground click at (300, 200),
let me look at that part of the image."*

### Step 3 — MLP on Prompt Tokens

A standard feed-forward MLP refines each prompt token independently:

```
queries = queries + MLP(queries)
+ LayerNorm
```

### Step 4 — Cross-Attention: Image → Prompt (The Unique Direction!)

Now the image tokens **look at the prompt tokens** to understand the context:

```
q = keys + key_pe           # Image tokens become queries here!
k = queries + query_pe      # Prompt tokens become keys!

cross_attn_image_to_token(q=k, k=q, v=queries)
    → each image token aggregates info from the prompt tokens

keys = keys + attn_out
+ LayerNorm
```

This is what makes it "two-way." The image features themselves get updated
based on what the user prompted. A region far from the user's click learns
to suppress its own importance; a region near the click is amplified.

---

## Final Attention Layer (outside the blocks)

After both TwoWayAttentionBlocks, one final cross-attention is run:

```python
q = queries + point_embedding    # final prompt tokens with PE
k = keys + image_pe              # final image tokens with PE

attn_out = final_attn_token_to_image(q, k, v=keys)
queries = queries + attn_out
queries = norm_final_attn(queries)
```

This is one more round of "prompts look at image" to make sure the output
tokens have fully absorbed the relevant image features.

---

## What Comes Out of the Two-Way Transformer

```
return queries, keys
```

| Output | Shape | Content |
|---|---|---|
| `queries` | `(B, 5+N, 256)` | Updated prompt + output tokens (rich with image info) |
| `keys` | `(B, 4096, 256)` | Updated image features (informed by the prompt) |

The Mask Decoder then slices these:

```python
# The IoU token output (index 0)
iou_token_out = queries[:, 0, :]           # (B, 256)

# The 4 mask token outputs (indices 1 to 4)
mask_tokens_out = queries[:, 1:5, :]      # (B, 4, 256)

# The updated image features (reshaped back to spatial)
keys = keys.reshape(B, 64, 64, 256).permute(0, 3, 1, 2)  # (B, 256, 64, 64)
```

These three things are what produce the final masks and IoU scores.

---

## Why Two-Way? The Intuition

Imagine you're using a GPS. In a one-way system:
- You just tell the GPS your destination. The GPS gives you directions.

In a two-way system:
- You tell the GPS your destination (Prompt → Image)
- The GPS also tells you about relevant road conditions near your destination (Image → Prompt)
- Both of you update your understanding of each other

In SAM, after the two-way transformer, the image features "know" they are being
asked about a specific object, and the prompt tokens "know" exactly what that
object looks like in the image. Both sides are enriched.
