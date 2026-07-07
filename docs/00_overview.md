# SAM — Segment Anything Model: Architecture Overview

SAM is a promptable image segmentation model built by Meta AI Research (2023).  
It is **not** just a fine-tuned ViT — it introduces several brand-new design ideas that
make it fundamentally different from a standard Vision Transformer or a classic segmentation
model like U-Net.

---

## The Big Picture — Three Modules

```
Image (B,3,1024,1024)
        │
        ▼
┌─────────────────┐
│  Image Encoder  │  ── Heavy ViT (run once per image)
│  (ViT-H/L/B)   │
└────────┬────────┘
         │  (B,256,64,64)
         ▼
┌─────────────────┐       ┌──────────────────────┐
│ Prompt Encoder  │ ◄─────│ User Prompt           │
│  (lightweight)  │       │  - Point(s) + Labels  │
└────────┬────────┘       │  - Box                │
         │                │  - Mask               │
         │ sparse (B,N,256) + dense (B,256,64,64) └──────────────────────┘
         ▼
┌─────────────────┐
│  Mask Decoder   │  ── Two-Way Transformer + Hypernetwork
└────────┬────────┘
         │
         ▼
  masks (B, 1 or 3, 256, 256)
  iou_predictions (B, 1 or 3)
```

---

## What Makes SAM Different From a Standard ViT?

| Feature | Standard ViT | SAM |
|---|---|---|
| Attention scope | Full global (all patches attend to all) | Windowed (local) + global at select layers |
| Position encoding | Absolute learned | Absolute (image) + Relative (per-window) |
| Prompt support | None | 3 types: points, boxes, masks |
| Decoder | Simple linear head | Two-Way Transformer + Hypernetwork MLP |
| Output | Single class map | 4 candidate masks + IoU scores |
| Normalization | LayerNorm (1D) | Custom LayerNorm2d (spatial feature maps) |

---

## Documentation Index

| File | Topic |
|---|---|
| `01_image_encoder.md` | Patch Embedding, Absolute Position Encoding, Window Partition/Unpartition, LayerNorm2d |
| `02_prompt_encoder.md` | The 3 prompt types: Points, Boxes, Masks and how each is encoded |
| `03_two_way_transformer.md` | The N+5 token sequence, Two-Way Attention Block, and what comes out |
| `04_mask_decoder.md` | Output upscaling, the Hypernetwork MLP, mask generation, IoU prediction |
| `05_unique_tricks.md` | Deep-dives: zip trick, nn.ModuleList, LayerNorm2d, Variable-weight MLP |
