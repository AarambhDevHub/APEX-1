# 32 — Vision Capabilities

APEX-1 v2.3.0 adds the first vision-language path.

The core idea is simple:

```text
image → vision encoder → projector/resampler → visual tokens → APEX transformer
```

The language model remains decoder-only. Images become **continuous visual tokens** that are inserted into the prompt at the existing `<|img|>` token.

---

## Why not convert images into text first?

A caption-only pipeline loses information. If an external captioner says “a car on a road,” the language model never sees the visual layout, colors, small objects, text in the image, or spatial details.

APEX-1 Vision instead passes image features directly into the model context.

---

## Architecture

```text
pixel_values [B, 3, H, W]
        │
        ▼
┌────────────────────────┐
│ NativeVisionEncoder    │
│ ViT-style patch encoder│
└──────────┬─────────────┘
           │ image_features [B, N, d_vision]
           ▼
┌────────────────────────┐
│ VisionToTextProjector  │
│ Perceiver/MLP bridge   │
└──────────┬─────────────┘
           │ visual_tokens [B, T, d_model]
           ▼
text: <|user|> <|img|> What is this? <|assistant|>
                 │
                 ▼
replace <|img|> with T visual tokens
                 │
                 ▼
APEX-1 transformer blocks
                 │
                 ▼
text logits
```

---

## Why Perceiver resampling?

A 224×224 image with 16×16 patches creates 196 patch tokens. Higher resolution creates many more. Feeding all patch tokens directly into the LLM is expensive.

The Perceiver-style resampler compresses many patch features into a fixed number of visual tokens, such as 64. This keeps compute stable and makes long context training easier.

---

## New config section

```yaml
vision:
  enabled: true
  image_size: 224
  patch_size: 16
  in_channels: 3
  encoder_type: native_vit
  d_vision: 512
  n_layers: 6
  n_heads: 8
  mlp_ratio: 4.0
  dropout: 0.0
  projector_type: perceiver
  n_visual_tokens: 64
  projector_hidden_dim: 1024
  projector_layers: 2
  image_token_id: 8
  freeze_vision_encoder: false
  freeze_language_model: false
```

---

## Basic forward pass

```python
import torch

from apex.config import get_tiny_vision_config
from apex.model.apex_vision_model import APEX1VisionModel

cfg = get_tiny_vision_config()
model = APEX1VisionModel(cfg)

# token 8 is <|img|> in the minimal tokenizer fallback
token_ids = torch.tensor([[1, 4, 8, 20, 21, 5, 30]])
pixel_values = torch.randn(1, 3, 32, 32)

out = model(token_ids=token_ids, pixel_values=pixel_values)
print(out["logits"].shape)
```

---

## Training labels

Because `<|img|>` is replaced by continuous visual tokens, labels must be expanded:

```python
from apex.training.vision_losses import expand_labels_for_visual_tokens

expanded_labels = expand_labels_for_visual_tokens(
    token_ids=token_ids,
    labels=labels,
    image_token_id=cfg.vision.image_token_id,
    n_visual_tokens=cfg.vision.n_visual_tokens,
)
```

The inserted visual token positions become `-100`, so cross entropy ignores them.

---

## Training stages

### Stage 1 — Alignment

Freeze the language model and train only:

- vision encoder
- vision projector

Use image-caption pairs. The target is normal next-token prediction for the caption/answer only.

### Stage 2 — Instruction tuning

Train on image-question-answer examples:

```json
{"image":"cat.jpg","prompt":"What is in this image?","response":"A cat sitting on a chair."}
```

Start by unfreezing:

- projector
- final language layers
- optionally vision encoder

### Stage 3 — Advanced visual reasoning

Add datasets for:

- OCR
- charts
- diagrams
- screenshots
- spatial reasoning
- multi-image comparison

---

## Important limitation

This code adds the **architecture and training path** for vision. It will not magically understand real images until trained or connected to a pretrained vision encoder. The tiny demo uses random weights, so it only proves that the pipeline works.

---

## Best next upgrade

For strong real-world understanding, add an optional frozen CLIP/SigLIP/DINOv2 encoder adapter:

```text
frozen pretrained vision encoder → APEX projector → APEX language model
```

Then train the projector first. This is much cheaper than training a vision encoder from scratch.
