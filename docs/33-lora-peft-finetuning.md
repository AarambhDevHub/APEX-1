# 33 — LoRA & PEFT Fine-Tuning

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Time to read:** ~25 minutes  
> **You will learn:** How APEX-1 fine-tunes with LoRA adapters instead of updating the full model.

---

## 1. What Is Fine-Tuning?

Pretraining teaches a model general language patterns. Fine-tuning teaches it a specific behavior.

Examples:

- Answer in a customer support style
- Follow your app's instruction format
- Learn a domain-specific writing style
- Improve code answers for a specific stack

Full fine-tuning updates all model weights. That works, but it is expensive.

---

## 2. What Is PEFT?

**PEFT** means **Parameter-Efficient Fine-Tuning**.

Instead of updating every parameter in the model, PEFT updates only a small number of extra parameters.

The base model stays mostly frozen.

This is useful because:

- less GPU memory
- smaller checkpoints
- faster experiments
- easier adapter sharing
- less risk of destroying base model knowledge

---

## 3. What Is LoRA?

**LoRA** means **Low-Rank Adaptation**.

A normal linear layer does:

\[
y = xW^T
\]

LoRA keeps `W` frozen and adds a small trainable update:

\[
y = xW^T + \frac{\alpha}{r}xA^TB^T
\]

Where:

| Symbol | Meaning |
|---|---|
| `W` | original frozen weight |
| `A` | small trainable down projection |
| `B` | small trainable up projection |
| `r` | rank, usually 4/8/16 |
| `alpha` | scaling value |

The trainable update is:

\[
\Delta W = BA
\]

If `W` is huge, `A` and `B` are much smaller.

---

## 4. APEX-1 Implementation

APEX-1 implements LoRA from scratch in:

```txt
apex/model/lora.py
```

Core class:

```python
LoRALinear(base_layer, r=8, alpha=16, dropout=0.05)
```

It wraps any `nn.Linear` module.

Before:

```txt
W_Q: Linear(d_model -> n_heads * d_head)
```

After:

```txt
W_Q: LoRALinear(
  base_layer = frozen Linear(...)
  lora_A     = trainable Linear(in_features -> r)
  lora_B     = trainable Linear(r -> out_features)
)
```

---

## 5. Target Modules

The default config applies LoRA to:

```yaml
target_modules:
  - W_Q
  - W_K
  - W_V
  - W_O
  - W_DKV
  - W_UK
  - W_UV
  - W_DQ
  - W_UQ
  - W_KR
  - W_QR
  - W_gate
  - W_up
  - W_down
  - router
```

This covers:

- GQA attention
- MLA attention
- output projection
- SwiGLU FFN
- MoE routed experts
- MoE router

---

## 6. Config

Use:

```yaml
peft:
  enabled: true
  method: lora
  r: 4
  alpha: 8
  dropout: 0.0
  freeze_base_model: true
  target_modules:
    - W_Q
    - W_K
    - W_V
    - W_O
    - W_gate
    - W_up
    - W_down
  modules_to_save: []
  bias: none
```

For small CPU tests:

```bash
python examples/lora_finetune_demo.py
```

For CLI fine-tuning:

```bash
python scripts/finetune_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir checkpoints/lora \
  --max-steps 20
```

Dry run:

```bash
python scripts/finetune_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --dry-run \
  --max-steps 5
```

---

## 7. Saving Adapters

LoRA saves only adapter weights:

```python
from apex.model.lora import save_lora_adapters

save_lora_adapters(model, "checkpoints/lora/adapter_final.pt", config.peft)
```

This checkpoint is much smaller than the full model checkpoint.

---

## 8. Loading Adapters

Create the same base model with PEFT enabled, then load:

```python
from apex.model.lora import load_lora_adapters

load_lora_adapters(model, "checkpoints/lora/adapter_final.pt")
```

---

## 9. Merging for Inference

You can merge LoRA into base weights:

```python
from apex.model.lora import merge_lora_weights

merge_lora_weights(model)
```

After merging:

\[
W_{\text{merged}} = W + \frac{\alpha}{r}BA
\]

This removes adapter overhead during inference.

---

## 10. Why This Matters

Without PEFT, fine-tuning a large model means storing optimizer state and gradients for almost everything.

With LoRA:

- base weights are frozen
- only low-rank matrices train
- adapter checkpoint is small
- you can train multiple personalities/domains using separate adapters

This makes APEX-1 much more practical for learners and solo developers.
