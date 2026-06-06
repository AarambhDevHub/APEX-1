# 36 — DoRA: Weight-Decomposed LoRA Fine-Tuning

APEX-1 v2.8.0 adds an educational implementation of **DoRA**: Weight-Decomposed Low-Rank Adaptation.

LoRA updates a frozen weight matrix with a low-rank delta:

```txt
W' = W + ΔW
ΔW = B × A × (alpha / r)
```

DoRA keeps the low-rank update, but changes the way the final adapted weight is represented:

```txt
W_dora = m × normalize(W + ΔW)
```

Where:

- `W` is the frozen base weight
- `ΔW` is the low-rank LoRA update
- `normalize(...)` gives the adapted direction
- `m` is a trainable magnitude vector

This separates **direction learning** from **magnitude learning**.

---

## Why DoRA Matters

LoRA is efficient, but it only changes the weight through a low-rank delta. Full fine-tuning can change both the direction and the scale of each weight vector. DoRA tries to close that gap by making the magnitude explicit and trainable while still using low-rank adapters for directional changes.

In APEX-1, DoRA is implemented for learning clarity:

```txt
frozen base Linear
+ trainable LoRA A/B matrices
+ trainable per-output DoRA magnitude
```

APEX-1 also includes `qdora`, which combines the educational 4-bit frozen base from QLoRA with DoRA's magnitude/direction adapter.

---

## Configuration

```yaml
peft:
  enabled: true
  method: dora
  r: 4
  alpha: 8
  dropout: 0.0
  freeze_base_model: true
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
  modules_to_save: []
  bias: none
```

For quantized DoRA experiments:

```yaml
peft:
  enabled: true
  method: qdora
  quantization_bits: 4
  quant_type: nf4
  double_quant: true
  compute_dtype: float32
```

---

## Run the CPU Demo

```bash
python examples/dora_finetune_demo.py
```

This verifies:

- DoRA adapter injection
- frozen base weights
- trainable LoRA direction matrices
- trainable DoRA magnitude vectors
- adapter-only save/load
- merge and unload into a plain model
- plain model checkpoint compatibility

---

## Fine-Tune with DoRA

```bash
python scripts/finetune_dora.py \
  --config configs/apex1_tiny_dora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir outputs/dora-test \
  --max-steps 10
```

Dry run with synthetic data:

```bash
python scripts/finetune_dora.py \
  --config configs/apex1_tiny_dora.yaml \
  --dry-run \
  --max-steps 5
```

Run QDoRA:

```bash
python scripts/finetune_dora.py \
  --config configs/apex1_tiny_qdora.yaml \
  --method qdora \
  --dry-run \
  --max-steps 5
```

---

## What Gets Trained?

For DoRA, APEX-1 trains only:

```txt
lora_A.weight
lora_B.weight
dora_magnitude
```

The base model stays frozen.

---

## Merge and Export

DoRA adapters can be merged and unloaded exactly like LoRA:

```python
from apex.model.lora import merge_and_unload_lora_weights

merge_and_unload_lora_weights(model)
```

After this, the model no longer contains PEFT wrappers and its state dict can be loaded into a plain APEX-1 model.

---

## Tests

```bash
pytest tests/test_dora.py -v
```

Recommended full PEFT suite:

```bash
pytest tests/test_lora_peft.py -v
pytest tests/test_lora_inference.py -v
pytest tests/test_qlora.py -v
pytest tests/test_dora.py -v
```
