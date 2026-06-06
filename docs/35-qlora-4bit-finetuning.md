# 35 — QLoRA 4-bit Fine-Tuning

APEX-1 v2.7.0 adds an educational QLoRA-style workflow.

The goal is to teach how low-memory adapter fine-tuning works:

```txt
frozen full-precision base + LoRA adapters        # v2.5.0 LoRA
frozen 4-bit quantized base + LoRA adapters       # v2.7.0 QLoRA-style path
```

This implementation is written directly in PyTorch so every step is readable.
It is CPU-friendly for tiny models and does not require external PEFT or
bitsandbytes packages.

---

## 1. What Problem Does QLoRA Solve?

Full fine-tuning updates every weight in the model. LoRA reduces training cost by
freezing the base model and training only small low-rank matrices.

QLoRA goes one step further:

1. keep the base model frozen
2. store the frozen base weights in 4-bit form
3. dequantize the base weight during forward pass
4. train only LoRA adapter matrices

This reduces memory used by the frozen base model while preserving the adapter
fine-tuning workflow.

---

## 2. The Core Formula

Classic LoRA computes:

```txt
y = xW^T + xA^T B^T * (alpha / r)
```

QLoRA-style training changes only how `W` is stored:

```txt
W_float32 -> quantize_4bit(W) -> W_4bit
W_dequant = dequantize(W_4bit)
y = xW_dequant^T + xA^T B^T * (alpha / r)
```

The base weight is frozen. Gradients flow into `A` and `B`, not into `W_4bit`.

---

## 3. NF4-style Quantization

APEX-1 v2.7.0 uses an educational 16-value NF4-style codebook.

Instead of evenly spacing all 16 values, NF4-style codebooks place more values
near zero because neural-network weights are often concentrated near zero.

The flow is:

```txt
weight row -> row absmax scale -> normalized values in [-1, 1]
normalized values -> nearest codebook index in [0, 15]
indices -> packed uint8 storage
```

Each stored index needs 4 bits. Two indices fit into one `uint8` byte.

---

## 4. Double Quantization

Normal 4-bit quantization still needs scale values. For each output row, APEX-1
stores an absmax scale.

Double quantization compresses those scales too:

```txt
scale_float32 -> scale_uint8 + global_scale
```

This is not a CUDA-optimized implementation. It is a readable educational
version that demonstrates the same idea.

---

## 5. Main Classes

### `QuantizedLinear4bit`

Located in:

```txt
apex/model/lora.py
```

It replaces a frozen `nn.Linear` base layer with:

```txt
qweight      # packed 4-bit indices
scale_q      # optional uint8 row scales
scale_scale  # scale used to recover row scales
bias         # frozen bias, if present
```

During forward pass:

```python
weight = dequantize_4bit_weight(...)
return F.linear(x, weight, bias)
```

### `QLoRALinear`

`QLoRALinear` combines:

```txt
QuantizedLinear4bit base layer
LoRA A matrix
LoRA B matrix
```

Forward pass:

```python
result = quantized_base_layer(x)
update = lora_B(lora_A(dropout(x))) * scaling
return result + update
```

---

## 6. Configuration

Use:

```yaml
# configs/apex1_tiny_qlora.yaml
peft:
  enabled: true
  method: qlora
  r: 4
  alpha: 8
  dropout: 0.0
  freeze_base_model: true
  quantization_bits: 4
  quant_type: nf4
  double_quant: true
  compute_dtype: float32
```

Important fields:

| Field | Meaning |
|---|---|
| `method: qlora` | Use quantized frozen base + LoRA adapters |
| `quantization_bits: 4` | Use 4-bit base-weight storage |
| `quant_type: nf4` | Use NF4-style codebook |
| `double_quant: true` | Quantize row scales too |
| `compute_dtype: float32` | Dequantized compute dtype for CPU demos |

---

## 7. Run the Demo

```bash
python examples/qlora_finetune_demo.py
```

Expected behavior:

```txt
LoRA/QLoRA modules inserted: ...
QLoRA modules inserted: ...
QLoRA storage summary: ...
APEX-1 PEFT / LoRA / QLoRA Parameter Summary
One-step QLoRA loss: ...
Saved QLoRA adapter: ...
LoRA/QLoRA modules after merge+unload: 0
```

The generated adapter is still an adapter-only checkpoint. It does not store the
base model.

---

## 8. Fine-tune with CLI

```bash
python scripts/finetune_qlora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir outputs/qlora-test \
  --max-steps 20
```

Dry-run with random synthetic data:

```bash
python scripts/finetune_qlora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --dry-run \
  --max-steps 5
```

Output adapter:

```txt
outputs/qlora-test/adapter_final.pt
```

---

## 9. Generate with a QLoRA Adapter

The v2.6.0 generation CLI works with QLoRA configs too:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --adapter outputs/qlora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
```

Because APEX-1 does not ship with a pretrained base model, output quality will
look random until a real tokenizer and trained base checkpoint are used. That is
expected. The important test is that the adapter loads and generation runs.

---

## 10. Merge and Export

QLoRA merge/export does this:

```txt
4-bit quantized base -> dequantized float base
LoRA delta -> added into float base
QLoRA wrapper -> replaced with plain nn.Linear
checkpoint -> saved as normal APEX checkpoint
```

Command:

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --adapter outputs/qlora-test/adapter_final.pt \
  --output outputs/merged-apex-qlora.pt
```

Then run normal generation:

```bash
python scripts/generate.py \
  --config configs/apex1_tiny.yaml \
  --checkpoint outputs/merged-apex-qlora.pt \
  --prompt "Hello"
```

---

## 11. Tests

```bash
pytest tests/test_qlora.py -v
```

The tests verify:

- 4-bit pack/unpack roundtrip
- quantize/dequantize shape correctness
- quantized linear forward pass
- QLoRA module injection
- frozen quantized base + trainable adapters
- forward/backward pass
- adapter save/load
- merge/unload into plain model
- storage summary compression ratio

---

## 12. What This Does Not Implement Yet

This v2.7.0 implementation is educational. It does **not** include:

- bitsandbytes CUDA kernels
- true paged optimizers
- fused dequantization matmul kernels
- real large-model memory benchmark
- production-grade quantization calibration

Those are good future versions.

Recommended next releases:

```txt
v2.8.0 — DoRA adapter support
v2.9.0 — QLoRA + DPO alignment
v3.0.0 — real tokenizer + tiny pretrained checkpoint
```

---

## 13. Mental Model

Think of QLoRA like this:

```txt
Base model = big frozen knowledge storage
4-bit quantization = compressed frozen storage
LoRA adapters = small trainable behavior patch
Merge/export = bake the patch into a normal model checkpoint
```

APEX-1 v2.7.0 teaches that whole workflow in readable PyTorch.
