# 34 — LoRA Adapter Inference & Merge

APEX-1 v2.5.0 taught how to train LoRA adapters. APEX-1 v2.6.0 completes the workflow by showing how to **load**, **generate with**, **merge**, and **export** those adapters.

This lesson answers the practical question every PEFT system must solve:

> After I fine-tune a small adapter, how do I actually use it?

---

## 1. Plain-English Definition

A LoRA adapter is a small checkpoint that stores only the learned low-rank update matrices.

During training, the base model is frozen and only these adapter matrices are updated:

```txt
W' = W + ΔW
ΔW = B × A × (alpha / r)
```

For inference, there are two common choices:

1. **Load the adapter dynamically** — keep the base model frozen and add the LoRA branch during forward passes.
2. **Merge the adapter** — add `ΔW` directly into the base weight `W`, then use the merged model like a normal model.

APEX-1 v2.6.0 supports both.

---

## 2. Real-World Analogy

Think of the base model as a large textbook.

A LoRA adapter is like a small sticky-note pack that adds corrections for one topic.

You can use the textbook with sticky notes attached, or you can rewrite the corrections directly into a new printed copy.

| Mode | Analogy | Code Path |
|---|---|---|
| Dynamic adapter | textbook + sticky notes | `generate_with_lora.py` |
| Merged model | new printed textbook | `merge_lora.py` |

---

## 3. Why Loading Order Matters

A plain APEX checkpoint has keys like:

```txt
blocks.0.attn.W_Q.weight
blocks.0.attn.W_Q.bias
```

A LoRA-wrapped model has keys like:

```txt
blocks.0.attn.W_Q.base_layer.weight
blocks.0.attn.W_Q.lora_A.weight
blocks.0.attn.W_Q.lora_B.weight
```

So if we build the LoRA model first and then load a plain base checkpoint, the names do not match.

Correct order:

```txt
1. Disable PEFT
2. Build plain APEX model
3. Load base checkpoint
4. Enable PEFT
5. Inject LoRA wrappers
6. Load adapter checkpoint
```

This logic lives in:

```py
apex/model/lora_inference.py
```

---

## 4. Generate With a LoRA Adapter

After training v2.5.0:

```bash
python scripts/finetune_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir outputs/lora-test \
  --max-steps 10
```

You get:

```txt
outputs/lora-test/adapter_final.pt
```

Now generate with it:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
```

With a trained base checkpoint:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --checkpoint checkpoints/base.pt \
  --adapter outputs/lora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
```

---

## 5. Merge Adapter Into Base Weights

For deployment-style experiments, merge the adapter:

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

With a trained base checkpoint:

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --checkpoint checkpoints/base.pt \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

The default merge path does two things:

```txt
1. merge LoRA delta into base Linear weights
2. remove LoRA wrapper modules
```

After that, the checkpoint is a plain APEX checkpoint and can be used by normal generation:

```bash
python scripts/generate.py \
  --config configs/apex1_tiny.yaml \
  --checkpoint outputs/merged-apex-lora.pt \
  --prompt "Hello"
```

---

## 6. Runtime Merge vs Export Merge

APEX-1 now supports two merge styles.

### Runtime merge

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --merge-before-generate \
  --prompt "Hello"
```

This keeps LoRA wrapper modules attached, but marks them as merged so the forward pass skips the adapter branch.

### Export merge

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

This removes LoRA wrappers completely and saves a plain model checkpoint.

---

## 7. Code Walkthrough

### `load_lora_model_for_inference`

```py
result = load_lora_model_for_inference(
    config=config,
    adapter="outputs/lora-test/adapter_final.pt",
    checkpoint="checkpoints/base.pt",
    device="cpu",
)
```

Returns:

```py
LoRAInferenceLoadResult(
    model=model,
    adapter_info=adapter_info,
    device=device,
    lora_modules=85,
    merged_for_runtime=False,
    unloaded=False,
)
```

### `merge_and_unload_lora_weights`

```py
merge_and_unload_lora_weights(model)
```

This replaces:

```py
LoRALinear(base_layer + lora_A + lora_B)
```

with:

```py
nn.Linear(merged_weight)
```

So the model becomes plain again.

---

## 8. Test It

Run:

```bash
pytest tests/test_lora_inference.py -v
```

Expected:

```txt
5 passed
```

Run demo:

```bash
python examples/lora_generation_demo.py
```

Expected behavior:

```txt
Saved adapter: .../adapter_final.pt
Inference LoRA modules: 85
Merged for runtime: True
Generated token ids: [...]
Merged checkpoint loaded into plain APEX-1 model successfully
```

---

## 9. What v2.6.0 Adds

| Feature | Status |
|---|---|
| Generate with saved LoRA adapter | ✅ Complete |
| Safe base checkpoint + adapter load order | ✅ Complete |
| Runtime merge before generation | ✅ Complete |
| Merge and unload LoRA wrappers | ✅ Complete |
| Save plain merged APEX checkpoint | ✅ Complete |
| CPU smoke demo | ✅ Complete |
| Unit tests for adapter inference and merge | ✅ Complete |

---

## 10. Important Limitation

APEX-1 does not ship with a large pretrained checkpoint. If you fine-tune a randomly initialized tiny model, the adapter workflow will work, but the text quality will not be meaningful.

That is expected.

The educational goal of v2.6.0 is to teach the full engineering workflow:

```txt
train adapter -> save adapter -> load adapter -> generate -> merge -> export
```

Once APEX-1 has a real trained base checkpoint, the same workflow becomes directly useful for real domain adaptation.
