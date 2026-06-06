# 37 — Adapter-DPO Alignment

APEX-1 v2.9.0 adds adapter-based Direct Preference Optimization.

Previous releases gave us:

```txt
v2.5.0  LoRA training
v2.6.0  adapter inference and merge/export
v2.7.0  QLoRA 4-bit PEFT
v2.8.0  DoRA / QDoRA
v2.9.0  adapter-DPO alignment
```

## Plain-English Definition

DPO trains a model from preference pairs:

```txt
prompt
chosen response
rejected response
```

The model is rewarded when it gives higher probability to the chosen response
than the rejected response.

In APEX-1 v2.9.0, we do not update the whole model. We update only PEFT adapter
parameters.

## Why Adapter-DPO?

Full DPO updates all model weights. That is expensive.

Adapter-DPO does this instead:

```txt
frozen base model + trainable adapter + frozen reference model
```

This makes preference alignment much cheaper and easier to test on a small machine.

## DPO Loss

For a prompt `x`, chosen response `y_w`, and rejected response `y_l`:

```txt
log_ratio_chosen   = log πθ(y_w | x) - log πref(y_w | x)
log_ratio_rejected = log πθ(y_l | x) - log πref(y_l | x)

loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
```

Where:

- `πθ` is the trainable policy model with adapters.
- `πref` is the frozen reference model.
- `β` controls how conservative the update is.

## Files Added

```txt
apex/alignment/adapter_dpo.py
scripts/finetune_adapter_dpo.py
examples/adapter_dpo_demo.py
tests/test_adapter_dpo.py
configs/apex1_tiny_lora_dpo.yaml
configs/apex1_tiny_qlora_dpo.yaml
configs/apex1_tiny_dora_dpo.yaml
configs/apex1_tiny_qdora_dpo.yaml
data/samples/tiny_preference.jsonl
```

## Run Demo

```bash
python examples/adapter_dpo_demo.py
```

## Train Tiny Adapter-DPO

```bash
python scripts/finetune_adapter_dpo.py \
  --config configs/apex1_tiny_lora_dpo.yaml \
  --data data/samples/tiny_preference.jsonl \
  --output-dir outputs/adapter-dpo-test \
  --max-steps 10
```

## Train DoRA-DPO

```bash
python scripts/finetune_adapter_dpo.py \
  --config configs/apex1_tiny_dora_dpo.yaml \
  --method dora \
  --data data/samples/tiny_preference.jsonl \
  --output-dir outputs/dora-dpo-test \
  --max-steps 10
```

## What Gets Saved?

The trainer saves adapter-only checkpoints:

```txt
outputs/adapter-dpo-test/
└── adapter_final.pt
```

The base model is not saved inside the adapter checkpoint.

## Important Note

APEX-1 still does not ship with a large pretrained checkpoint. Tiny random models
will not produce high-quality aligned text. This release teaches the DPO
mechanics and verifies the adapter-only training path.
