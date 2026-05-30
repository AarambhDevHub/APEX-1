# 33 — Evaluation, Benchmarking, and Model Inspection

APEX-1 v2.4.0 adds the tools learners need after they build a model:

1. evaluate it
2. benchmark it
3. inspect it
4. visualize its architecture
5. run tiny dataset examples

Building a model is only half of AI engineering. The next question is:

> How do I know the model is working?

This lesson answers that question with small, readable utilities.

---

## 1. Model Evaluation System

The new `apex/eval/` package includes:

```txt
apex/eval/
├── __init__.py
├── metrics.py
├── perplexity.py
├── generation_quality.py
├── vision_eval.py
└── benchmark.py
```

### Perplexity

Perplexity measures how surprised the model is by the next token.

Lower perplexity usually means the model assigns higher probability to the
correct continuation.

```python
from apex.eval import compute_perplexity

result = compute_perplexity(model, dataloader)
print(result.loss)
print(result.perplexity)
```

### Token Accuracy

Accuracy checks whether the highest-logit token matches the target token.

```python
from apex.eval import next_token_accuracy

acc = next_token_accuracy(logits[:, :-1], input_ids[:, 1:])
print(acc.accuracy)
```

### Generation Quality

The generation quality helpers compute simple text statistics:

- average length
- distinct-1
- distinct-2
- repetition rate

These are not final quality metrics, but they are useful smoke checks.

---

## 2. Tiny Benchmark CLI

Run:

```bash
python scripts/benchmark.py --batch-size 1 --seq-len 16 --repeats 5
```

Vision benchmark:

```bash
python scripts/benchmark.py --vision --batch-size 1 --seq-len 16 --repeats 5
```

Example output:

```txt
| Metric | Value |
|---|---:|
| Device | cpu |
| Batch size | 1 |
| Sequence length | 16 |
| Repeats | 5 |
| Mean forward time | 42.000 ms |
| Tokens / second | 380.95 |
```

This benchmark is intentionally tiny. It helps learners compare model configs on
normal machines without requiring a GPU.

---

## 3. Model Inspector

Run:

```bash
python scripts/inspect_model.py
```

Vision model inspection:

```bash
python scripts/inspect_model.py --vision
```

The inspector prints:

- total parameters
- trainable parameters
- active parameters
- global MLA layer count
- local GQA+SW layer count
- MoE layer count
- dense FFN layer count
- skip-gate layer count
- vision parameter count
- per-layer map

This is useful because learners can see the architecture without reading every
source file first.

---

## 4. Architecture Diagram Generator

Run:

```bash
python scripts/print_architecture.py
```

Vision diagram:

```bash
python scripts/print_architecture.py --vision
```

Markdown layer table:

```bash
python scripts/print_architecture.py --table
```

The diagram explains the full route from input to logits:

```txt
Text Input
  └─ Token Embedding × √d
      └─ Transformer Blocks
          ├─ Layer 00: Local GQA+SW + Dense FFN + SkipGate
          ├─ Layer 01: Local GQA+SW + MoE FFN + SkipGate
          └─ Layer 05: Global MLA + MoE FFN + SkipGate
              └─ Final RMSNorm
                  └─ Tied LM Head → logits
```

For vision configs, it also shows:

```txt
Image Input
  └─ Vision Encoder
      └─ Vision Projector
          └─ Insert at <|img|> inside token embedding stream
```

---

## 5. Mini Dataset Examples

New sample files:

```txt
data/samples/
├── tiny_text.jsonl
├── tiny_sft.jsonl
├── tiny_preference.jsonl
└── tiny_vision.jsonl
```

These are not real datasets. They are small examples that teach the expected
format.

Run:

```bash
python examples/tiny_dataset_demo.py
```

---

## 6. More Examples

New examples:

```txt
examples/eval_demo.py
examples/benchmark_demo.py
examples/inspect_model_demo.py
examples/architecture_diagram_demo.py
examples/tiny_dataset_demo.py
```

Recommended order:

```bash
python examples/inspect_model_demo.py
python examples/architecture_diagram_demo.py
python examples/eval_demo.py
python examples/benchmark_demo.py
python examples/tiny_dataset_demo.py
```

---

## Why This Release Matters

APEX-1 already teaches how to build the model.

v2.4.0 teaches how to ask engineering questions about the model:

```txt
How many parameters does it have?
Which layers are global?
Which layers use MoE?
How fast is the forward pass?
What is the output shape?
Can I compute perplexity?
Can I validate vision outputs?
```

This is the bridge from model architecture to model engineering.
