<div align="center">

# 🔺 APEX-1

### A Best-of-All-Worlds Large Language + Vision Model — v2.8.0

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Architecture%20Complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Core%20%2B%20Vision%20%2B%20LoRA-success.svg)]()
[![Docs](https://img.shields.io/badge/Docs-36%20Guides-orange.svg)](docs/)
[![Course](https://img.shields.io/badge/Course-Free%20%26%20Open-purple.svg)](docs/00-introduction.md)
[![PEFT](https://img.shields.io/badge/PEFT-DoRA%20Adapters-red.svg)](docs/36-dora-weight-decomposed-lora.md)

**Inspired by:** Claude · GPT-style systems · DeepSeek-V3/R1 · Qwen · Gemma · GLM · KIMI · MiniMax · Llama · LLaVA · Flamingo · ViT · LoRA/PEFT research

*Build a frontier-grade language + vision model from scratch. Understand every line. Fine-tune it efficiently. Train LoRA, QLoRA, and DoRA adapters, then load, merge, and export them.*

---

### 🆓 This Course Is Completely Free

Other LLM courses charge **$50–$500+** for content like this. APEX-1 is free and always will be — 36 lessons, 4 math references, 24 bug-fix engineering lessons, full annotated source code, CPU-friendly demos, vision architecture, LoRA/PEFT fine-tuning, adapter inference/merge workflows, QLoRA-style 4-bit fine-tuning, DoRA weight-decomposed adapters, and a growing test suite. No paywalls. No sign-ups. Just open source.

If this helped you learn, please consider supporting so we can keep building free education:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Free%20Education-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/aarambhdevhub)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor%20This%20Work-EA4AAA?style=for-the-badge&logo=github-sponsors&logoColor=white)](https://github.com/sponsors/aarambh-darshan)
[![Razorpay](https://img.shields.io/badge/Razorpay-Donate-02042B?style=for-the-badge&logo=razorpay&logoColor=white)](https://razorpay.me/@aarambhdevhub)

> *Every contribution — however small — directly funds more free content, more lessons, and more open-source AI education for everyone.*

</div>

---

## 🎓 What Is APEX-1?

APEX-1 is four things at once.

**As an architecture**, it is a production-grade decoder-only transformer that synthesizes strong ideas from modern AI systems into one coherent educational model — Multi-Head Latent Attention, Mixture of Experts routing, GQA, sliding-window attention, multi-token prediction, thinking mode, GRPO-style alignment, Constitutional AI, and more.

**As a course**, it is a complete beginner-to-expert curriculum for understanding how modern large language models actually work — not toy GPT-2 clones, but the real techniques behind frontier-style models. Every component is documented, every design decision is explained, and real engineering bugs are preserved as learning material.

**As a vision-language model preview**, APEX-1 accepts images through the existing `<|img|>` token. Images are encoded into visual tokens, projected into the APEX hidden space, and processed by the same decoder-only transformer context. This is the foundation for image captioning, visual question answering, screenshot understanding, chart understanding, and future multi-image reasoning.

**As a PEFT learning project**, APEX-1 now supports the complete LoRA lifecycle:

```txt
train adapter -> save adapter -> load adapter -> generate -> merge -> unload -> export
```

As of **v2.5.0**, APEX-1 includes LoRA/PEFT fine-tuning. As of **v2.6.0**, it also includes adapter inference and merge/export workflows. As of **v2.7.0**, APEX-1 includes an educational QLoRA-style path with 4-bit quantized frozen base weights plus trainable LoRA adapters. As of **v2.8.0**, it adds DoRA/QDoRA: trainable magnitude vectors plus low-rank direction updates. You can freeze the base model, train only adapter parameters, save adapter-only checkpoints, load adapters for generation, merge adapters into base weights, unload PEFT wrappers, and export plain APEX checkpoints for deployment-style experiments.

> **If you have ever wanted to understand what is really inside a modern LLM — not just the theory but the actual code — this is for you.**

---



## 🚀 What Is New in v2.8.0?

APEX-1 v2.8.0 adds **DoRA: Weight-Decomposed LoRA**.

This release extends the PEFT stack:

```txt
LoRA  -> low-rank adapter direction update
QLoRA -> 4-bit frozen base + LoRA adapters
DoRA  -> LoRA direction update + trainable weight magnitude
QDoRA -> 4-bit frozen base + DoRA magnitude/direction adapters
```

| Feature | Status |
|---|---|
| `method: dora` in `PEFTConfig` | ✅ Complete |
| `DoRALinear` adapter layer | ✅ Complete |
| Trainable `dora_magnitude` vector | ✅ Complete |
| Direction update through LoRA A/B matrices | ✅ Complete |
| Frozen base model training | ✅ Complete |
| Adapter-only save/load with DoRA magnitude | ✅ Complete |
| Merge and unload DoRA into plain `nn.Linear` | ✅ Complete |
| Optional `method: qdora` experiment | ✅ Complete |
| New CLI: `scripts/finetune_dora.py` | ✅ Complete |
| New demo: `examples/dora_finetune_demo.py` | ✅ Complete |
| New tests: `tests/test_dora.py` | ✅ Complete |
| New guide: `docs/36-dora-weight-decomposed-lora.md` | ✅ Complete |

### Why DoRA Matters

LoRA efficiently learns a low-rank update. DoRA goes one step closer to full fine-tuning by separating each adapted weight into:

```txt
magnitude × direction
```

The direction is updated with LoRA. The magnitude is a small trainable vector. This lets APEX-1 teach why adapter methods can behave more like full fine-tuning while still training only a small number of parameters.

---

## 🚀 What Is New in v2.7.0?

APEX-1 v2.7.0 adds **QLoRA-style 4-bit PEFT fine-tuning from scratch**.

This release teaches how adapter training can be combined with a quantized frozen base model:

```txt
float base Linear -> 4-bit quantized frozen base -> train LoRA adapters only
```

| Feature | Status |
|---|---|
| NF4-style 4-bit codebook quantization | ✅ Complete |
| Packed 4-bit weight storage | ✅ Complete |
| Optional double quantization of row scales | ✅ Complete |
| `QuantizedLinear4bit` frozen base layer | ✅ Complete |
| `QLoRALinear` adapter wrapper | ✅ Complete |
| Automatic QLoRA injection via `peft.method: qlora` | ✅ Complete |
| QLoRA adapter-only save/load | ✅ Complete |
| QLoRA merge + unload into plain checkpoint | ✅ Complete |
| New CLI: `scripts/finetune_qlora.py` | ✅ Complete |
| New demo: `examples/qlora_finetune_demo.py` | ✅ Complete |
| New config: `configs/apex1_tiny_qlora.yaml` | ✅ Complete |
| New tests: `tests/test_qlora.py` | ✅ Complete |
| New guide: `docs/35-qlora-4bit-finetuning.md` | ✅ Complete |
| CUDA NF4 kernels / paged optimizers | Future work |

### Why v2.7.0 Matters

LoRA reduces trainable parameters. QLoRA reduces base-model memory too by keeping the frozen base in 4-bit quantized form while training only adapters.

You can now run:

```bash
python examples/qlora_finetune_demo.py
```

Or fine-tune with a tiny CPU-friendly config:

```bash
python scripts/finetune_qlora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir outputs/qlora-test \
  --max-steps 10
```

Run the new tests:

```bash
pytest tests/test_qlora.py -v
pytest tests/test_dora.py -v
```

---
## 🚀 What Is New in v2.6.0?

APEX-1 v2.6.0 adds **LoRA adapter inference and merge/export workflows**.

This release completes the full PEFT lifecycle:

```txt
train adapter -> save adapter -> load adapter -> generate -> merge -> export
```

| Feature | Status |
|---|---|
| Generate with saved LoRA adapter | ✅ Complete |
| Safe base-checkpoint-first load order | ✅ Complete |
| Runtime merge before generation | ✅ Complete |
| Merge LoRA into base weights | ✅ Complete |
| Unload LoRA wrappers after merge | ✅ Complete |
| Save plain merged APEX checkpoint | ✅ Complete |
| New helper module: `apex/model/lora_inference.py` | ✅ Complete |
| New CLI: `scripts/generate_with_lora.py` | ✅ Complete |
| New CLI: `scripts/merge_lora.py` | ✅ Complete |
| New demo: `examples/lora_generation_demo.py` | ✅ Complete |
| New tests: `tests/test_lora_inference.py` | ✅ Complete |
| New guide: `docs/34-lora-inference-and-merge.md` | ✅ Complete |

### Why v2.6.0 Matters

v2.5.0 proved that APEX-1 can train LoRA adapters. v2.6.0 proves those adapters are useful after training.

You can now generate with a saved adapter:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
```

You can merge an adapter into a plain checkpoint:

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

Then use the merged checkpoint with normal generation:

```bash
python scripts/generate.py \
  --config configs/apex1_tiny.yaml \
  --checkpoint outputs/merged-apex-lora.pt \
  --prompt "Hello"
```

---

## 🚀 What Is New in v2.5.0?

APEX-1 v2.5.0 added **LoRA + PEFT fine-tuning from scratch**.

This release teaches how modern models are adapted without retraining every parameter.

| Feature | Status |
|---|---|
| `PEFTConfig` in `apex/config.py` | ✅ Complete |
| `LoRALinear` adapter layer | ✅ Complete |
| Automatic LoRA injection into target modules | ✅ Complete |
| Freeze-base-model training | ✅ Complete |
| Trainable parameter summaries | ✅ Complete |
| Adapter-only save/load | ✅ Complete |
| Merge and unmerge LoRA weights | ✅ Complete |
| CPU-friendly LoRA demo | ✅ Complete |
| LoRA SFT trainer | ✅ Complete |
| LoRA fine-tuning CLI | ✅ Complete |
| New docs lesson: `33-lora-peft-finetuning.md` | ✅ Complete |
| New tests: `tests/test_lora_peft.py` | ✅ Complete |

### Why LoRA Matters

Full fine-tuning updates every model parameter. That is expensive, memory-heavy, and hard to experiment with.

LoRA — **Low-Rank Adaptation** — freezes the original weight matrix and learns a small low-rank update:

```txt
W' = W + ΔW
ΔW = B × A × (alpha / r)
```

Instead of training a full matrix, LoRA trains only two smaller matrices:

```txt
A: [r, in_features]
B: [out_features, r]
```

This makes fine-tuning much cheaper while still allowing the model to adapt to a new task, dataset, style, or domain.

---

## 🧩 LoRA / PEFT Capability Status

APEX-1 includes an educational PEFT implementation built directly in PyTorch.

> **Important:** APEX-1 does not ship with a large pretrained checkpoint. The LoRA code is fully functional and testable, but meaningful fine-tuning quality requires a trained base checkpoint and real dataset. This is intentional: APEX-1 teaches the architecture and training mechanics from scratch.

| Capability | Status |
|---|---|
| Freeze base model | ✅ Complete |
| Inject adapters into `nn.Linear` modules | ✅ Complete |
| Target MLA, GQA, FFN, MoE, and router projections | ✅ Complete |
| Adapter-only checkpoint save/load | ✅ Complete |
| Train only LoRA parameters | ✅ Complete |
| Merge LoRA into base weights | ✅ Complete |
| Unmerge LoRA from base weights | ✅ Complete |
| CLI for SFT fine-tuning | ✅ Complete |
| CPU smoke demo | ✅ Complete |
| Generate with saved adapter | ✅ Complete |
| Merge and unload adapters into plain checkpoint | ✅ Complete |
| DoRA magnitude/direction adapters | ✅ Complete in v2.8.0 |
| QDoRA quantized DoRA experiment | ✅ Complete in v2.8.0 |
| QLoRA-style 4-bit quantized base + LoRA adapters | ✅ Complete |
| Real high-quality fine-tuning | Requires trained base checkpoint + dataset |

Default LoRA target modules include:

```txt
W_Q, W_K, W_V, W_O,
W_DKV, W_UK, W_UV,
W_DQ, W_UQ, W_KR, W_QR,
W_gate, W_up, W_down,
router
```

These cover APEX-1's attention projections, MLA projections, feed-forward projections, and MoE routing layers.

---

## 🖼️ Vision Capability Status

APEX-1 includes a complete **educational multimodal architecture**. It can load an image, preprocess it, encode it into patch features, project those features into APEX's language hidden space, replace the `<|img|>` placeholder with continuous visual tokens, and run image + text through the same decoder-only transformer.

This means the vision pipeline is architecturally complete and fully testable on CPU.

> **Important:** APEX-1 does **not** ship with pretrained vision weights. The model can process images, but real semantic image understanding requires training on image-caption or visual-instruction datasets. This is intentional: APEX-1 is a course project for learning the architecture from scratch, not a large pretrained multimodal checkpoint.

| Capability | Status |
|---|---|
| Image preprocessing | ✅ Complete |
| Native patch-based vision encoder | ✅ Complete |
| Vision-to-language projector | ✅ Complete |
| `<|img|>` placeholder replacement | ✅ Complete |
| Visual tokens inside APEX transformer | ✅ Complete |
| Vision SFT loss and label expansion | ✅ Complete |
| CPU tests and demos | ✅ Complete |
| Real image understanding | Requires training / pretrained weights |

---

## 🎯 Who Is This For?

| Background | What You Will Get |
|---|---|
| **CS / Engineering students** | A hands-on project that covers what university ML courses often skip |
| **Self-taught developers** | A structured path from "what is a token" to "how does PEFT fine-tuning work" |
| **ML practitioners** | Deep dives into MLA, MoE, speculative decoding, LoRA, QLoRA, adapter merge/export, and modern alignment techniques |
| **Researchers** | A reproducible educational reference architecture synthesizing modern LLM/VLM components |
| **YouTube / content learners** | 35 documentation files, each structured like a complete lesson |
| **Open-source builders** | A codebase designed for reading, modifying, testing, and teaching |

---

## 📚 The Curriculum — 35 Lessons

Every lesson follows the same five-step format:

> **Plain-English definition → Real-world analogy → LaTeX math → Full annotated source code → Design rationale**

### 🟢 Part 1 — Foundations

| Lesson | Topic | Key Concepts |
|---|---|---|
| [00](docs/00-introduction.md) | What Is a Language Model? | Tokens, loss, training loop |
| [01](docs/01-project-structure.md) | Project Structure | Every file explained, reading order |
| [02](docs/02-configuration.md) | Configuration System | Hyperparameters, YAML loading, validation |
| [03](docs/03-tokenizer.md) | Tokenizer | BPE algorithm, special tokens, SFT masking |

### 🔵 Part 2 — Building Blocks

| Lesson | Topic | Key Concepts |
|---|---|---|
| [04](docs/04-embeddings-and-rmsnorm.md) | Embeddings & RMSNorm | Weight tying, √d scaling, normalization math |
| [05](docs/05-positional-encoding-rope.md) | RoPE & YaRN | Rotation math, context extension |
| [06](docs/06-attention-masks.md) | Attention Masks | Prefix bidir, causal, sliding window |

### 🟣 Part 3 — Attention Mechanisms

| Lesson | Topic | Key Concepts |
|---|---|---|
| [07](docs/07-attention-mla.md) | Multi-Head Latent Attention | KV cache reduction, latent compression |
| [08](docs/08-attention-gqa.md) | GQA + Sliding Window | Group sharing, local/global ratio |

### 🟠 Part 4 — Feed-Forward Networks & Experts

| Lesson | Topic | Key Concepts |
|---|---|---|
| [09](docs/09-ffn-swiglu.md) | FFN & SwiGLU | Gating, activation design, 3-matrix FFN |
| [10](docs/10-mixture-of-experts.md) | Mixture of Experts | Hierarchical routing, sparse activation |
| [11](docs/11-skip-gate.md) | Dynamic Skip Gate | Conditional compute, STE binary threshold |
| [12](docs/12-load-balancer.md) | Auxiliary-Loss-Free Load Balancer | Expert collapse prevention |
| [13](docs/13-multi-token-head.md) | Multi-Token Prediction | Extra training signal, speculative decoding |

### 🔴 Part 5 — The Full Model

| Lesson | Topic | Key Concepts |
|---|---|---|
| [14](docs/14-transformer-block.md) | Transformer Block | Pre-norm, residuals, layer assignment |
| [15](docs/15-full-model.md) | Complete APEX-1 Model | End-to-end model assembly |

### 🟡 Part 6 — Training

| Lesson | Topic | Key Concepts |
|---|---|---|
| [16](docs/16-training-losses.md) | Training Losses | Cross-entropy, SFT masking, speculative loss |
| [17](docs/17-scheduler-and-optimizer.md) | Optimizer & LR Schedule | AdamW math, cosine warmup |
| [18](docs/18-training-pipeline.md) | Training Pipeline | Mixed precision, gradient accumulation |
| [19](docs/19-checkpointing.md) | Checkpointing | RNG state, resume training |
| [20](docs/20-datasets.md) | Datasets | Streaming, packing, padding masks |

### ⚪ Part 7 — Text Generation

| Lesson | Topic | Key Concepts |
|---|---|---|
| [21](docs/21-generation-sampling.md) | Sampling Strategies | KV cache, temperature, top-p, top-k |
| [22](docs/22-speculative-decoding.md) | Speculative Decoding | Draft-verify loop, probabilistic acceptance |
| [23](docs/23-thinking-mode.md) | Thinking Mode | CoT scratchpad, reasoning budget |

### 🟤 Part 8 — Alignment & Safety

| Lesson | Topic | Key Concepts |
|---|---|---|
| [24](docs/24-reward-model.md) | Reward Model | Bradley-Terry loss |
| [25](docs/25-dpo.md) | DPO | Implicit reward, preference optimization |
| [26](docs/26-grpo.md) | GRPO | RL without value function, group-relative rewards |
| [27](docs/27-process-reward-model.md) | Process Reward Model | Step-level rewards |
| [28](docs/28-constitutional-ai.md) | Constitutional AI | Critique-revision loop |
| [29](docs/29-combined-reward.md) | Combined Reward | Tri-signal reward formula |

### ⚫ Part 9 — Utilities, Walkthrough & Multimodal

| Lesson | Topic | Key Concepts |
|---|---|---|
| [30](docs/30-utilities.md) | Utilities | Shape checker, FLOPs, param counter |
| [31](docs/31-end-to-end-walkthrough.md) | End-to-End Walkthrough | Install → pretrain → SFT → generate |
| [32](docs/32-vision-capabilities.md) | Vision Capabilities | Image encoder, visual-token insertion, multimodal SFT |

### 🔴 Part 10 — Efficient Fine-Tuning

| Lesson | Topic | Key Concepts |
|---|---|---|
| [33](docs/33-lora-peft-finetuning.md) | LoRA & PEFT Fine-Tuning | Low-rank adapters, frozen base model, adapter checkpoints, merge/unmerge |
| [34](docs/34-lora-inference-and-merge.md) | LoRA Inference & Merge | Load adapters for generation, merge into base weights, unload wrappers, export plain checkpoints |
| [35](docs/35-qlora-4bit-finetuning.md) | QLoRA 4-bit Fine-Tuning | NF4-style quantization, double quantization, frozen 4-bit base weights, trainable adapters |

---

## 📐 Mathematical Reference

Four companion reference documents cover every formula used in APEX-1 with full derivations:

| Part | Topics | Formulas |
|---|---|---|
| [Part 1](docs/APEX-1-Mathematical-Reference-Part1.md) | Embedding, RMSNorm, RoPE, YaRN | F1–F8 |
| [Part 2](docs/APEX-1-Mathematical-Reference-Part2.md) | SDPA, MHA, GQA, MLA, Sliding Window, Masks | F9–F15 |
| [Part 3](docs/APEX-1-Mathematical-Reference-Part3.md) | SwiGLU, MoE, Load Balancing, Skip Gate, Multi-Token | F16–F21 |
| [Part 4](docs/APEX-1-Mathematical-Reference-Part4.md) | AdamW, LR Schedule, DPO, GRPO, Sampling, Full Pipeline | F22–F34 |

**34 formulas. 4 parts. Every derivation explained step by step.**

---

## 🐛 The Bug-Fix Pedagogy

APEX-1 contains **24 documented bugs** — found, fixed, and explained in detail. This is intentional.

Real engineering is not writing perfect code. It is finding subtle shape mismatches, off-by-one errors in loss computation, and silent incorrect behavior in KV caches. Each bug in APEX-1 comes with:

- What the original code did
- Why it was wrong
- The exact failure mode
- The fix and why it works
- A regression test to prevent recurrence

This is what most courses skip and what real ML engineers spend much of their time doing.

| Bug | File | What Was Wrong |
|---|---|---|
| BUG-01 | `attention.py` | MLA K_rope cache was always zeros — corrupting all autoregressive steps |
| BUG-02 | `attention.py` | W_O had wrong input dimension — crashed every forward pass |
| BUG-03 | `constitutional.py` | Critique always returned `violated=False` — safety was a no-op |
| BUG-04 | `grpo.py` | Generation loop reset logits every step — never produced real responses |
| BUG-05 | `reward_model.py` | `Optional` imported after the class that used it — NameError on load |
| BUG-06 | `prm.py` | `None` tokenizer caused cryptic AttributeError instead of clear message |
| BUG-07 | `apex_model.py` | Wrong RoPE cache passed to MLA layers — shape mismatch |
| BUG-08 | `ffn.py` | MoE dispatch silently wrong when multiple tokens routed to same expert |
| BUG-09 | `generator.py` | KV cache position detection used `isinstance` — fragile and wrong |
| BUG-10 | `mask.py` | Sliding window mask used Python loop — slow at long context |
| BUG-11 | `trainer.py` | Load balancer used global config `n_experts`, not per-layer actual count |
| BUG-12 | `losses.py` | Short-sequence speculative loss produced `nan` — silent training corruption |
| BUG-13 | `checkpoint.py` | Python RNG state saved as PyTorch tensor — non-reproducible resume |
| BUG-14 | `tokenizer.py` | Thinking tokens inherited wrong type — excluded from SFT loss |
| BUG-15 | `generator.py` | Speculative acceptance was greedy argmax — biased output distribution |
| BUG-16 | `dpo.py` | Prompt processed causally in DPO — weaker context representation |
| BUG-17 | `flops.py` | SwiGLU elementwise multiply missing from FLOPs estimate |
| BUG-18 | `config.py` | `d_model` mismatch logged as warning, not error — silent model corruption |
| BUG-19 | `block.py` | `is_moe` flag ignored `config.moe.enabled` — wrong FFN type |
| BUG-20 | `train.py` | Log file written to CWD — failed in read-only environments |
| BUG-21 | `generator.py` | Thinking start token consumed 1 budget slot |
| BUG-22 | `rope.py` | YaRN scaling used Python loop over `d_head` — slow for large models |
| BUG-23 | `shape_checker.py` | Always created a new model instead of using the provided one |
| BUG-24 | `dataset.py` | Padding tokens included in training loss — corrupted pretraining signal |

---

## 🏗️ Architecture

APEX-1 picks strong ideas from modern LLM and VLM systems and turns them into a readable educational implementation.

| Feature | Source / Inspiration | Why It Matters |
|---|---|---|
| Large vocabulary | Qwen-style tokenizer design | Better multilingual and code coverage |
| RoPE + YaRN extension | KIMI / DeepSeek-style context extension | Longer context without rewriting attention |
| Multi-Head Latent Attention | DeepSeek-V3-style MLA | Smaller KV cache |
| GQA + Sliding Window | Llama / Mistral-style efficient attention | Fast local attention |
| Interleaved local/global attention | Gemma-style pattern | Long-context efficiency |
| Prefix bidirectional attention | GLM-style prompting | Full context over system prefix |
| SwiGLU activation | PaLM / Llama-style FFN | Stronger nonlinear representation |
| Hierarchical MoE | DeepSeek / MiniMax-style sparse experts | More capacity with less active compute |
| Auxiliary-loss-free load balancing | DeepSeek-style balancing | Stable expert use without LM-loss interference |
| Dynamic skip gate | Conditional-compute research | Save FFN compute on easy tokens |
| Multi-token prediction | DeepSeek-style training signal | Richer next-token supervision |
| Thinking mode | Reasoning-model style scratchpad | Controlled reasoning budget |
| GRPO alignment | DeepSeek-R1-style RL | Group-relative reward optimization |
| Constitutional AI | Anthropic-style safety process | Critique and revision pipeline |
| Visual token bridge | LLaVA / Flamingo / Qwen-VL style | Images become language-context tokens |
| Native ViT encoder | Vision Transformer | From-scratch image patch encoder |
| Perceiver resampler | Flamingo-style compression | Fixed visual token budget |
| LoRA adapters | LoRA / PEFT research | Efficient task adaptation without full fine-tuning |
| Adapter checkpoints | PEFT workflow | Save small trainable deltas instead of full model copies |
| Adapter merge/export | Deployment-style PEFT workflow | Convert adapter form into plain model checkpoints |
| QLoRA 4-bit adapters | QLoRA research | Keep frozen base weights quantized while training adapters |

```txt
Text tokens [batch, seq_len]
        │
        ▼
Image pixels [batch, 3, H, W]        Optional LoRA/PEFT adapters
        │                                      │
        ▼                                      ▼
┌─────────────────────┐              ┌──────────────────────┐
│ Vision Encoder      │              │ LoRA wrapped Linear  │
│ ViT patch features  │              │ W' = W + BA(alpha/r) │
└─────────┬───────────┘              └──────────────────────┘
          ▼
┌─────────────────────┐
│ Vision Projector    │  Perceiver/MLP → d_model visual tokens
└─────────┬───────────┘
          ▼
Inserted at <|img|> inside token embeddings
                │
                ▼
┌─────────────────────┐
│  Embedding × √d     │  Weight-tied with LM head
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│         × n_layers Transformer Blocks        │
│                                              │
│  ┌─────────┐    ┌──────────────────────┐     │
│  │ RMSNorm │───►│ Attention            │     │
│  └─────────┘    │  MLA / GQA+SW        │     │
│                 │  + optional LoRA     │     │
│                 └──────────┬───────────┘     │
│                    + residual                │
│                            │                 │
│  ┌─────────┐    ┌─────────▼──────────┐      │
│  │ Skip    │───►│ FFN / MoE          │      │
│  │ Gate    │    │ + optional LoRA    │      │
│  └─────────┘    └──────────┬─────────┘      │
│                    + residual (gated)        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   RMSNorm     │
              │   LM Head     │  → logits [batch, seq, vocab]
              │   Spec Heads  │  → speculative predictions
              └───────────────┘
```

---

## 📊 Model Sizes

| Parameter | Tiny | Small | Medium | Large |
|---|---:|---:|---:|---:|
| `d_model` | 64 | 512 | 2,048 | 7,168 |
| `n_layers` | 6 | 12 | 36 | 72 |
| `n_heads_q` | 4 | 8 | 16 | 128 |
| `n_experts` | 4 | 8 | 64 | 256 |
| `max_seq_len` | 256 | 8K | 64K | 128K |
| Total params | ~1M | ~100M | ~7B | ~900B |
| Active params | tiny CPU | ~40M | ~2B | ~45B |

### Vision Configuration

| Vision Parameter | Tiny Vision Default | Why |
|---|---:|---|
| `image_size` | 64 | CPU-friendly demos |
| `patch_size` | 16 | 4×4 image patches |
| `vision_hidden_dim` | 64 | Matches tiny model scale |
| `num_visual_tokens` | 4 | Fixed visual-token budget |
| Projector | Perceiver / MLP | Compress image features into language tokens |

### LoRA Configuration

| LoRA Parameter | Tiny LoRA Default | Why |
|---|---:|---|
| `r` | 4 | Small low-rank adapter for CPU demos |
| `alpha` | 8 | Common scaling ratio |
| `dropout` | 0.0 | Deterministic smoke tests |
| `freeze_base_model` | `true` | Train only adapter parameters |
| `bias` | `none` | Simple PEFT baseline |
| `modules_to_save` | `[]` | Adapter-only checkpoint by default |

Start with **APEX-1-Tiny** (`configs/apex1_tiny.yaml`) — ~1M params, runs on CPU in seconds.

For multimodal lessons, start with **APEX-1-Tiny-Vision** (`configs/apex1_tiny_vision.yaml`).

For PEFT lessons, start with **APEX-1-Tiny-LoRA** (`configs/apex1_tiny_lora.yaml`).

For LoRA inference/merge lessons, use **APEX-1-Tiny-LoRA-Inference** (`configs/apex1_tiny_lora_inference.yaml`) or the normal LoRA config.

For QLoRA lessons, use **APEX-1-Tiny-QLoRA** (`configs/apex1_tiny_qlora.yaml`).

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/AarambhDevHub/APEX-1.git
cd APEX-1

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Run a text forward pass
python examples/forward_pass_demo.py

# Run text generation
python examples/generation_demo.py

# Try thinking mode
python examples/thinking_mode_demo.py

# Visualize attention masks
python examples/mask_visualization.py

# Try a vision forward pass
python examples/vision_forward_demo.py

# Try LoRA / PEFT fine-tuning smoke demo
python examples/lora_finetune_demo.py

# Try LoRA adapter generation smoke demo
python examples/lora_generation_demo.py

# Try QLoRA / 4-bit PEFT smoke demo
python examples/qlora_finetune_demo.py

# Try DoRA PEFT smoke demo
python examples/dora_finetune_demo.py

# Run LoRA tests
pytest tests/test_lora_peft.py -v
pytest tests/test_lora_inference.py -v
pytest tests/test_qlora.py -v
pytest tests/test_dora.py -v

# Run vision tests
pytest tests/test_vision.py -v

# Run the full suite
pytest tests/ -v
```

Expected vision demo output:

```txt
Input text tokens: (1, 7)
Visual tokens inserted: 4
Logits: (1, 10, 1000)
Hidden states: (1, 10, 64)
KV cache layers: 6
```

Expected LoRA fine-tune demo output:

```txt
LoRA modules inserted: ...
Trainable params: ...
Trainable percent: ...
Saved adapter: ...
```

Expected LoRA generation demo output:

```txt
Generated with adapter:
...
Merged checkpoint saved:
...
```

---

## 🧪 Evaluation, Benchmarking & Inspection — v2.4.0+

APEX-1 includes a small educational evaluation and benchmarking toolkit.

This helps learners answer practical model-engineering questions:

- How many parameters does my model have?
- Which layers use global MLA vs local GQA+SW?
- Which layers use MoE vs dense FFN?
- How fast is a tiny forward pass?
- What is the model's next-token perplexity?
- Are generated texts repetitive?
- Did the vision forward pass insert the expected number of visual tokens?
- How many parameters are trainable after LoRA injection?
- Does a saved LoRA adapter load and generate correctly?
- Does a merged LoRA checkpoint match adapter-form inference?
- How much storage does a QLoRA quantized base layer save?

Commands:

```bash
# Inspect parameters and layer types
python scripts/inspect_model.py
python scripts/inspect_model.py --vision

# Print an ASCII architecture diagram
python scripts/print_architecture.py
python scripts/print_architecture.py --vision
python scripts/print_architecture.py --table

# Run a tiny CPU benchmark
python scripts/benchmark.py --batch-size 1 --seq-len 16 --repeats 5
python scripts/benchmark.py --vision --batch-size 1 --seq-len 16 --repeats 5

# Run demos
python examples/eval_demo.py
python examples/benchmark_demo.py
python examples/inspect_model_demo.py
python examples/architecture_diagram_demo.py
python examples/tiny_dataset_demo.py

# Run evaluation and inspector tests
pytest tests/test_eval_and_inspector.py -v
```

---

## 🔧 LoRA / PEFT Fine-Tuning — v2.5.0

### 1. Use the LoRA config

```yaml
# configs/apex1_tiny_lora.yaml
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

### 2. Run the CPU LoRA demo

```bash
python examples/lora_finetune_demo.py
```

This verifies:

- base model construction
- LoRA adapter injection
- frozen base parameters
- trainable adapter parameters
- one optimization step
- adapter-only checkpoint save

### 3. Fine-tune with the CLI

```bash
python scripts/finetune_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir runs/lora_tiny \
  --max-steps 20
```

Optional arguments:

```bash
python scripts/finetune_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --tokenizer tokenizer.json \
  --checkpoint checkpoints/base_model.pt \
  --adapter adapters/domain_adapter.pt \
  --output-dir runs/lora_domain \
  --max-steps 100
```

### 4. Save only adapter weights

LoRA checkpoints are small because they save only adapter parameters:

```txt
runs/lora_tiny/
├── adapter_step_10.pt
├── adapter_step_20.pt
└── adapter_final.pt
```

### 5. Load adapter weights

```python
from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.lora import load_lora_adapters

config = APEXConfig.from_yaml("configs/apex1_tiny_lora.yaml")
model = APEX1Model(config)

load_lora_adapters(model, "runs/lora_tiny/adapter_final.pt")
```

### 6. Merge LoRA weights for deployment-style experiments

```python
from apex.model.lora import merge_lora_weights, unmerge_lora_weights

merge_lora_weights(model)
# run inference with merged weights

unmerge_lora_weights(model)
# return to adapter form
```

---

## 🔁 LoRA Inference & Merge — v2.6.0

### 1. Generate with a saved adapter

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
```

Optional base checkpoint:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --checkpoint checkpoints/base_model.pt \
  --adapter outputs/lora-test/adapter_final.pt \
  --tokenizer tokenizer.json \
  --prompt "Write a short Python function" \
  --max-tokens 128
```

### 2. Generate with runtime-merged adapter

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --merge \
  --prompt "Explain Rust ownership simply"
```

### 3. Merge adapter into a plain checkpoint

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

Optional base checkpoint:

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --checkpoint checkpoints/base_model.pt \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

### 4. Use merged checkpoint with normal generation

```bash
python scripts/generate.py \
  --config configs/apex1_tiny.yaml \
  --checkpoint outputs/merged-apex-lora.pt \
  --prompt "Hello"
```

### 5. Run v2.6.0 tests

```bash
pytest tests/test_lora_inference.py -v
```

---


## 🧊 QLoRA 4-bit Fine-Tuning — v2.7.0

### 1. Use the QLoRA config

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

### 2. Run the CPU QLoRA demo

```bash
python examples/qlora_finetune_demo.py
```

This verifies:

- base model construction
- 4-bit quantized base linear layers
- QLoRA adapter injection
- frozen quantized base parameters
- trainable adapter parameters
- one optimization step
- adapter-only checkpoint save
- merge + unload into plain model form

### 3. Fine-tune with the QLoRA CLI

```bash
python scripts/finetune_qlora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir outputs/qlora-test \
  --max-steps 20
```

Dry-run with synthetic data:

```bash
python scripts/finetune_qlora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --dry-run \
  --max-steps 5
```

### 4. Generate or merge with existing adapter tools

The v2.6.0 inference tools also work with QLoRA configs:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --adapter outputs/qlora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply"

python scripts/merge_lora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --adapter outputs/qlora-test/adapter_final.pt \
  --output outputs/merged-apex-qlora.pt
```

> **Educational note:** this implementation includes NF4-style quantization and double quantization, but not CUDA kernels or paged optimizers. It is designed for learning and CPU tests.

---
## 📁 Project Structure

```txt
APEX-1/
├── apex/
│   ├── __init__.py               # Package version and public exports
│   ├── config.py                 # Hyperparameters: text, training, alignment, vision, PEFT
│   │
│   ├── eval/
│   │   ├── metrics.py            # Token accuracy + token cross-entropy
│   │   ├── perplexity.py         # Next-token perplexity evaluation
│   │   ├── generation_quality.py # distinct-n, repetition, average length
│   │   ├── vision_eval.py        # Vision forward-output validation
│   │   └── benchmark.py          # Tiny forward-pass benchmark helper
│   │
│   ├── model/
│   │   ├── norm.py               # RMSNorm
│   │   ├── rope.py               # RoPE + YaRN
│   │   ├── mask.py               # Attention mask builder
│   │   ├── attention.py          # MLA + GQA+SW
│   │   ├── ffn.py                # DenseFFN + MoEFFN
│   │   ├── skip_gate.py          # Dynamic skip gate
│   │   ├── load_balancer.py      # Auxiliary-loss-free balancer
│   │   ├── multi_token_head.py   # Speculative prediction heads
│   │   ├── block.py              # One complete transformer block
│   │   ├── lora.py               # LoRA + QLoRA layers, 4-bit quantization, save/load, merge/unmerge
│   │   ├── lora_inference.py     # Load adapters, merge, unload, export plain checkpoints
│   │   ├── apex_model.py         # Complete text-only APEX-1 model + PEFT support
│   │   └── apex_vision_model.py  # Multimodal APEX-1 model: text + image tokens
│   │
│   ├── vision/
│   │   ├── __init__.py           # Vision module exports
│   │   ├── preprocess.py         # Image loading, resizing, normalization
│   │   ├── encoder.py            # Native patch-based vision encoder
│   │   └── projector.py          # Vision-to-language projector / visual token mapper
│   │
│   ├── tokenizer/
│   │   ├── tokenizer.py          # BPE tokenizer + <|img|> placeholder support
│   │   └── train_tokenizer.py    # Tokenizer training script
│   │
│   ├── generation/               # Sampling + generation engine
│   │
│   ├── training/
│   │   ├── losses.py             # Text pretraining + SFT losses
│   │   ├── vision_losses.py      # Vision SFT loss + visual-token label expansion
│   │   ├── trainer.py            # Base training loop
│   │   ├── peft.py               # PEFT / LoRA SFT trainer
│   │   ├── scheduler.py          # LR scheduler
│   │   └── checkpoint.py         # Save / load checkpoints
│   │
│   ├── alignment/                # Reward model, DPO, GRPO, PRM, CAI
│   │
│   ├── data/
│   │   ├── dataset.py            # Text dataset classes + DataLoader factories
│   │   └── vision_dataset.py     # Image-caption / vision-instruction dataset
│   │
│   └── utils/                    # Shape checker, FLOPs, param counter, architecture maps
│
├── configs/
│   ├── apex1_tiny.yaml                 # Tiny text-only config
│   ├── apex1_small.yaml                # Small text-only config
│   ├── apex1_medium.yaml               # Medium text-only config
│   ├── apex1_large.yaml                # Large text-only config
│   ├── apex1_tiny_vision.yaml          # Tiny multimodal config
│   ├── apex1_tiny_lora.yaml            # Tiny LoRA / PEFT config
│   ├── apex1_tiny_lora_inference.yaml  # Tiny LoRA inference/merge config
│   ├── apex1_tiny_qlora.yaml           # Tiny QLoRA / 4-bit PEFT config
│   └── apex1_tiny_qlora_inference.yaml # Tiny QLoRA inference/merge config
│
├── docs/
│   ├── 00-introduction.md
│   ├── ...
│   ├── 31-end-to-end-walkthrough.md
│   ├── 32-vision-capabilities.md
│   ├── 33-lora-peft-finetuning.md
│   ├── 34-lora-inference-and-merge.md
│   ├── 35-qlora-4bit-finetuning.md
│   ├── 36-dora-weight-decomposed-lora.md
│   └── APEX-1-Mathematical-Reference-Part*.md
│
├── tests/
│   ├── test_all.py               # Core unit tests
│   ├── test_bugfixes.py          # Regression tests for documented bugs
│   ├── test_vision.py            # Vision config, encoder, projector, model, loss tests
│   ├── test_eval_and_inspector.py# Evaluation, benchmark, inspector tests
│   ├── test_lora_peft.py         # LoRA / PEFT training tests
│   ├── test_lora_inference.py    # LoRA adapter inference + merge/export tests
│   └── test_qlora.py             # QLoRA 4-bit adapter tests
│
├── examples/
│   ├── forward_pass_demo.py      # Text forward-pass demo
│   ├── generation_demo.py        # Text generation demo
│   ├── thinking_mode_demo.py     # Thinking mode demo
│   ├── mask_visualization.py     # Attention mask visualization
│   ├── vision_forward_demo.py    # Image + text forward-pass demo
│   ├── vision_chat_demo.py       # Vision chat prompt demo
│   ├── eval_demo.py              # Evaluation demo
│   ├── benchmark_demo.py         # Benchmark demo
│   ├── inspect_model_demo.py     # Model inspector demo
│   ├── architecture_diagram_demo.py
│   ├── tiny_dataset_demo.py
│   ├── lora_finetune_demo.py     # LoRA / PEFT CPU smoke demo
│   ├── lora_generation_demo.py   # LoRA adapter generation + merge demo
│   └── qlora_finetune_demo.py    # QLoRA 4-bit PEFT CPU smoke demo
│
├── scripts/
│   ├── train.py                  # Text training CLI
│   ├── generate.py               # Text generation CLI
│   ├── benchmark.py              # CLI benchmark
│   ├── inspect_model.py          # CLI model inspector
│   ├── print_architecture.py     # CLI architecture diagram
│   ├── finetune_lora.py          # LoRA / PEFT fine-tuning CLI
│   ├── generate_with_lora.py     # Generate with saved LoRA adapter
│   ├── merge_lora.py             # Merge adapter and export plain checkpoint
│   └── finetune_qlora.py         # QLoRA / 4-bit PEFT fine-tuning CLI
│
├── data/samples/
│   ├── tiny_text.jsonl           # Text pretraining format example
│   ├── tiny_sft.jsonl            # Supervised fine-tuning format example
│   ├── tiny_preference.jsonl     # Preference data format example
│   └── tiny_vision.jsonl         # Vision instruction format example
│
├── README.md
├── CHANGELOG.md
├── pyproject.toml
└── LICENSE
```

---

## 🧪 What's New by Version


### v2.7.0 — QLoRA 4-bit PEFT Fine-Tuning

- **QLoRA-style adapters** — `QLoRALinear` wraps targeted projections with a frozen 4-bit base plus trainable LoRA matrices.
- **NF4-style quantization** — educational 16-value codebook with more resolution near zero.
- **Packed 4-bit storage** — two 4-bit codes stored in one uint8 byte.
- **Double quantization** — optional uint8 quantization of row scales.
- **QuantizedLinear4bit** — frozen base layer that dequantizes on forward.
- **QLoRA merge/export** — dequantize, add adapter delta, unload wrapper, and save a plain checkpoint.
- **New CLI** — `scripts/finetune_qlora.py`.
- **New CPU demo** — `examples/qlora_finetune_demo.py`.
- **New config** — `configs/apex1_tiny_qlora.yaml`.
- **New tests** — `tests/test_qlora.py`.
- **New guide** — `docs/35-qlora-4bit-finetuning.md`.

### v2.6.0 — LoRA Adapter Inference & Merge

- **Adapter generation CLI** — `scripts/generate_with_lora.py` loads saved adapters and runs generation.
- **Safe load order** — base checkpoint loads first, then LoRA is injected, then adapter weights load.
- **Runtime merge option** — generate with adapter-form weights or merge before generation.
- **Plain checkpoint export** — `scripts/merge_lora.py` merges LoRA deltas and saves a normal APEX checkpoint.
- **Unload LoRA wrappers** — merged models can be converted back to standard `nn.Linear` modules.
- **Inference helper module** — `apex/model/lora_inference.py` centralizes adapter loading, merging, unloading, and export utilities.
- **New CPU demo** — `examples/lora_generation_demo.py`.
- **New tests** — `tests/test_lora_inference.py`.
- **New config** — `configs/apex1_tiny_lora_inference.yaml`.
- **New guide** — `docs/34-lora-inference-and-merge.md`.

### v2.5.0 — LoRA / PEFT Fine-Tuning

- **LoRA from scratch** — `LoRALinear` wraps `nn.Linear` with trainable low-rank adapters.
- **PEFT config** — `PEFTConfig` added to `APEXConfig`.
- **Automatic adapter injection** — target attention, MLA, FFN, and MoE router modules by name.
- **Frozen base model** — train only adapter parameters.
- **Adapter-only checkpointing** — save/load tiny adapter state dicts.
- **Merge/unmerge support** — merge LoRA deltas into base weights for inference experiments.
- **LoRA SFT trainer** — `PEFTSFTTrainer` trains only `requires_grad=True` parameters.
- **New CLI** — `scripts/finetune_lora.py`.
- **New CPU demo** — `examples/lora_finetune_demo.py`.
- **New config** — `configs/apex1_tiny_lora.yaml`.
- **New tests** — `tests/test_lora_peft.py`.
- **New guide** — `docs/33-lora-peft-finetuning.md`.

### v2.4.0 — Evaluation, Benchmarking, Inspection & Vision Improvements

- Vision capability architecture — native ViT-style image encoder, vision-to-language projector, and `APEX1VisionModel`.
- `<|img|>` is active — image placeholders are replaced by continuous visual tokens before the transformer runs.
- CPU-friendly multimodal demos — `examples/vision_forward_demo.py` and `examples/vision_chat_demo.py`.
- Vision training utilities — JSONL vision dataset, visual-token label expansion, and multimodal SFT loss.
- Evaluation toolkit — token metrics, perplexity, generation-quality helpers.
- Inspection tools — parameter counts, layer-type summaries, ASCII architecture diagrams.
- Benchmarking helpers — small CPU forward-pass benchmark scripts.
- New guide — `docs/32-vision-capabilities.md`.

Full history in [CHANGELOG.md](CHANGELOG.md).

---

## 🗺️ Learning Path

**If you are completely new to AI:**  
Start at [docs/00-introduction.md](docs/00-introduction.md) and read in order. Each lesson builds on the previous one. By lesson 15 you will understand the complete forward pass of a modern LLM.

**If you know PyTorch but not transformers:**  
Start at [docs/04-embeddings-and-rmsnorm.md](docs/04-embeddings-and-rmsnorm.md). Skip lessons 00–03 or skim them.

**If you understand transformers but not modern LLMs:**  
Start at [docs/07-attention-mla.md](docs/07-attention-mla.md) — this is where APEX-1 diverges from standard transformer tutorials.

**If you want to understand alignment:**  
Jump directly to Part 8 (docs 24–29). The GRPO lesson (doc 26) is particularly relevant to current reasoning-model research.

**If you want to understand multimodal models:**  
Read [docs/32-vision-capabilities.md](docs/32-vision-capabilities.md). It explains image preprocessing, patch encoding, visual tokens, projector design, and multimodal SFT loss.

**If you want to understand efficient fine-tuning:**  
Read [docs/33-lora-peft-finetuning.md](docs/33-lora-peft-finetuning.md). It explains LoRA math, adapter injection, frozen-base training, adapter checkpoints, and merge/unmerge.

**If you want to understand QLoRA / quantized-base adapter training:**  
Read [docs/35-qlora-4bit-finetuning.md](docs/35-qlora-4bit-finetuning.md). It explains NF4-style quantization, packed 4-bit weights, double quantization, and frozen quantized base training.

**If you want to understand adapter inference and export:**  
Read [docs/34-lora-inference-and-merge.md](docs/34-lora-inference-and-merge.md). It explains how to load saved adapters, generate with them, merge them into base weights, unload wrappers, and save plain checkpoints.

**If you want the math:**  
The [Mathematical Reference](docs/APEX-1-Mathematical-Reference-Part1.md) covers all 34 formulas with full derivations and numerical examples.

---

## ✅ Recommended Test Commands Before Release

```bash
# Core tests
pytest tests/test_all.py -v
pytest tests/test_bugfixes.py -v

# Vision tests
pytest tests/test_vision.py -v

# Evaluation and inspector tests
pytest tests/test_eval_and_inspector.py -v

# LoRA / PEFT tests
pytest tests/test_lora_peft.py -v

# LoRA inference / merge tests
pytest tests/test_lora_inference.py -v

# QLoRA 4-bit PEFT tests
pytest tests/test_qlora.py -v
pytest tests/test_dora.py -v

# Everything
pytest tests/ -v
```

---

## 🤝 Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where contributions help:

- Kaggle/Colab training notebooks for APEX-1-Tiny
- Small pretrained educational checkpoints
- Additional test coverage for alignment modules
- Additional LoRA/PEFT examples
- Adapter inference, merge, QLoRA, and deployment examples
- CPU-friendly vision examples, datasets, and training notebooks
- Translations of documentation to other languages
- Bug reports and fixes

---

## 📜 Citation

```bibtex
@software{apex1_2026,
  title   = {APEX-1: A Best-of-All-Worlds Large Language + Vision Model},
  author  = {Aarambh Dev Hub},
  year    = {2026},
  url     = {https://github.com/AarambhDevHub/APEX-1},
  license = {Apache-2.0}
}
```

---

## 🙏 Acknowledgments

APEX-1 stands on the shoulders of giants. Architectural and educational inspiration includes:

- **Anthropic / Claude** — Constitutional AI and reasoning-style safety workflows
- **OpenAI / GPT-style systems** — process reward modeling and large-scale language modeling ideas
- **DeepSeek-V3/R1** — MLA, GRPO, auxiliary-loss-free load balancing, reasoning-model ideas
- **Alibaba / Qwen** — large vocabulary and multilingual/code-tokenizer inspiration
- **Google / Gemma and ViT** — efficient attention patterns and vision transformer foundations
- **Zhipu AI / GLM** — prefix bidirectional attention inspiration
- **Moonshot AI / KIMI** — YaRN-style long-context extension inspiration
- **MiniMax** — efficient MoE design inspiration
- **Meta / Llama** — GQA, SwiGLU, and practical transformer engineering inspiration
- **LLaVA / Flamingo / Qwen-VL research** — visual-token bridge and multimodal instruction-tuning pattern
- **LoRA / PEFT research community** — efficient low-rank adaptation, adapter fine-tuning, QLoRA-style quantized PEFT, and adapter merge/export workflows

---

## 💬 Community

Join our Discord for discussions, questions, and study groups:

[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/HDth6PfCnp)

---

## ❤️ Support the Work

APEX-1 is free and open source. If it helped you learn, consider supporting:

| Platform | Link |
|---|---|
| ☕ Buy Me a Coffee | [buymeacoffee.com/aarambhdevhub](https://buymeacoffee.com/aarambhdevhub) |
| 💖 GitHub Sponsors | [github.com/sponsors/aarambh-darshan](https://github.com/sponsors/aarambh-darshan) |
| 💳 Razorpay | [razorpay.me/@aarambhdevhub](https://razorpay.me/@aarambhdevhub) |

---

## 📄 License

[Apache License 2.0](LICENSE) — Copyright 2024–2026 Aarambh Dev Hub

Free to use, modify, and distribute with attribution.

---

<div align="center">

*Built with ❤️ by Aarambh Dev Hub — Teaching AI from the ground up.*

**[Start Learning →](docs/00-introduction.md)** · **[Learn Vision →](docs/32-vision-capabilities.md)** · **[Learn LoRA/PEFT →](docs/33-lora-peft-finetuning.md)** · **[Learn LoRA Inference →](docs/34-lora-inference-and-merge.md)** · **[Learn QLoRA →](docs/35-qlora-4bit-finetuning.md)**

</div>
