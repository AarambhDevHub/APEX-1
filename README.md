<div align="center">

# 🔺 APEX-1

### A Best-of-All-Worlds Large Language + Vision Model — v2.3.0

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Architecture%20Complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-96%20Passing-success.svg)]()
[![Docs](https://img.shields.io/badge/Docs-32%20Guides-orange.svg)](docs/)
[![Course](https://img.shields.io/badge/Course-Free%20%26%20Open-purple.svg)](docs/00-introduction.md)

**Inspired by:** Claude · GPT-4.5 · DeepSeek-V3/R1 · Qwen3 · Gemma 4 · GLM-4 · KIMI · MiniMax · Llama 3 · LLaVA · Flamingo · ViT

*Build a frontier-grade language + vision model from scratch. Understand every line.*

---

### 🆓 This Course Is Completely Free

Other LLM courses charge **$50–$500+** for content like this. APEX-1 is free and always will be — 32 lessons, 4 math references, 24 bug-fix engineering lessons, full annotated source code, and 96 tests. No paywalls. No sign-ups. Just open source.

If this helped you learn, please consider supporting so we can keep building free education:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Free%20Education-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/aarambhdevhub)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor%20This%20Work-EA4AAA?style=for-the-badge&logo=github-sponsors&logoColor=white)](https://github.com/sponsors/aarambh-darshan)
[![Razorpay](https://img.shields.io/badge/Razorpay-Donate-02042B?style=for-the-badge&logo=razorpay&logoColor=white)](https://razorpay.me/@aarambhdevhub)

> *Every contribution — however small — directly funds more free content, more lessons, and more open-source AI education for everyone.*

</div>

---

## 🎓 What Is APEX-1?

APEX-1 is two things at once.

**As an architecture**, it is a production-grade decoder-only transformer that synthesizes the single best innovation from each major AI lab into one coherent design — Multi-Head Latent Attention from DeepSeek, Mixture of Experts routing, GRPO alignment from DeepSeek-R1, Constitutional AI from Anthropic, GQA from Llama 3, and more.

**As a course**, it is a complete beginner-to-expert curriculum for understanding how modern large language models actually work — not toy GPT-2 clones, but the real techniques inside frontier models like Claude, GPT-4, and DeepSeek. Every component is fully documented, every design decision is explained, and 24 real bugs are preserved and explained as engineering lessons.

**As a vision-language model preview**, APEX-1 can now accept images through the existing `<|img|>` token. Images are encoded into visual tokens, projected into the APEX hidden space, and processed by the same decoder-only transformer context. This is the foundation for image captioning, visual question answering, screenshot understanding, chart understanding, and future multi-image reasoning.


> **If you have ever wanted to understand what is really inside a modern LLM — not just the theory but the actual code — this is for you.**

---

## 🖼️ Vision Capability Status

APEX-1 includes a complete **educational multimodal architecture**. It can load an image, preprocess it, encode it into patch features, project those features into APEX's language hidden space, replace the `<|img|>` placeholder with continuous visual tokens, and run image + text through the same decoder-only transformer.

This means the vision pipeline is architecturally complete and fully testable on CPU.

> **Important:** APEX-1 does **not** ship with pretrained vision weights. The model can process images, but real semantic image understanding requires training on image-caption or visual-instruction datasets. This is intentional: APEX-1 is a course project for learning the architecture from scratch, not a large pretrained multimodal checkpoint.

Course status:

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
| **CS / Engineering students** | A complete hands-on project that covers what university ML courses skip |
| **Self-taught developers** | A structured path from "what is a token" to "how does GRPO work" |
| **ML practitioners** | Deep dives into MLA, MoE, speculative decoding, and modern alignment techniques |
| **Researchers** | A fully-specified, reproducible reference architecture synthesizing 2024–2025 frontier techniques |
| **YouTube / content learners** | 32 documentation files, each structured as a complete lesson |

---

## 📚 The Curriculum — 32 Lessons

Every lesson follows the same five-step format:

> **Plain-English definition → Real-world analogy → LaTeX math → Full annotated source code → Design rationale**

### 🟢 Part 1 — Foundations

| Lesson | Topic | Key Concepts |
|---|---|---|
| [00](docs/00-introduction.md) | What Is a Language Model? | Tokens, loss, training loop |
| [01](docs/01-project-structure.md) | Project Structure | Every file explained, reading order |
| [02](docs/02-configuration.md) | Configuration System | Hyperparameters, YAML loading, BUG-18 validation |
| [03](docs/03-tokenizer.md) | Tokenizer | BPE algorithm, special tokens, SFT masking, BUG-14 |

### 🔵 Part 2 — Building Blocks

| Lesson | Topic | Key Concepts |
|---|---|---|
| [04](docs/04-embeddings-and-rmsnorm.md) | Embeddings & RMSNorm | Weight tying, √d scaling, normalisation math |
| [05](docs/05-positional-encoding-rope.md) | RoPE & YaRN | Rotation math, three-regime YaRN, BUG-22 |
| [06](docs/06-attention-masks.md) | Attention Masks | Prefix bidir, causal, sliding window, BUG-10 |

### 🟣 Part 3 — Attention Mechanisms

| Lesson | Topic | Key Concepts |
|---|---|---|
| [07](docs/07-attention-mla.md) | Multi-Head Latent Attention | 93% KV cache reduction, BUG-01, BUG-02 |
| [08](docs/08-attention-gqa.md) | GQA + Sliding Window | Group sharing, local/global ratio |

### 🟠 Part 4 — Feed-Forward Networks & Experts

| Lesson | Topic | Key Concepts |
|---|---|---|
| [09](docs/09-ffn-swiglu.md) | FFN & SwiGLU | Gating, dead neurons, 3-matrix design, BUG-17 |
| [10](docs/10-mixture-of-experts.md) | Mixture of Experts | 3-tier hierarchy, routing math, BUG-08 |
| [11](docs/11-skip-gate.md) | Dynamic Skip Gate | STE binary threshold, 25–35% FFN savings |
| [12](docs/12-load-balancer.md) | Auxiliary-Loss-Free Load Balancer | Expert collapse, sign-gradient bias, BUG-11 |
| [13](docs/13-multi-token-head.md) | Multi-Token Prediction | 4× training signal, speculative decoding, BUG-12 |

### 🔴 Part 5 — The Full Model

| Lesson | Topic | Key Concepts |
|---|---|---|
| [14](docs/14-transformer-block.md) | Transformer Block | Pre-norm, residuals, layer assignment, BUG-19 |
| [15](docs/15-full-model.md) | Complete APEX-1 Model | Two RoPE caches BUG-07, KV position BUG-09 |

### 🟡 Part 6 — Training

| Lesson | Topic | Key Concepts |
|---|---|---|
| [16](docs/16-training-losses.md) | Training Losses | Cross-entropy, SFT masking, BUG-12 NaN fix |
| [17](docs/17-scheduler-and-optimizer.md) | Optimizer & LR Schedule | AdamW full math, cosine warmup |
| [18](docs/18-training-pipeline.md) | Training Pipeline | Mixed precision, gradient accumulation, BUG-11 |
| [19](docs/19-checkpointing.md) | Checkpointing | RNG state, resume training, BUG-13 |
| [20](docs/20-datasets.md) | Datasets | Streaming, packing, BUG-24 padding mask |

### ⚪ Part 7 — Text Generation

| Lesson | Topic | Key Concepts |
|---|---|---|
| [21](docs/21-generation-sampling.md) | Sampling Strategies | KV cache, temperature, top-p, top-k |
| [22](docs/22-speculative-decoding.md) | Speculative Decoding | Draft-verify loop, probabilistic acceptance, BUG-15 |
| [23](docs/23-thinking-mode.md) | Thinking Mode | CoT scratchpad, budget, BUG-21 |

### 🟤 Part 8 — Alignment & Safety

| Lesson | Topic | Key Concepts |
|---|---|---|
| [24](docs/24-reward-model.md) | Reward Model | Bradley-Terry loss, BUG-05 import fix |
| [25](docs/25-dpo.md) | DPO | Implicit reward, closed-form preference, BUG-16 |
| [26](docs/26-grpo.md) | GRPO | RL without value function, PPO-clip, BUG-04 |
| [27](docs/27-process-reward-model.md) | Process Reward Model | Step-level rewards, BUG-06 |
| [28](docs/28-constitutional-ai.md) | Constitutional AI | Critique-revision loop, BUG-03 |
| [29](docs/29-combined-reward.md) | Combined Reward | Tri-signal formula, ablation results |

### ⚫ Part 9 — Utilities & Walkthrough

| Lesson | Topic | Key Concepts |
|---|---|---|
| [30](docs/30-utilities.md) | Utilities | Shape checker BUG-23, FLOPs BUG-17, param counter |
| [31](docs/31-end-to-end-walkthrough.md) | End-to-End Walkthrough | Full runnable code: install → pretrain → SFT → generate |
| [32](docs/32-vision-capabilities.md) | Vision Capabilities | Image encoder, Perceiver projector, visual-token insertion, multimodal SFT |

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
- Why it was wrong (with the exact failure mode)
- The fix and why it works
- A regression test to prevent recurrence

This is what most courses skip and what real ML engineers spend most of their time doing.

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
| BUG-10 | `mask.py` | Sliding window mask used Python loop — 128K iterations at long context |
| BUG-11 | `trainer.py` | Load balancer used global config n_experts, not per-layer actual count |
| BUG-12 | `losses.py` | Short-sequence speculative loss produced `nan` — silent training corruption |
| BUG-13 | `checkpoint.py` | Python RNG state saved as PyTorch tensor — non-reproducible resume |
| BUG-14 | `tokenizer.py` | Thinking tokens inherited wrong type — excluded from SFT loss |
| BUG-15 | `generator.py` | Speculative acceptance was greedy argmax — biased output distribution |
| BUG-16 | `dpo.py` | Prompt processed causally in DPO — weaker context representation |
| BUG-17 | `flops.py` | SwiGLU elementwise multiply missing from FLOPs estimate |
| BUG-18 | `config.py` | d_model mismatch logged as warning, not error — silent model corruption |
| BUG-19 | `block.py` | `is_moe` flag ignored `config.moe.enabled` — wrong FFN type |
| BUG-20 | `train.py` | Log file written to CWD — failed in read-only environments |
| BUG-21 | `generator.py` | Thinking start token consumed 1 budget slot |
| BUG-22 | `rope.py` | YaRN scaling used Python loop over d_head — slow for large models |
| BUG-23 | `shape_checker.py` | Always created a new model instead of using the provided one |
| BUG-24 | `dataset.py` | Padding tokens included in training loss — corrupted pretraining signal |

---

## 🏗️ Architecture

APEX-1 picks the single best innovation from each frontier lab:

| Feature | Source | Why It Wins |
|---|---|---|
| Large vocabulary (151K tokens) | Qwen3 | Better multilingual & code coverage |
| RoPE + YaRN extension | KIMI / DeepSeek | Extends context without retraining |
| Multi-Head Latent Attention (MLA) | DeepSeek-V3 | 93% KV cache reduction |
| GQA + Sliding Window | Llama 3 / Mistral | Efficient local attention |
| Interleaved local/global (1:6) | Gemma 4 | Long-context at fraction of compute |
| Prefix bidirectional attention | GLM-4 | Full context over system prompt |
| SwiGLU activation | PaLM / Llama | ~1–2% perplexity gain over ReLU |
| 3-tier hierarchical MoE (256 experts) | DeepSeek-V3 | Frontier quality at fraction of FLOPs |
| Auxiliary-loss-free load balancing | DeepSeek-V3 | Stable expert utilization, zero LM loss interference |
| Dynamic skip gate | Early-exit research | 25–35% FFN compute saved |
| Multi-token prediction | DeepSeek-V3 | 3× richer training signal, 2.5× inference speedup |
| Thinking mode (CoT) | DeepSeek-R1 / Claude | Built-in reasoning scratchpad |
| GRPO alignment | DeepSeek-R1 | Stable RL, no reward model needed |
| Constitutional AI | Anthropic | Safety baked in, not patched on |
| Visual token bridge | LLaVA / Flamingo / Qwen-VL style | Images become context tokens without rewriting the decoder |
| Native ViT encoder | Vision Transformer | From-scratch educational image encoder |
| Perceiver resampler | Flamingo-style compression | Fixed visual token budget for efficient multimodal context |

```
Input tokens [batch, seq_len]
        │
        ▼
Image pixels [batch, 3, H, W]
        │
        ▼
┌─────────────────────┐
│ Vision Encoder      │  ViT patch features
└─────────┬───────────┘
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
│  └─────────┘    │  MLA (global layers) │     │
│                 │  GQA+SW (local)      │     │
│                 └──────────┬───────────┘     │
│                    + residual                │
│                            │                 │
│  ┌─────────┐    ┌─────────▼──────────┐      │
│  │ Skip    │───►│ FFN                │      │
│  │ Gate    │    │  Dense (even layers)│      │
│  └─────────┘    │  MoE   (odd layers)│      │
│                 └──────────┬─────────┘      │
│                    + residual (gated)        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   RMSNorm     │
              │   LM Head     │  → logits [batch, seq, vocab]
              │   Spec Heads  │  → 4 speculative predictions
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

Start with **APEX-1-Tiny** (`configs/apex1_tiny.yaml`) — ~1M params, runs on CPU in seconds. For multimodal lessons, start with **APEX-1-Tiny-Vision** (`configs/apex1_tiny_vision.yaml`), which also runs on CPU for forward-pass demos and tests.

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

# Run a text forward pass (no training needed)
python examples/forward_pass_demo.py

# Run text generation
python examples/generation_demo.py

# Try thinking mode
python examples/thinking_mode_demo.py

# Visualise attention masks
python examples/mask_visualization.py

# Try a vision forward pass (CPU-friendly)
python examples/vision_forward_demo.py

# Run only vision tests
pytest tests/test_vision.py -v

# Run the full suite: 86 core + 10 vision tests
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

---

## 🧪 Evaluation, Benchmarking & Inspection — v2.4.0

APEX-1 now includes a small educational evaluation and benchmarking toolkit.

This helps learners answer practical model-engineering questions:

- How many parameters does my model have?
- Which layers use global MLA vs local GQA+SW?
- Which layers use MoE vs dense FFN?
- How fast is a tiny forward pass?
- What is the model's next-token perplexity?
- Are generated texts repetitive?
- Did the vision forward pass insert the expected number of visual tokens?

New commands:

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

# Run new demos
python examples/eval_demo.py
python examples/benchmark_demo.py
python examples/inspect_model_demo.py
python examples/architecture_diagram_demo.py
python examples/tiny_dataset_demo.py

# Run the new tests
pytest tests/test_eval_and_inspector.py -v
```

---

## 📁 Project Structure

```txt
APEX-1/
├── apex/
│   ├── config.py                 # All hyperparameters — text, training, alignment, vision
│   │
│   ├── eval/
│   │   ├── metrics.py                  # token accuracy + token cross-entropy
│   │   ├── perplexity.py               # next-token perplexity evaluation
│   │   ├── generation_quality.py       # distinct-n, repetition, average length
│   │   ├── vision_eval.py              # vision forward-output validation
│   │   └── benchmark.py                # tiny forward-pass benchmark helper
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
│   │   ├── apex_model.py         # The complete text-only APEX-1 model
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
│   ├── training/
│   │   ├── losses.py             # Text pretraining + SFT losses
│   │   ├── vision_losses.py      # Vision SFT loss + visual-token label expansion
│   │   ├── trainer.py            # Training loop
│   │   ├── scheduler.py          # LR scheduler
│   │   └── checkpoint.py         # Save / load checkpoints
│   │
│   ├── alignment/                # Reward model, DPO, GRPO, PRM, CAI
│   │
│   ├── data/
│   │   ├── dataset.py            # Text dataset classes + DataLoader factories
│   │   └── vision_dataset.py     # Image-caption / vision-instruction dataset
│   │
│   └── utils/                    # Shape checker, FLOPs, param counter, ASCII / Markdown architecture map
│
├── configs/
│   ├── apex1_tiny.yaml           # Tiny text-only config
│   ├── apex1_small.yaml          # Small text-only config
│   ├── apex1_medium.yaml         # Medium text-only config
│   ├── apex1_large.yaml          # Large text-only config
│   └── apex1_tiny_vision.yaml    # Tiny multimodal config for CPU tests/demos
│
├── docs/
│   ├── 00-introduction.md
│   ├── ...
│   ├── 31-end-to-end-walkthrough.md
│   ├── 32-vision-capabilities.md # Vision architecture lesson
│   └── APEX-1-Mathematical-Reference-Part*.md
│
├── tests/
│   ├── test_all.py               # Core unit tests
│   ├── test_bugfixes.py          # Regression tests for documented bugs
│   └── test_vision.py            # Vision config, encoder, projector, model, loss tests
│
├── examples/
│   ├── forward_pass_demo.py      # Text forward-pass demo
│   ├── generation_demo.py        # Text generation demo
│   ├── thinking_mode_demo.py     # Thinking mode demo
│   ├── mask_visualization.py     # Attention mask visualisation
│   ├── vision_forward_demo.py    # Image + text forward-pass demo
│   └── vision_chat_demo.py       # Vision chat prompt demo
│
├── scripts/
│   ├── train.py                  # Text training CLI
│   ├── generate.py               # Text generation CLI
│   ├── benchmark.py                # CLI benchmark
│   ├── inspect_model.py            # CLI model inspector
│   └── print_architecture.py       # CLI architecture diagram
│
├── data/samples/
│   ├──  tiny_text.jsonl             # text pretraining format example
│   ├──  tiny_sft.jsonl              # supervised fine-tuning format example
│   ├──  tiny_preference.jsonl       # preference data format example
│   └──  tiny_vision.jsonl           # vision instruction format example
│
├── README.md
├── CHANGELOG.md
├── pyproject.toml
└── LICENSE
```

---


## 🧪 What's New in v2.4.0

- **Vision capability architecture** — native ViT-style image encoder, vision-to-language projector, and `APEX1VisionModel`.
- **`<|img|>` is now active** — image placeholders are replaced by continuous visual tokens before the transformer runs.
- **CPU-friendly multimodal demos** — `examples/vision_forward_demo.py` and `examples/vision_chat_demo.py`.
- **Vision training utilities** — JSONL vision dataset, visual-token label expansion, and multimodal SFT loss.
- **New guide** — `docs/32-vision-capabilities.md`.
- **96 passing tests** — 86 core tests + 10 vision tests.

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
Jump directly to Part 8 (docs 24–29). The GRPO lesson (doc 26) is particularly relevant to current frontier research.

**If you want the math:**
The [Mathematical Reference](docs/APEX-1-Mathematical-Reference-Part1.md) covers all 34 formulas with full derivations and numerical examples.

---

## 🤝 Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where contributions help:
- Kaggle/Colab training notebooks for APEX-1-Tiny
- Additional test coverage for alignment modules
- Translations of documentation to other languages
- Bug reports and fixes
- CPU-friendly vision examples, datasets, and training notebooks

---

## 📜 Citation

```bibtex
@software{apex1_2026,
  title  = {APEX-1: A Best-of-All-Worlds Large Language + Vision Model},
  author = {Aarambh Dev Hub},
  year   = {2026},
  url    = {https://github.com/AarambhDevHub/APEX-1},
  license = {Apache-2.0}
}
```

---

## 🙏 Acknowledgments

APEX-1 stands on the shoulders of giants. Architectural innovations from:

- **Anthropic** (Claude) — Constitutional AI, reasoning approach
- **OpenAI** (GPT-4.5) — Process Reward Models
- **DeepSeek** (V3/R1) — MLA, GRPO, auxiliary-loss-free load balancing
- **Alibaba** (Qwen3) — Large vocabulary design
- **Google** (Gemma 4) — Interleaved attention pattern
- **Zhipu AI** (GLM-4) — Prefix bidirectional attention
- **Moonshot AI** (KIMI) — YaRN context extension
- **MiniMax** — Efficient MoE design
- **Meta** (Llama 3) — GQA + sliding window, SwiGLU
- **LLaVA / Flamingo / ViT research** — visual-token bridge, image encoder, and multimodal instruction-tuning pattern

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

**[Start Learning →](docs/00-introduction.md)**

</div>