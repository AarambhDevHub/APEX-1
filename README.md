<div align="center">

# 🔺 APEX-1

### A Best-of-All-Worlds Large Language Model — v2.2.0

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Architecture%20Complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-86%20Passing-success.svg)]()

**Inspired by:** Claude · GPT-4.5 · DeepSeek-V3/R1 · Qwen3 · Gemma 4 · GLM-4 · KIMI · MiniMax · Llama 3

*Every component fully specified. Every gap filled.*

</div>

---

## What's New in v2.2.0

- **9 additional bug fixes** across training, tokenizer, generation, alignment, utilities, config, and CLI.
- **Speculative loss guard** — multi-token head losses now handle short sequences without producing `nan` (BUG-12).
- **Thinking token types** — `<|thinking|>` / `<|/thinking|>` now always labelled as assistant for SFT loss (BUG-14).
- **Probabilistic speculative decoding** — draft acceptance uses `min(1, p_target/p_draft)` instead of greedy argmax (BUG-15).
- **DPO bidirectional prompt attention** — `dpo_loss` now passes `prefix_len` for GLM-4-style prompt encoding (BUG-16).
- **Accurate FLOPs estimation** — SwiGLU elementwise multiply now included in counts (BUG-17).
- **Strict config validation** — `d_model != n_heads_q * d_head` now raises `ValueError` instead of warning (BUG-18).
- **Robust training CLI** — log file written to checkpoint dir with graceful fallback (BUG-20).
- **Shape checker accepts model** — `verify_shapes()` can now validate a pre-built model (BUG-23).
- **Streaming dataset padding fix** — padding tokens now excluded from training loss via `attention_mask` (BUG-24).

See the full [CHANGELOG](CHANGELOG.md) for details.

## Overview

APEX-1 is a decoder-only transformer that synthesizes the **single best innovation** from each frontier AI lab into one coherent, production-ready design:

| Feature | Source | Why It Wins |
|---|---|---|
| Large vocabulary (151K tokens) | Qwen3 | Better multilingual & code coverage |
| RoPE + YaRN extension | KIMI / DeepSeek | Extends context without retraining |
| Multi-Head Latent Attention (MLA) | DeepSeek-V3 | 93% KV cache reduction |
| GQA + Sliding Window | Llama 3 / Mistral | Efficient local attention |
| Interleaved local/global (1:6) | Gemma 4 | Long-context at fraction of compute |
| Prefix bidirectional attention | GLM-4 | Full context over system prompt |
| SwiGLU activation | PaLM / Llama | ~1-2% perplexity gain |
| 3-tier hierarchical MoE (256 experts) | DeepSeek-V3 | Frontier quality at fraction of FLOPs |
| Auxiliary-loss-free load balancing | DeepSeek-V3 | Stable utilization, zero LM loss interference |
| Dynamic skip gate | Early-exit research | 25-35% FFN compute saved |
| Multi-token prediction head | DeepSeek-V3 | 3× richer training signal |
| Thinking mode (CoT) | DeepSeek-R1 / Claude | Built-in reasoning scratchpad |
| GRPO alignment | DeepSeek-R1 | Stable RL, no reward model needed |
| Constitutional AI | Anthropic | Safety baked in, not patched on |

## Architecture

```
Input tokens [batch, seq_len]
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

## Model Sizes

| Parameter | Small | Medium | Large |
|---|---|---|---|
| `d_model` | 512 | 2,048 | 7,168 |
| `n_layers` | 12 | 36 | 72 |
| `n_heads_q` | 8 | 16 | 128 |
| `n_experts` | 8 | 64 | 256 |
| `max_seq_len` | 8K | 64K | 128K |
| Total params | ~100M | ~7B | ~900B |
| Active params | ~40M | ~2B | ~45B |

## Quick Start

```bash
# Clone
git clone https://github.com/AarambhDevHub/APEX-1.git
cd APEX-1

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Run forward pass demo
python examples/forward_pass_demo.py

# Run generation demo
python examples/generation_demo.py

# Run tests
pytest tests/ -v
```

## 📖 Documentation

A complete **beginner-to-expert** documentation suite lives in [`docs/`](docs/). Every file follows the same five-step format: **plain-English definition → analogy → LaTeX math → full annotated source code → design rationale**. All 24 bug fixes are explained in context.

### 🟢 Part 1 — Foundations

| File | What You Will Learn |
|---|---|
| [00 — Introduction](docs/00-introduction.md) | What an LLM is, key vocab, why build from scratch |
| [01 — Project Structure](docs/01-project-structure.md) | Every file and folder explained, reading order |
| [02 — Configuration](docs/02-configuration.md) | `APEXConfig` dataclasses, YAML loading, BUG-18 validation fix |
| [03 — Tokenizer](docs/03-tokenizer.md) | BPE, special tokens, token types for SFT masking, BUG-14 |

### 🔵 Part 2 — Building Blocks

| File | What You Will Learn |
|---|---|
| [04 — Embeddings & RMSNorm](docs/04-embeddings-and-rmsnorm.md) | Embedding scale, weight tying, RMSNorm formula |
| [05 — Positional Encoding (RoPE & YaRN)](docs/05-positional-encoding-rope.md) | Rotation math, three-regime YaRN scaling, BUG-22 vectorisation |
| [06 — Attention Masks](docs/06-attention-masks.md) | Prefix-bidir + causal + sliding window, BUG-10 vectorised fix |

### 🟣 Part 3 — Attention Mechanisms

| File | What You Will Learn |
|---|---|
| [07 — Multi-Head Latent Attention (MLA)](docs/07-attention-mla.md) | 93% KV cache compression, BUG-01 K_rope zeros, BUG-02 W_O size |
| [08 — GQA + Sliding Window](docs/08-attention-gqa.md) | Group sharing, window locality, 5:1 local/global ratio |

### 🟠 Part 4 — Feed-Forward Networks & Experts

| File | What You Will Learn |
|---|---|
| [09 — FFN & SwiGLU](docs/09-ffn-swiglu.md) | 3-matrix SwiGLU, dead neuron problem, BUG-17 FLOPs fix |
| [10 — Mixture of Experts](docs/10-mixture-of-experts.md) | 3-tier hierarchy, routing math, BUG-08 dispatch fix |
| [11 — Dynamic Skip Gate](docs/11-skip-gate.md) | STE binary threshold, 25-35% FFN savings, learned skip patterns |
| [12 — Auxiliary-Loss-Free Load Balancer](docs/12-load-balancer.md) | Expert collapse, sign-gradient bias update, BUG-11 |
| [13 — Multi-Token Prediction Head](docs/13-multi-token-head.md) | 4-head speculative prediction, auxiliary loss, BUG-12 NaN |

### 🔴 Part 5 — The Full Model

| File | What You Will Learn |
|---|---|
| [14 — Transformer Block](docs/14-transformer-block.md) | Pre-norm, residual connections, layer type assignment, BUG-19 |
| [15 — APEX1Model](docs/15-full-model.md) | Two RoPE caches BUG-07, KV position detection BUG-09, full forward pass |

### 🟡 Part 6 — Training

| File | What You Will Learn |
|---|---|
| [16 — Training Losses](docs/16-training-losses.md) | Cross-entropy, SFT token masking, speculative loss, BUG-12 |
| [17 — Optimizer & LR Scheduler](docs/17-scheduler-and-optimizer.md) | AdamW full math, cosine warmup curve, hyperparameter guide |
| [18 — Training Pipeline](docs/18-training-pipeline.md) | Mixed precision, gradient accumulation, full `PreTrainer`, BUG-11 |
| [19 — Checkpointing](docs/19-checkpointing.md) | What to save, RNG state, BUG-13 Python RNG fix, resume example |
| [20 — Datasets](docs/20-datasets.md) | 4 dataset types, packing, streaming, BUG-24 padding mask |

### ⚪ Part 7 — Text Generation

| File | What You Will Learn |
|---|---|
| [21 — Sampling Strategies](docs/21-generation-sampling.md) | KV cache, temperature/top-p/top-k/rep-penalty, full generator |
| [22 — Speculative Decoding](docs/22-speculative-decoding.md) | Draft-verify loop, probabilistic acceptance math, BUG-15 |
| [23 — Thinking Mode](docs/23-thinking-mode.md) | CoT scratchpad, budget management, BUG-21, temperature switching |

### 🟤 Part 8 — Alignment & Safety

| File | What You Will Learn |
|---|---|
| [24 — Reward Model](docs/24-reward-model.md) | Bradley-Terry loss, preference pairs, BUG-05 import fix |
| [25 — DPO](docs/25-dpo.md) | Implicit reward derivation, closed-form preference opt, BUG-16 |
| [26 — GRPO](docs/26-grpo.md) | RL without value function, PPO-clip explained, BUG-04 generation |
| [27 — Process Reward Model](docs/27-process-reward-model.md) | Step-level rewards, product scoring, BUG-06 tokenizer warning |
| [28 — Constitutional AI](docs/28-constitutional-ai.md) | Critique-revision loop, constitution format, BUG-03 |
| [29 — Combined Reward](docs/29-combined-reward.md) | Tri-signal formula (outcome + process + CAI), ablation table |

### ⚫ Part 9 — Utilities & Walkthrough

| File | What You Will Learn |
|---|---|
| [30 — Utilities](docs/30-utilities.md) | Shape checker BUG-23, FLOPs BUG-17, param counter |
| [31 — End-to-End Walkthrough](docs/31-end-to-end-walkthrough.md) | Full runnable code: install → pretrain → SFT → generate |

> **Start here if you are new:** [docs/00-introduction.md](docs/00-introduction.md)

---

## Project Structure

```
APEX-1/
├── apex/
│   ├── config.py              # Configuration dataclasses + YAML loading
│   ├── model/
│   │   ├── norm.py            # RMSNorm
│   │   ├── rope.py            # RoPE + YaRN
│   │   ├── mask.py            # Attention mask builder
│   │   ├── attention.py       # MLA + GQA+SW attention
│   │   ├── ffn.py             # DenseFFN + MoEFFN
│   │   ├── skip_gate.py       # Dynamic skip gate
│   │   ├── load_balancer.py   # Auxiliary-loss-free load balancing
│   │   ├── multi_token_head.py# Speculative prediction heads
│   │   ├── block.py           # Transformer block
│   │   └── apex_model.py      # Complete APEX-1 model
│   ├── tokenizer/
│   │   ├── tokenizer.py       # BPE tokenizer with special tokens
│   │   └── train_tokenizer.py # Tokenizer training script
│   ├── generation/
│   │   ├── sampler.py         # Sampling strategies
│   │   └── generator.py       # Generation engine + thinking mode
│   ├── training/
│   │   ├── losses.py          # Pretrain + SFT loss functions
│   │   ├── trainer.py         # PreTrainer + SFTTrainer
│   │   ├── scheduler.py       # Cosine warmup LR schedule
│   │   └── checkpoint.py      # Save/load checkpoints
│   ├── alignment/
│   │   ├── reward_model.py    # Reward model + Bradley-Terry loss
│   │   ├── dpo.py             # Direct Preference Optimization
│   │   ├── grpo.py            # Group Relative Policy Optimization
│   │   ├── prm.py             # Process Reward Model
│   │   ├── constitutional.py  # Constitutional AI
│   │   └── combined_reward.py # Combined reward function
│   ├── data/
│   │   ├── dataset.py         # Dataset classes
│   │   └── data_loader.py     # DataLoader factories
│   └── utils/
│       ├── param_counter.py   # Parameter counting
│       ├── shape_checker.py   # Shape verification
│       └── flops.py           # FLOPs estimation
├── configs/                   # YAML configs for all sizes
├── tests/                     # Comprehensive test suite
├── examples/                  # Demo scripts
└── APEX-1-Model-Architecture.md  # Full architecture document
```

## Citation

```bibtex
@software{apex1_2026,
  title  = {APEX-1: A Best-of-All-Worlds Large Language Model},
  author = {Aarambh Dev Hub},
  year   = {2026},
  url    = {https://github.com/AarambhDevHub/APEX-1},
  license = {Apache-2.0}
}
```

## Acknowledgments

APEX-1 stands on the shoulders of giants. We gratefully acknowledge the architectural innovations from:

- **Anthropic** (Claude) — Constitutional AI, reasoning approach
- **OpenAI** (GPT-4.5) — Process Reward Models
- **DeepSeek** (V3/R1) — MLA, GRPO, auxiliary-loss-free load balancing
- **Alibaba** (Qwen3) — Large vocabulary design
- **Google** (Gemma 4) — Interleaved attention pattern
- **Zhipu AI** (GLM-4) — Prefix bidirectional attention
- **Moonshot AI** (KIMI) — YaRN context extension
- **MiniMax** — Efficient MoE design
- **Meta** (Llama 3) — GQA + sliding window, SwiGLU

## 💬 Community

Join our Discord for discussions, support, and updates:

[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white)](https://discord.gg/HDth6PfCnp)

## ❤️ Support the Work

If APEX-1 is useful to you, consider supporting the project:

| Platform | Link |
|---|---|
| ☕ Buy Me a Coffee | [buymeacoffee.com/aarambhdevhub](https://buymeacoffee.com/aarambhdevhub) |
| 💖 GitHub Sponsors | [github.com/sponsors/aarambh-darshan](https://github.com/sponsors/aarambh-darshan) |
| 💳 Razorpay | [razorpay.me/@aarambhdevhub](https://razorpay.me/@aarambhdevhub) |

Your support helps us continue building open-source AI for everyone. 🙏

## License

[Apache License 2.0](LICENSE) — Copyright 2024-2026 Aarambh Dev Hub
