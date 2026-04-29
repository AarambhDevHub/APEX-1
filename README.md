<div align="center">

# 🔺 APEX-1

### A Best-of-All-Worlds Large Language Model — v2.1.0

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Architecture%20Complete-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-86%20Passing-success.svg)]()

**Inspired by:** Claude · GPT-4.5 · DeepSeek-V3/R1 · Qwen3 · Gemma 4 · GLM-4 · KIMI · MiniMax · Llama 3

*Every component fully specified. Every gap filled.*

</div>

---

## What's New in v2.1.0

- **15 bug fixes** across model core, alignment pipeline, training, and generation — including critical fixes to MLA KV caching (BUG-01), attention output projection (BUG-02), and Constitutional AI (BUG-03).
- **Vectorised mask building** — sliding-window and global masks now use `torch.arange` broadcasting instead of Python loops, dramatically faster at long contexts.
- **Vectorised YaRN scaling** — `apply_yarn_scaling` replaced per-dimension Python loop with `torch.where` operations.
- **Load balancer optimised** — expert counting uses `torch.bincount` instead of a Python loop.
- **Shape checker fixed** — `verify_shapes()` correctly validates the updated MLA tuple cache format.
- **86 tests passing** — comprehensive regression test suite covering all fixed bugs.

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
