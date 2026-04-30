# APEX-1 Documentation

> **Complete beginner-to-expert guide** — learn how a real AI language model is designed,
> coded, and trained by reading through every file in this project.

---

## 📚 How to Read This Documentation

If you are **new to AI/ML**, start at `00-introduction.md` and read in order.
Each document builds on the previous one. By the end, you will understand how
a modern large language model is built from scratch.

If you want to **jump to a specific topic**, use the table below.

---

## Table of Contents

### 🟢 Part 1 — Foundations (Start Here)

| # | File | What You Will Learn |
|---|---|---|
| 00 | [Introduction — What is an LLM?](00-introduction.md) | What AI models are, why we build them |
| 01 | [Project Structure](01-project-structure.md) | Every file and folder explained |
| 02 | [Configuration](02-configuration.md) | The blueprint (`config.py`) |
| 03 | [Tokenizer](03-tokenizer.md) | Turning text into numbers |

### 🔵 Part 2 — Building Blocks

| # | File | What You Will Learn |
|---|---|---|
| 04 | [Embeddings & RMSNorm](04-embeddings-and-rmsnorm.md) | Word vectors, normalisation |
| 05 | [Positional Encoding — RoPE & YaRN](05-positional-encoding-rope.md) | Teaching position to the model |
| 06 | [Attention Masks](06-attention-masks.md) | Who can see what |

### 🟣 Part 3 — Attention Mechanisms

| # | File | What You Will Learn |
|---|---|---|
| 07 | [Multi-Head Latent Attention (MLA)](07-attention-mla.md) | 93% KV cache reduction |
| 08 | [Grouped Query Attention + Sliding Window](08-attention-gqa.md) | Efficient local attention |

### 🟠 Part 4 — Feed-Forward Networks & Experts

| # | File | What You Will Learn |
|---|---|---|
| 09 | [FFN & SwiGLU](09-ffn-swiglu.md) | The fact-memory layer |
| 10 | [Mixture of Experts (MoE)](10-mixture-of-experts.md) | 256 specialists, 2 active |
| 11 | [Dynamic Skip Gate](11-skip-gate.md) | Skipping easy tokens |
| 12 | [Auxiliary-Loss-Free Load Balancer](12-load-balancer.md) | Balancing experts |
| 13 | [Multi-Token Prediction Head](13-multi-token-head.md) | Predicting 4 tokens at once |

### 🔴 Part 5 — The Full Model

| # | File | What You Will Learn |
|---|---|---|
| 14 | [Transformer Block](14-transformer-block.md) | One complete layer |
| 15 | [APEX1Model — The Complete Model](15-full-model.md) | All layers assembled |

### 🟡 Part 6 — Training

| # | File | What You Will Learn |
|---|---|---|
| 16 | [Training Losses](16-training-losses.md) | How the model learns |
| 17 | [Optimizer & LR Scheduler](17-scheduler-and-optimizer.md) | AdamW + cosine warmup |
| 18 | [Training Pipeline](18-training-pipeline.md) | The full training loop |
| 19 | [Checkpointing](19-checkpointing.md) | Saving and restoring |
| 20 | [Datasets](20-datasets.md) | Loading and preparing data |

### ⚪ Part 7 — Text Generation

| # | File | What You Will Learn |
|---|---|---|
| 21 | [Sampling Strategies](21-generation-sampling.md) | How text is generated |
| 22 | [Speculative Decoding](22-speculative-decoding.md) | 3× faster generation |
| 23 | [Thinking Mode](23-thinking-mode.md) | Built-in reasoning scratchpad |

### 🟤 Part 8 — Alignment & Safety

| # | File | What You Will Learn |
|---|---|---|
| 24 | [Reward Model](24-reward-model.md) | What humans prefer |
| 25 | [DPO — Direct Preference Optimization](25-dpo.md) | Training on preferences |
| 26 | [GRPO — Group Relative Policy Optimization](26-grpo.md) | RL without a value function |
| 27 | [Process Reward Model](27-process-reward-model.md) | Rewarding good reasoning |
| 28 | [Constitutional AI](28-constitutional-ai.md) | Safety baked in |
| 29 | [Combined Reward](29-combined-reward.md) | All signals together |

### ⚫ Part 9 — Utilities & Walkthrough

| # | File | What You Will Learn |
|---|---|---|
| 30 | [Utilities](30-utilities.md) | Shape checker, FLOPs, param counter |
| 31 | [End-to-End Walkthrough](31-end-to-end-walkthrough.md) | Full journey, runnable code |

---

## Mathematical Reference (Advanced)

The original mathematical reference documents are still available:

| Part | Topics |
|---|---|
| [Math Ref Part 1](APEX-1-Mathematical-Reference-Part1.md) | Embedding, RMSNorm, RoPE, YaRN |
| [Math Ref Part 2](APEX-1-Mathematical-Reference-Part2.md) | SDPA, MHA, GQA, MLA, Masks |
| [Math Ref Part 3](APEX-1-Mathematical-Reference-Part3.md) | SwiGLU, MoE, Load Balancing, Skip Gate |
| [Math Ref Part 4](APEX-1-Mathematical-Reference-Part4.md) | AdamW, LR Schedule, DPO, GRPO, Sampling |

---

## Full Architecture Document

- [APEX-1 Model Architecture](../APEX-1-Model-Architecture.md) — Complete technical design

---

## Quick Start (Run the Code Now)

```bash
# 1. Install
pip install -e ".[all]"

# 2. Run a forward pass
python examples/forward_pass_demo.py

# 3. Generate text
python examples/generation_demo.py

# 4. Generate with thinking mode
python examples/thinking_mode_demo.py

# 5. Run all tests
pytest tests/ -v
```
