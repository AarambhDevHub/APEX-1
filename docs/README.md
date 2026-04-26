# APEX-1 Documentation

## Architecture

- [APEX-1 Model Architecture](../APEX-1-Model-Architecture.md) — Full design document

## Mathematical Reference

Comprehensive mathematical reference covering every formula used in APEX-1:

| Part | Topics | Link |
|---|---|---|
| **Part 1** | Embedding, RMSNorm, RoPE, YaRN | [Part 1](APEX-1-Mathematical-Reference-Part1.md) |
| **Part 2** | SDPA, MHA, GQA, MLA, Sliding Window, Masks | [Part 2](APEX-1-Mathematical-Reference-Part2.md) |
| **Part 3** | SwiGLU, MoE, Load Balancing, Skip Gate, Multi-Token | [Part 3](APEX-1-Mathematical-Reference-Part3.md) |
| **Part 4** | AdamW, LR Schedule, DPO, GRPO, Sampling, Full Pipeline | [Part 4](APEX-1-Mathematical-Reference-Part4.md) |

**Total: 34 formulas · 24 sections · 1,421 lines**

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Run demos
python examples/forward_pass_demo.py
python examples/generation_demo.py
python examples/thinking_mode_demo.py
python examples/mask_visualization.py

# Training
python scripts/train.py --config configs/apex1_tiny.yaml --mode pretrain --dry-run

# Generation
python scripts/generate.py --config configs/apex1_tiny.yaml --random --prompt "Hello"
```

## API Reference

### Core Model
- `APEX1Model(config)` — Complete model assembly
- `APEXTransformerBlock(layer_idx, config)` — Single transformer layer
- `MLAAttention(config)` — Multi-Head Latent Attention (global layers)
- `GQASlidingWindowAttention(config)` — GQA + Sliding Window (local layers)
- `DenseFFN(config)` — SwiGLU FFN
- `MoEFFN(config)` — Mixture of Experts FFN

### Generation
- `APEX1Generator(model, gen_config)` — Text generation engine
- `GenerationConfig(...)` — Sampling parameters

### Training
- `PreTrainer(model, config, loader)` — Pretraining pipeline
- `SFTTrainer(model, config, loader)` — SFT pipeline

### Alignment
- `RewardModel(backbone, d_model)` — Bradley-Terry reward model
- `dpo_loss(model, ref, chosen, rejected)` — DPO loss
- `grpo_training_step(model, ref, ...)` — GRPO step
- `ProcessRewardModel(backbone, d_model)` — Step-level PRM
- `ConstitutionalAI(model, tokenizer)` — Constitutional critique
- `combined_reward(outcome, process, constitutional)` — Combined reward
