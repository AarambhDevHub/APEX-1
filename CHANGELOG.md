# Changelog

All notable changes to APEX-1 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-01

### Added
- Complete APEX-1 architecture implementation (v2.0)
- Multi-Head Latent Attention (MLA) on global layers — 93% KV cache reduction
- Grouped Query Attention + Sliding Window on local layers
- Interleaved local/global attention at 1:6 ratio
- Prefix bidirectional attention for system prompts (GLM-4 style)
- SwiGLU Feed-Forward Network with 3 weight matrices
- Mixture of Experts (MoE) with up to 256 routed experts
- Auxiliary-loss-free load balancing (DeepSeek-V3 approach)
- Dynamic skip gate with straight-through estimator
- Multi-token prediction head (4 speculative heads)
- RoPE + YaRN for context extension up to 1M+ tokens
- Flash Attention v3 integration via PyTorch SDPA
- Thinking mode with `<|thinking|>` budget enforcement
- Complete generation engine with KV cache management
- Speculative decoding using multi-token prediction heads
- Temperature, top-p, top-k, and repetition penalty sampling
- BPE tokenizer with 151,643 tokens and all special tokens
- Chat template formatting (system/user/assistant/thinking)
- Pretraining pipeline with multi-token auxiliary loss
- SFT pipeline with assistant-only loss masking
- Reward Model with Bradley-Terry loss
- DPO loss function
- GRPO full rollout loop with group-normalized advantages
- Process Reward Model (PRM) for step-level scoring
- Constitutional AI critique and revision loop
- Combined reward function (outcome + process + constitutional)
- AdamW optimizer with cosine warmup schedule
- Mixed precision training (AMP/FP16)
- Distributed training support (DDP)
- Gradient accumulation and clipping
- Checkpoint save/load with full state
- Streaming dataset for large corpora
- Data packing into fixed-length sequences
- Parameter counting and FLOPs estimation utilities
- Shape verification against architecture specification
- 4 model size configurations (Tiny, Small, Medium, Large)
- Comprehensive unit test suite
- Example scripts for forward pass, generation, thinking, mask visualization
- Full open-source repository structure

### Model Sizes
- **APEX-1-Small**: ~100M total params, ~40M active
- **APEX-1-Medium**: ~7B total params, ~2B active
- **APEX-1-Large**: ~900B total params, ~45B active
