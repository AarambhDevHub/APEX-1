# Changelog

All notable changes to APEX-1 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-04-30

### Fixed

**Training & Loss (BUG-12, BUG-24)**

- **BUG-12 `losses.py`** — Speculative head losses now guard against short sequences where the offset `k` leaves fewer than 1 overlapping token. The previous guard `if k >= token_ids.shape[1]` was off-by-one — when `seq_len - k < 1` the sliced tensors were empty, causing `nan` from the cross-entropy loss.

- **BUG-24 `dataset.py`** — `StreamingPretrainDataset` now emits an `attention_mask` alongside `input_ids`. Previously, when the final buffer was shorter than `seq_len`, padding tokens were added but treated as real training data — polluting the loss signal with meaningless pad-token predictions.

**Tokenizer (BUG-14)**

- **BUG-14 `tokenizer.py`** — `get_token_types()` now explicitly maps `<|thinking|>` and `<|/thinking|>` tokens to type 2 (assistant). Previously these tokens inherited the current type, which would be wrong if a thinking block appeared without a preceding `<|assistant|>` token — the thinking content would be labelled as system/user and excluded from the SFT loss.

**Generation (BUG-15)**

- **BUG-15 `generator.py`** — Speculative decoding draft acceptance is now probabilistic using `min(1, p_target / p_draft)` instead of greedy argmax comparison. The greedy approach altered the output distribution by only accepting drafts that matched the verification model's argmax, biasing output toward deterministic behaviour regardless of temperature.

**Alignment (BUG-16)**

- **BUG-16 `dpo.py`** — `dpo_loss` now passes `prefix_len=prompt_len` to the model so that prompt tokens receive bidirectional attention (GLM-4 style) instead of causal-only. Previously `prefix_len` defaulted to 0, producing a weaker contextual representation that degrades DPO training quality.

**Utilities (BUG-17, BUG-23)**

- **BUG-17 `flops.py`** — SwiGLU elementwise multiply `gate * value` was missing from the FLOPs estimate. Each SwiGLU layer performs `S × d_ffn` elementwise multiply ops in addition to the 3 matrix multiplications. The fix adds this contribution to both dense and MoE FFN estimates.

- **BUG-23 `shape_checker.py`** — `verify_shapes()` now accepts an optional `model` parameter instead of always creating a new `APEX1Model` internally. Previously it always instantiated a fresh model, meaning it tested a randomly-initialised model rather than the caller's actual model.

**Configuration (BUG-18)**

- **BUG-18 `config.py`** — `validate()` now raises `ValueError` when `d_model != n_heads_q * d_head` instead of logging a warning. The mismatch causes a hard shape error in the attention output projection (`W_O`), so it must be caught before model construction.

**CLI (BUG-20)**

- **BUG-20 `train.py`** — The training log file is now written to `<checkpoint_dir>/training.log` instead of unconditionally to CWD. The previous `FileHandler("training.log")` would fail with permission errors in read-only environments or pollute unrelated directories. The file handler is added lazily after arguments are parsed with a graceful fallback.

### Changed

- `apex/utils/shape_checker.py` — `verify_shapes()` signature now includes `model: Optional[APEX1Model] = None`.
- `apex/data/dataset.py` — `StreamingPretrainDataset.__iter__()` now yields dicts with both `input_ids` and `attention_mask`.
- `apex/utils/flops.py` — FLOPs estimates are now slightly higher due to the SwiGLU elementwise multiply correction.

---

## [2.1.0] - 2026-04-29

### Fixed

**Critical bugs (BUG-01 through BUG-07)**

- **BUG-01 `attention.py`** — MLA KV cache is now a tuple `(c_kv, K_rope_cache)`. Previously `K_rope_cache` was always re-initialised to zeros, causing all autoregressive steps after the first to attend to garbage positional encodings. The fix stores rotated K_rope values alongside the compressed content latent `c_kv` and concatenates them correctly at each decoding step.

- **BUG-02 `attention.py`** — `W_O` is now initialised with `n_heads_q * d_head` input features (not `n_heads_q * (d_head + d_head_rope)`). The rope component lives only in Q and K; the attention output `weights @ V` has head dimension `d_head`, so the merged input to `W_O` is `n_heads_q * d_head`. The previous initialisation caused a shape-mismatch crash on every forward pass.

- **BUG-03 `constitutional.py`** — `critique_response()` now calls `model.generate()` and parses the YES/NO judgment from the output. Previously it hardcoded `violated=False` for every principle, making Constitutional AI a complete no-op with no safety signal.

- **BUG-04 `grpo.py`** — The generation loop in `grpo_full_loop` now uses `APEX1Generator` instead of a broken manual single-token loop. The old loop passed a single token to the model at each step without a KV cache, reset logits on every iteration, and never produced coherent multi-token responses.

- **BUG-05 `reward_model.py`** — `from typing import Optional` is now at the top of the file. The original placement at the very bottom caused a `NameError` when the `RewardModel.forward()` signature was evaluated.

- **BUG-06 `prm.py`** — `score_steps_from_text` now raises a clear `ValueError` when `tokenizer=None` is passed (as was done in `combined_reward.py`), instead of crashing with `AttributeError: 'NoneType' object has no attribute 'encode'`. A new companion method `score_steps_from_text_pretokenized` is provided for callers that already have token IDs.

- **BUG-07 `apex_model.py`** — RoPE caches are now matched to their layer type. MLA (global) layers receive the `d_head_rope`-wide cache; GQA (local) layers receive the `d_head`-wide cache. Previously the model selected one cache at the model level, causing dimension mismatches for mixed-type stacks.

**Serious bugs (BUG-08 through BUG-13)**

- **BUG-08 `ffn.py`** — MoE expert dispatch now correctly handles batches of `n_e > 1` tokens routed to the same expert. The input is reshaped to `[1, n_e, d_model]` before calling `DenseFFN`, then the batch dim is squeezed away. The previous `unsqueeze(0)/squeeze(0)` pattern only worked when `n_e == 1`; with `n_e > 1` it silently used `n_e` as the sequence dimension, producing wrong gradients and wrong outputs.

- **BUG-09 `generator.py`** — KV-cache position tracking now uses `is_global_layer()` to determine the cache format instead of `isinstance(kv_caches[0], torch.Tensor)`. This is more robust to config changes and correctly handles the updated MLA cache format (which is now a tuple, not a bare tensor — see BUG-01).

- **BUG-10 `mask.py`** — The sliding-window mask is now fully vectorised with `torch.arange` broadcasting. The previous Python `for` loop executed 128 000 iterations per local layer per forward pass at 128 K context, dominating training wall-clock time. The new implementation is a single tensor operation.

- **BUG-11 `trainer.py`** — Each `LoadBalancer` is now created with `n_experts` taken from the actual MoE layer (`moe_ffn.n_experts`) instead of the global `config.moe.n_experts`. If per-layer expert counts differ the old code would silently use the wrong target rate. Also ensured that bias tensors are moved to the correct device via `MoEFFN.set_expert_bias()`.

- **BUG-13 `checkpoint.py`** — The `"python"` RNG state now stores `random.getstate()` (Python `random` module) and `"cpu"` stores `torch.random.get_rng_state()`. Previously both entries stored the same PyTorch tensor state, meaning the Python `random` module state was never saved or restored.

**Minor / code-quality bugs (BUG-19, BUG-21, BUG-22)**

- **BUG-19 `block.py`** — The `is_moe` flag now checks `config.moe.enabled` before evaluating the layer-frequency condition. Previously, blocks in a non-MoE model could be incorrectly labelled as MoE in `extra_repr()` output.

- **BUG-21 `generator.py`** — `thinking_token_count` is no longer incremented for the `<|thinking_start|>` token itself, so the full budget is available for actual thinking content.

- **BUG-22 `rope.py`** — `apply_yarn_scaling` is now fully vectorised using `torch.where` over dimension tensors. The previous Python `for` loop over all head dimensions ran in O(d_head) Python iterations, which was slow for large models.

### Added

- `tests/test_bugfixes.py` — Comprehensive regression test suite with dedicated test classes for each of the 15 fixed bugs (BUG-01 through BUG-22, excluding advisory-only entries).

### Changed

- `apex/model/attention.py` — `MLAAttention.forward()` now accepts and returns `Optional[MLACache]` where `MLACache = tuple[Tensor, Tensor]` (c_kv, K_rope_cache).  Callers that previously passed a bare tensor cache must be updated.
- `apex/alignment/prm.py` — `score_steps_from_text` now raises `ValueError` on `None` tokenizer instead of crashing silently.

**Post-review fixes and improvements**

- **`shape_checker.py`** — `verify_shapes()` now correctly validates MLA KV caches as tuples `(c_kv, K_rope)` instead of bare tensors. After the BUG-01 cache format change, the `isinstance(kv, torch.Tensor)` check always failed, causing all MLA layer shape checks to report false failures.

- **`test_all.py`** — `test_mla_kv_cache_growth` now accesses the `c_kv` tensor via `kv[0]` to match the updated MLA tuple cache format from BUG-01.

- **`rope.py`** — Reordered `torch.where` operations in `apply_yarn_scaling` so that the high-frequency override (no scaling) is applied last, giving it correct priority over low-frequency scaling.

### Improved

- **`mask.py`** — Removed dead code left over from the BUG-10 vectorisation refactor: an unused `causal = torch.tril(...)` variable and a Python `for` loop that was immediately overwritten by the vectorised broadcast below it.

- **`attention.py`** — Moved `from apex.model.rope import rotate_half` from inside `MLAAttention.forward()` to the module-level imports. The inline import was executing on every forward pass unnecessarily.

- **`load_balancer.py`** — Replaced the Python `for` loop counting per-expert assignments with a single `torch.bincount()` call. Significantly faster with large expert counts (e.g., 256 in APEX-1-Large).

---

## [2.0.0] - 2026-04-26

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