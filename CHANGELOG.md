# Changelog

All notable changes to APEX-1 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v3.0.0 — Course-Ready Stable Release

APEX-1 v3.0.0 is the stable course-ready release for the full "Build Your Own AI Model From Scratch" learning path.

This release does not add another large model feature. Instead, it makes the repository easier to teach, verify, test, and maintain.

### Added

- Added `MODEL_CARD.md`.
- Added `COURSE_READY_CHECKLIST.md`.
- Added `docs/38-course-ready-release.md`.
- Added `scripts/course_ready_check.py`.
- Added GitHub Actions CI workflow at `.github/workflows/ci.yml`.
- Added pull request template at `.github/pull_request_template.md`.
- Added `V3_0_0_GITHUB_RELEASE_NOTES.md`.

### Changed

- Updated project version from `2.9.0` to `3.0.0`.
- Updated README title/version to `v3.0.0`.
- Updated README curriculum references from 35 lessons to 37 lessons.
- Fixed release history:
  - `v2.5.0` = LoRA / PEFT fine-tuning
  - `v2.6.0` = LoRA inference + merge/export
  - `v2.7.0` = QLoRA 4-bit fine-tuning
  - `v2.8.0` = DoRA / QDoRA
  - `v2.9.0` = Adapter-DPO alignment
  - `v3.0.0` = Course-ready stable release
- Updated Makefile with course-ready commands:
  - `make course-check`
  - `make course-check-examples`
  - `make course-check-tests`
  - `make course-check-full`
  - `make demo-all`
- Updated Dockerfile to remove the missing `requirements.txt` copy and use a CPU-friendly course setup.

### Fixed

- Fixed README version confusion where DoRA was described as v2.9.0 instead of v2.8.0.
- Fixed Docker build setup that referenced a missing `requirements.txt`.

### Verification

Recommended local verification:

```bash
python scripts/course_ready_check.py --mode quick
python scripts/course_ready_check.py --mode examples
pytest tests/ -v
```

### Important Note

APEX-1 remains an educational architecture. It does not ship with a large pretrained checkpoint. Tiny CPU demos verify mechanics and architecture, while real high-quality generation requires real training data, compute, trained checkpoints, evaluation, and safety testing.

## v2.9.0 — Adapter-DPO Alignment

v2.9.0 adds Direct Preference Optimization for PEFT adapters.

### Added

- `apex/alignment/adapter_dpo.py`
  - adapter-DPO loss
  - preference JSONL dataset
  - adapter-only DPO trainer
  - frozen reference model helper
- `scripts/finetune_adapter_dpo.py`
- `examples/adapter_dpo_demo.py`
- `tests/test_adapter_dpo.py`
- `docs/37-adapter-dpo-alignment.md`
- `data/samples/tiny_preference.jsonl`
- Adapter-DPO configs:
  - `configs/apex1_tiny_lora_dpo.yaml`
  - `configs/apex1_tiny_qlora_dpo.yaml`
  - `configs/apex1_tiny_dora_dpo.yaml`
  - `configs/apex1_tiny_qdora_dpo.yaml`

### Updated

- `apex/config.py`
  - added `AdapterDPOConfig`
  - added `adapter_dpo` section to `APEXConfig`
  - added `get_tiny_adapter_dpo_config()`
- `apex/__init__.py` version to `2.9.0`
- `pyproject.toml` version to `2.9.0`
- `README.md` v2.9.0 release notes
- `apex/model/lora.py` adapter checkpoint metadata version to `2.9.0`

### Test Commands

```bash
pytest tests/test_lora_peft.py -v
pytest tests/test_lora_inference.py -v
pytest tests/test_qlora.py -v
pytest tests/test_dora.py -v
pytest tests/test_adapter_dpo.py -v
python examples/adapter_dpo_demo.py
```

### Example Training

```bash
python scripts/finetune_adapter_dpo.py \
  --config configs/apex1_tiny_lora_dpo.yaml \
  --data data/samples/tiny_preference.jsonl \
  --output-dir outputs/adapter-dpo-test \
  --max-steps 10
```


## v2.8.0 — DoRA / Weight-Decomposed LoRA

APEX-1 v2.8.0 adds an educational DoRA implementation on top of the existing LoRA, QLoRA, and adapter inference stack.

### Added

- `DoRALinear` adapter layer.
- `QDoRALinear` optional quantized DoRA experiment.
- `peft.method: dora` and `peft.method: qdora` support.
- Trainable `dora_magnitude` parameters.
- Adapter-only save/load for DoRA magnitude + LoRA A/B direction weights.
- Merge/unload support for DoRA and QDoRA.
- `get_tiny_dora_config()` and `get_tiny_qdora_config()`.
- `configs/apex1_tiny_dora.yaml`.
- `configs/apex1_tiny_dora_inference.yaml`.
- `configs/apex1_tiny_qdora.yaml`.
- `scripts/finetune_dora.py`.
- `examples/dora_finetune_demo.py`.
- `tests/test_dora.py`.
- `docs/36-dora-weight-decomposed-lora.md`.

### Changed

- README updated to v2.8.0.
- `pyproject.toml` version bumped to `2.8.0`.
- PEFT summaries now report DoRA module counts.
- Adapter checkpoint metadata now includes `num_dora_modules`.
- Inference helpers now describe LoRA/QLoRA/DoRA adapter flows.

### Verify

```bash
pytest tests/test_lora_peft.py -v
pytest tests/test_lora_inference.py -v
pytest tests/test_qlora.py -v
pytest tests/test_dora.py -v
python examples/dora_finetune_demo.py
python scripts/finetune_dora.py   --config configs/apex1_tiny_dora.yaml   --data data/samples/tiny_sft.jsonl   --output-dir outputs/dora-test   --max-steps 10
```


## v2.7.0 — QLoRA 4-bit PEFT Fine-Tuning

APEX-1 v2.7.0 adds an educational QLoRA-style 4-bit fine-tuning workflow.

v2.5.0 added LoRA training. v2.6.0 added adapter inference and merge/export.
v2.7.0 now adds quantized-base adapter fine-tuning:

```txt
frozen 4-bit base weights + trainable LoRA adapters
```

### Added

- `QuantizedLinear4bit` for frozen 4-bit base projection storage.
- NF4-style 16-value codebook quantization.
- Packed 4-bit indices: two 4-bit values per `uint8` byte.
- Optional double quantization for row scales.
- `QLoRALinear` wrapper: quantized frozen base + trainable LoRA matrices.
- Automatic QLoRA injection through `peft.method: qlora`.
- QLoRA adapter-only checkpoint save/load using existing adapter format.
- QLoRA merge + unload into plain `nn.Linear` modules.
- QLoRA storage summary helper.
- `configs/apex1_tiny_qlora.yaml`.
- `configs/apex1_tiny_qlora_inference.yaml`.
- `scripts/finetune_qlora.py`.
- `examples/qlora_finetune_demo.py`.
- `tests/test_qlora.py`.
- `docs/35-qlora-4bit-finetuning.md`.

### Updated

- `apex/config.py`
  - Added QLoRA config fields to `PEFTConfig`.
  - Added validation for `method`, `quantization_bits`, `quant_type`, and `compute_dtype`.
  - Added `get_tiny_qlora_config()` preset.
- `apex/model/lora.py`
  - Extended LoRA system to support QLoRA modules.
  - Added 4-bit quantization helpers.
  - Updated merge/unload to support both LoRA and QLoRA.
- `apex/__init__.py`
  - Bumped version to `2.7.0`.
- `pyproject.toml`
  - Bumped project version to `2.7.0`.
  - Added QLoRA keywords.
- `README.md`
  - Added v2.7.0 feature overview, commands, project structure, and learning path.

### Verification Commands

```bash
pytest tests/test_lora_peft.py -v
pytest tests/test_lora_inference.py -v
pytest tests/test_qlora.py -v
python examples/qlora_finetune_demo.py
python scripts/finetune_qlora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --data data/samples/tiny_sft.jsonl \
  --output-dir outputs/qlora-test \
  --max-steps 10
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --adapter outputs/qlora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
python scripts/merge_lora.py \
  --config configs/apex1_tiny_qlora.yaml \
  --adapter outputs/qlora-test/adapter_final.pt \
  --output outputs/merged-apex-qlora.pt
```

### Notes

This is an educational implementation. It intentionally does not include custom
CUDA kernels, bitsandbytes integration, or paged optimizers yet.

## v2.6.0 — LoRA Adapter Inference & Merge

APEX-1 v2.6.0 completes the LoRA/PEFT workflow started in v2.5.0.

v2.5.0 proved that APEX-1 can train and save adapter-only checkpoints.
v2.6.0 adds the missing production-style path:

```txt
train adapter -> save adapter -> load adapter -> generate -> merge -> export
```

### Added

- `apex/model/lora_inference.py`
  - safe base-checkpoint-first loading
  - automatic LoRA adapter injection
  - adapter checkpoint loading for inference
  - merge-and-unload helper
  - merged checkpoint export helper

- `scripts/generate_with_lora.py`
  - generate text with a saved LoRA adapter
  - optional base checkpoint support
  - optional runtime merge before generation

- `scripts/merge_lora.py`
  - merge LoRA adapter weights into base model weights
  - unload LoRA wrappers by default
  - export plain APEX checkpoint compatible with normal generation

- `examples/lora_generation_demo.py`
  - CPU-friendly adapter lifecycle demo

- `tests/test_lora_inference.py`
  - adapter inference load test
  - runtime merge test
  - merge-and-unload test
  - plain checkpoint compatibility test
  - helper workflow test

- `docs/34-lora-inference-and-merge.md`
  - full lesson explaining adapter inference and merge

- `configs/apex1_tiny_lora_inference.yaml`
  - CPU-friendly config for adapter inference demos

### Updated

- `apex/model/lora.py`
  - `count_lora_modules`
  - `has_lora_adapters`
  - `require_lora_adapters`
  - `peft_config_to_dict`
  - safer `torch.load` compatibility helper
  - `merge_and_unload_lora_weights`
  - `save_merged_lora_checkpoint`
  - adapter metadata now includes version and module count

- `apex/__init__.py`
  - version bumped to `2.6.0`

- `pyproject.toml`
  - version bumped to `2.6.0`
  - metadata updated for adapter inference and merge

### Commands

Generate with adapter:

```bash
python scripts/generate_with_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --prompt "Explain Rust ownership simply" \
  --max-tokens 64
```

Merge adapter:

```bash
python scripts/merge_lora.py \
  --config configs/apex1_tiny_lora.yaml \
  --adapter outputs/lora-test/adapter_final.pt \
  --output outputs/merged-apex-lora.pt
```

Use merged checkpoint:

```bash
python scripts/generate.py \
  --config configs/apex1_tiny.yaml \
  --checkpoint outputs/merged-apex-lora.pt \
  --prompt "Hello"
```

Run tests:

```bash
pytest tests/test_lora_peft.py tests/test_lora_inference.py -v
```


## v2.5.0

### Added

- Native LoRA implementation in `apex/model/lora.py`
- `PEFTConfig` in `apex/config.py`
- Automatic LoRA injection inside `APEX1Model`
- Adapter-only saving and loading
- LoRA merge/unmerge helpers for inference/export
- PEFT SFT trainer in `apex/training/peft.py`
- Fine-tuning CLI: `scripts/finetune_lora.py`
- CPU-friendly LoRA demo: `examples/lora_finetune_demo.py`
- Tiny LoRA config: `configs/apex1_tiny_lora.yaml`
- Tests: `tests/test_lora_peft.py`
- New guide: `docs/33-lora-peft-finetuning.md`

### Changed

- Version bumped to `2.5.0`
- `APEX1Model` now supports optional PEFT adapter injection
- README can now document LoRA fine-tuning commands and workflow

### Why

APEX-1 already had pretraining, SFT, alignment, vision, and evaluation. LoRA/PEFT makes it practical to fine-tune APEX-1 on small custom datasets without training the full model.


## [2.4.0] — Evaluation, Benchmarking, and Model Inspector

### Added

- Added `apex/eval/` package for course-friendly model evaluation:
  - `metrics.py` for token accuracy and token cross-entropy.
  - `perplexity.py` for next-token language-model perplexity.
  - `generation_quality.py` for distinct-n, average length, and repetition checks.
  - `vision_eval.py` for validating APEX-1 vision forward outputs.
  - `benchmark.py` for tiny forward-pass benchmark helpers.
- Added model inspection utilities:
  - `apex/utils/model_inspector.py`
  - `scripts/inspect_model.py`
- Added architecture diagram utilities:
  - `apex/utils/architecture_diagram.py`
  - `scripts/print_architecture.py`
- Added benchmark CLI:
  - `scripts/benchmark.py`
- Added mini dataset examples:
  - `data/samples/tiny_text.jsonl`
  - `data/samples/tiny_sft.jsonl`
  - `data/samples/tiny_preference.jsonl`
  - `data/samples/tiny_vision.jsonl`
- Added new examples:
  - `examples/eval_demo.py`
  - `examples/benchmark_demo.py`
  - `examples/inspect_model_demo.py`
  - `examples/architecture_diagram_demo.py`
  - `examples/tiny_dataset_demo.py`
- Added `tests/test_eval_and_inspector.py` with 10 new tests.
- Added `docs/33-evaluation-benchmarking-inspection.md`.

### Changed

- Updated `apex/__init__.py` version to `2.4.0`.
- README should now describe APEX-1 as a model-building and model-evaluation course.
- Course count moves from 32 to 33 lessons.
- Test count moves from 96 to 106 tests after the new evaluation/inspector test file passes.

### Notes

v2.4.0 does not add heavier model architecture. Instead, it makes APEX-1 easier
to understand, measure, benchmark, inspect, and teach. This is the right next
step after v2.3.0 vision because learners need to know whether the model is
working before adding more complex features.


## [2.3.0] — Vision Preview

### Added

- Added `VisionConfig` to `apex/config.py`.
- Added `apex.vision` package:
  - `ImagePreprocessor`
  - `NativeVisionEncoder`
  - `VisionToTextProjector`
  - `PerceiverResampler`
- Added `APEX1VisionModel`, a decoder-only vision-language wrapper that inserts image-derived visual tokens at the `<|img|>` placeholder.
- Added `VisionInstructionDataset` and `collate_vision_batch` for image-question-answer JSONL data.
- Added `expand_labels_for_visual_tokens` and `compute_vision_sft_loss` for multimodal SFT.
- Added `configs/apex1_tiny_vision.yaml`.
- Added `examples/vision_forward_demo.py` and `examples/vision_chat_demo.py`.
- Added `tests/test_vision.py`.
- Added `docs/32-vision-capabilities.md`.

### Notes

- v2.3.0 adds the architecture and training path for image understanding.
- Real image understanding requires training or a future adapter to a pretrained vision encoder such as CLIP/SigLIP/DINOv2.
- Existing `APEX1Model` text-only behavior is unchanged.

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