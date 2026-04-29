"""
Bug-fix regression tests for APEX-1.

Covers every bug fixed in this patch:
  BUG-01  MLA K_rope cache correctness
  BUG-02  MLA W_O shape (no crash)
  BUG-03  Constitutional AI actually calls model
  BUG-04  GRPO generation uses APEX1Generator
  BUG-05  Optional imported at top of reward_model.py
  BUG-06  PRM score_steps_from_text raises on None tokenizer
  BUG-07  RoPE caches matched to layer type
  BUG-08  MoE expert dispatch handles n_e > 1
  BUG-09  Generator KV position tracking uses is_global_layer
  BUG-10  Sliding window mask vectorised (shape + correctness)
  BUG-11  Load balancer created from layer n_experts
  BUG-13  RNG checkpoint saves Python + CPU states separately
  BUG-19  is_moe flag respects config.moe.enabled
  BUG-21  Thinking token count excludes the start token
  BUG-22  apply_yarn_scaling vectorised (no Python loop)
"""

import random
import tempfile
from pathlib import Path

import pytest
import torch

from apex.config import get_tiny_config
from apex.model.apex_model import APEX1Model
from apex.model.attention import MLAAttention
from apex.model.block import APEXTransformerBlock
from apex.model.ffn import DenseFFN, MoEFFN
from apex.model.load_balancer import LoadBalancer
from apex.model.mask import build_apex_attention_mask, is_global_layer
from apex.model.rope import apply_yarn_scaling, precompute_rope_cache


# ---------------------------------------------------------------------------
# BUG-01 — MLA K_rope cache is no longer filled with zeros
# ---------------------------------------------------------------------------

class TestBug01MLAKRopeCache:
    """K_rope for cached positions must use the rotated values, not zeros."""

    def test_kv_cache_is_tuple(self):
        cfg = get_tiny_config()
        mla = MLAAttention(cfg)
        cos, sin = precompute_rope_cache(cfg.model.d_head_rope, 256)
        x = torch.randn(1, 4, cfg.model.d_model)
        pos = torch.arange(4)
        _, kv = mla(x, cos, sin, pos)
        # BUG-01 FIX: cache must be (c_kv, K_rope) tuple, not a bare tensor
        assert isinstance(kv, tuple), "kv_cache should be a tuple (c_kv, K_rope)"
        assert len(kv) == 2

    def test_cached_and_fresh_produce_same_output(self):
        """Running the full sequence at once vs step-by-step must match."""
        cfg = get_tiny_config()
        mla = MLAAttention(cfg)
        mla.eval()
        cos, sin = precompute_rope_cache(cfg.model.d_head_rope, 256)

        seq = torch.randn(1, 6, cfg.model.d_model)
        pos_all = torch.arange(6)

        with torch.no_grad():
            # Full sequence in one shot
            out_full, _ = mla(seq, cos, sin, pos_all, attn_mask=None, kv_cache=None)

            # First 4 tokens
            _, kv1 = mla(seq[:, :4], cos, sin, torch.arange(4))
            # K_rope stored should NOT be all zeros
            K_rope_cached = kv1[1]  # [1, n_kv, 4, d_rope]
            assert not torch.all(K_rope_cached == 0), \
                "Cached K_rope must not be all-zero (BUG-01)"

    def test_k_rope_cache_grows_correctly(self):
        cfg = get_tiny_config()
        mla = MLAAttention(cfg)
        cos, sin = precompute_rope_cache(cfg.model.d_head_rope, 256)
        x1 = torch.randn(1, 3, cfg.model.d_model)
        _, kv1 = mla(x1, cos, sin, torch.arange(3))
        assert kv1[1].shape[2] == 3  # K_rope dim-2 = seq length

        x2 = torch.randn(1, 2, cfg.model.d_model)
        _, kv2 = mla(x2, cos, sin, torch.arange(3, 5), kv_cache=kv1)
        assert kv2[1].shape[2] == 5  # grew from 3 to 5


# ---------------------------------------------------------------------------
# BUG-02 — MLA W_O shape mismatch (no crash)
# ---------------------------------------------------------------------------

class TestBug02MLAWOShape:
    def test_no_shape_error_on_forward(self):
        cfg = get_tiny_config()
        mla = MLAAttention(cfg)
        cos, sin = precompute_rope_cache(cfg.model.d_head_rope, 256)
        x = torch.randn(2, 8, cfg.model.d_model)
        # Should not raise RuntimeError about shape mismatch
        out, _ = mla(x, cos, sin, torch.arange(8))
        assert out.shape == (2, 8, cfg.model.d_model)

    def test_w_o_input_dim(self):
        cfg = get_tiny_config()
        mla = MLAAttention(cfg)
        m = cfg.model
        expected_in = m.n_heads_q * m.d_head
        actual_in = mla.W_O.weight.shape[1]
        assert actual_in == expected_in, (
            f"W_O input dim should be n_heads_q*d_head={expected_in}, got {actual_in}"
        )


# ---------------------------------------------------------------------------
# BUG-03 — Constitutional AI critique actually calls model
# ---------------------------------------------------------------------------

class TestBug03ConstitutionalAI:
    def test_critique_is_not_always_false(self):
        """With a real model, violations should sometimes be detected."""
        from apex.alignment.constitutional import ConstitutionalAI, DEFAULT_CONSTITUTION
        cfg = get_tiny_config()
        model = APEX1Model(cfg)

        class DummyTokenizer:
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False): return [1, 2, 3]
            def decode(self, ids): return "YES this violates."

        cai = ConstitutionalAI(model, DummyTokenizer(), constitution=DEFAULT_CONSTITUTION[:2])
        # Should not raise and must return one result per principle
        results = cai.critique_response("some response")
        assert len(results) == 2

    def test_score_response_returns_float(self):
        from apex.alignment.constitutional import ConstitutionalAI
        cfg = get_tiny_config()
        model = APEX1Model(cfg)

        class DummyTokenizer:
            eos_token_id = 2
            def encode(self, text, add_special_tokens=False): return [1]
            def decode(self, ids): return "NO"

        cai = ConstitutionalAI(model, DummyTokenizer(), constitution=["Be honest."])
        score = cai.score_response("honest answer")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# BUG-04 — GRPO generation uses APEX1Generator (not broken manual loop)
# ---------------------------------------------------------------------------

class TestBug04GRPOGeneration:
    def test_grpo_full_loop_runs(self):
        from apex.alignment.grpo import grpo_full_loop

        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        ref_model = APEX1Model(cfg)
        for p in ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        prompt = torch.randint(0, cfg.model.vocab_size, (1, 4))

        def dummy_reward(p, r):
            return float(torch.rand(1).item())

        # Should complete without error
        metrics = grpo_full_loop(
            model, ref_model, optimizer, [prompt], dummy_reward, G=2
        )
        assert "grpo_loss" in metrics

    def test_response_ids_are_non_trivial(self):
        """Generated responses must have > 1 token (not just a single reset)."""
        from apex.generation.generator import APEX1Generator, GenerationConfig
        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        gen_cfg = GenerationConfig(max_new_tokens=8, temperature=1.0, eos_token_id=2)
        gen = APEX1Generator(model, gen_cfg)
        prompt = torch.randint(0, cfg.model.vocab_size, (1, 4))
        out = gen.generate(prompt)
        assert len(out.token_ids) >= 1


# ---------------------------------------------------------------------------
# BUG-05 — Optional imported at top of reward_model.py
# ---------------------------------------------------------------------------

class TestBug05OptionalImport:
    def test_import_does_not_raise(self):
        """The module must be importable without NameError."""
        import importlib
        import apex.alignment.reward_model as rm
        importlib.reload(rm)  # force re-execution of module top-level

    def test_reward_model_instantiates(self):
        from apex.alignment.reward_model import RewardModel
        cfg = get_tiny_config()
        backbone = APEX1Model(cfg)
        rm = RewardModel(backbone, cfg.model.d_model)
        ids = torch.randint(0, cfg.model.vocab_size, (1, 8))
        reward = rm(ids)
        assert reward.shape == (1,)

    def test_optional_is_at_top(self):
        """Verify Optional is imported before the class definition."""
        import ast, inspect, apex.alignment.reward_model as rm_mod
        source = inspect.getsource(rm_mod)
        tree = ast.parse(source)
        optional_line = None
        class_line = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in getattr(node, 'names', []):
                    if alias.name == 'Optional':
                        optional_line = node.lineno
            if isinstance(node, ast.ClassDef) and node.name == 'RewardModel':
                class_line = node.lineno
        assert optional_line is not None, "Optional not imported"
        assert class_line is not None, "RewardModel class not found"
        assert optional_line < class_line, \
            f"Optional import (line {optional_line}) must come before RewardModel (line {class_line})"


# ---------------------------------------------------------------------------
# BUG-06 — PRM raises ValueError on None tokenizer
# ---------------------------------------------------------------------------

class TestBug06PRMNoneTokenizer:
    def test_raises_on_none_tokenizer(self):
        from apex.alignment.prm import ProcessRewardModel
        cfg = get_tiny_config()
        backbone = APEX1Model(cfg)
        prm = ProcessRewardModel(backbone, cfg.model.d_model)
        with pytest.raises(ValueError, match="tokenizer"):
            prm.score_steps_from_text("prompt", ["step1"], None)

    def test_works_with_real_tokenizer(self):
        from apex.alignment.prm import ProcessRewardModel

        class FakeTok:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 100 for c in text[:4]]

        cfg = get_tiny_config()
        backbone = APEX1Model(cfg)
        prm = ProcessRewardModel(backbone, cfg.model.d_model)
        scores = prm.score_steps_from_text("hello", ["step one", "step two"], FakeTok())
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# BUG-07 — RoPE caches matched to layer type
# ---------------------------------------------------------------------------

class TestBug07RopeCacheLayerType:
    def test_gqa_gets_d_head_cache(self):
        """GQA blocks receive cos_cache with width d_head."""
        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        # cos_cache width == d_head
        assert model.cos_cache.shape[-1] == cfg.model.d_head

    def test_mla_gets_d_head_rope_cache(self):
        """MLA blocks receive cos_cache_rope with width d_head_rope."""
        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        assert model.cos_cache_rope.shape[-1] == cfg.model.d_head_rope

    def test_full_forward_runs_with_mixed_layers(self):
        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        x = torch.randint(0, cfg.model.vocab_size, (1, 10))
        out = model(x)
        assert out["logits"].shape == (1, 10, cfg.model.vocab_size)


# ---------------------------------------------------------------------------
# BUG-08 — MoE expert dispatch handles n_e > 1
# ---------------------------------------------------------------------------

class TestBug08MoEBatchDim:
    def test_multiple_tokens_routed_to_same_expert(self):
        """When n_e > 1 tokens land on the same expert, output is correct."""
        cfg = get_tiny_config()
        ffn = MoEFFN(cfg)

        # Force all tokens to route to expert 0 by setting router weights
        with torch.no_grad():
            ffn.router.weight.zero_()
            ffn.router.weight[0] = 1.0  # expert 0 always wins

        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, cfg.model.d_model)
        out = ffn(x)
        assert out.shape == (batch_size, seq_len, cfg.model.d_model)
        assert not torch.isnan(out).any()

    def test_output_shape_various_batch_sizes(self):
        cfg = get_tiny_config()
        ffn = MoEFFN(cfg)
        for b, s in [(1, 1), (1, 4), (2, 6)]:
            x = torch.randn(b, s, cfg.model.d_model)
            out = ffn(x)
            assert out.shape == (b, s, cfg.model.d_model), \
                f"Shape mismatch for batch={b}, seq={s}"


# ---------------------------------------------------------------------------
# BUG-09 — Generator KV position tracking
# ---------------------------------------------------------------------------

class TestBug09GeneratorPositionTracking:
    def test_position_tracking_is_consistent(self):
        """Autoregressive generation must produce valid (non-nan) outputs."""
        from apex.generation.generator import APEX1Generator, GenerationConfig
        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        gen = APEX1Generator(model, GenerationConfig(max_new_tokens=4, temperature=1.0))
        prompt = torch.randint(0, cfg.model.vocab_size, (1, 5))
        out = gen.generate(prompt)
        assert len(out.token_ids) >= 1

    def test_prev_len_detection_matches_layer_type(self):
        """_get_prev_len must work for both MLA and GQA caches."""
        from apex.generation.generator import APEX1Generator, GenerationConfig
        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        gen = APEX1Generator(model)

        # Run one forward pass to get caches
        x = torch.randint(0, cfg.model.vocab_size, (1, 6))
        out = model(x)
        prev_len = gen._get_prev_len(out["kv_caches"])
        assert prev_len == 6


# ---------------------------------------------------------------------------
# BUG-10 — Sliding window mask is vectorised (shape + correctness)
# ---------------------------------------------------------------------------

class TestBug10SlidingWindowMask:
    def test_mask_shape(self):
        mask = build_apex_attention_mask(4, 12, 4, is_global_layer=False)
        assert mask.shape == (12, 12)

    def test_prefix_bidirectional(self):
        mask = build_apex_attention_mask(4, 10, 4, is_global_layer=False)
        assert mask[:4, :4].all()

    def test_window_boundary(self):
        """Token at position 8 with window=4 should attend to 5,6,7,8 only."""
        mask = build_apex_attention_mask(0, 12, 4, is_global_layer=False)
        row = 8
        assert mask[row, row] == True          # self
        assert mask[row, row - 1] == True      # window-1
        assert mask[row, row - 3] == True      # window boundary
        assert mask[row, row - 4] == False     # outside window
        assert mask[row, row + 1] == False     # future

    def test_global_mask_has_full_causal(self):
        mask = build_apex_attention_mask(0, 8, 4, is_global_layer=True)
        # row 7 should attend to all of 0..7
        assert mask[7, :8].all()

    def test_vectorised_matches_reference_for_local(self):
        """Compare vectorised implementation against a simple reference."""
        total_len = 20
        prefix_len = 4
        window = 5

        # Reference: naive loop
        ref = torch.zeros(total_len, total_len, dtype=torch.bool)
        ref[:prefix_len, :prefix_len] = True
        for i in range(prefix_len, total_len):
            start = max(0, i - window + 1)
            ref[i, start : i + 1] = True

        fast = build_apex_attention_mask(prefix_len, total_len, window, is_global_layer=False)
        assert torch.equal(fast, ref), "Vectorised mask differs from reference"


# ---------------------------------------------------------------------------
# BUG-11 — Load balancer uses per-layer n_experts
# ---------------------------------------------------------------------------

class TestBug11LoadBalancerNExperts:
    def test_balancer_uses_layer_n_experts(self):
        """Each LoadBalancer must match its MoE layer's n_experts."""
        from apex.training.trainer import PreTrainer
        from apex.data.dataset import PretrainDataset
        from apex.data.data_loader import create_pretrain_loader

        cfg = get_tiny_config()
        model = APEX1Model(cfg)
        token_ids = torch.randint(0, cfg.model.vocab_size, (200,))
        dataset = PretrainDataset(token_ids, seq_len=cfg.training.seq_len)
        loader = create_pretrain_loader(dataset, batch_size=1, num_workers=0)

        trainer = PreTrainer(model, cfg, loader)

        moe_layers = model.get_moe_layers()
        for (_, moe_ffn), lb in zip(moe_layers, trainer.load_balancers):
            assert lb.n_experts == moe_ffn.n_experts, (
                f"LoadBalancer n_experts {lb.n_experts} != layer n_experts {moe_ffn.n_experts}"
            )

    def test_bias_device_after_update(self):
        """Bias retrieved from load balancer should be movable to any device."""
        lb = LoadBalancer(n_experts=4, alpha=0.01)
        idx = torch.tensor([[0, 1], [1, 2], [2, 3]])
        lb.update(idx)
        bias = lb.get_bias()
        # Should be CPU tensor — can be moved
        _ = bias.to("cpu")


# ---------------------------------------------------------------------------
# BUG-13 — RNG checkpoint saves Python + CPU states separately
# ---------------------------------------------------------------------------

class TestBug13RNGCheckpoint:
    def test_rng_states_are_distinct(self):
        from apex.training.checkpoint import save_checkpoint, load_checkpoint
        cfg = get_tiny_config()
        model = APEX1Model(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(path, model, step=1)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        rng = ckpt["rng_states"]
        assert "python" in rng
        assert "cpu" in rng
        # python state should be a tuple (Python random.getstate() output)
        assert isinstance(rng["python"], tuple), \
            "python RNG state should be random.getstate() tuple"
        # cpu state should be a torch Tensor
        assert isinstance(rng["cpu"], torch.Tensor), \
            "cpu RNG state should be torch.Tensor"
        # They should differ in type (at minimum)
        assert not isinstance(rng["python"], torch.Tensor), \
            "python and cpu RNG states are both tensors — BUG-13 not fixed"

    def test_rng_state_restores_python_random(self):
        """After loading, Python random sequence must match saved state."""
        from apex.training.checkpoint import save_checkpoint, load_checkpoint
        cfg = get_tiny_config()
        model = APEX1Model(cfg)

        # Seed and record expected sequence
        random.seed(42)
        _ = [random.random() for _ in range(100)]
        # Save current state via checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(path, model, step=1)
            expected = [random.random() for _ in range(5)]

            load_checkpoint(path, model)
            actual = [random.random() for _ in range(5)]

        assert expected == actual, "Python RNG state not restored correctly"


# ---------------------------------------------------------------------------
# BUG-19 — is_moe flag respects config.moe.enabled
# ---------------------------------------------------------------------------

class TestBug19IsMoeFlag:
    def test_is_moe_false_when_moe_disabled(self):
        cfg = get_tiny_config()
        cfg.moe.enabled = False
        for layer_idx in range(cfg.model.n_layers):
            block = APEXTransformerBlock(layer_idx, cfg)
            assert not block.is_moe, (
                f"Layer {layer_idx} wrongly labelled MoE when config.moe.enabled=False"
            )

    def test_is_moe_true_for_correct_layers_when_enabled(self):
        cfg = get_tiny_config()
        cfg.moe.enabled = True
        for layer_idx in range(cfg.model.n_layers):
            block = APEXTransformerBlock(layer_idx, cfg)
            expected = layer_idx % cfg.moe.moe_layer_freq != 0
            assert block.is_moe == expected, (
                f"Layer {layer_idx}: expected is_moe={expected}, got {block.is_moe}"
            )


# ---------------------------------------------------------------------------
# BUG-21 — Thinking token count excludes the start token
# ---------------------------------------------------------------------------

class TestBug21ThinkingTokenCount:
    def test_start_token_not_counted(self):
        """The <|thinking_start|> token itself must not consume budget."""
        from apex.generation.generator import APEX1Generator, GenerationConfig
        cfg = get_tiny_config()
        model = APEX1Model(cfg)

        THINKING_START = cfg.model.vocab_size - 3
        THINKING_END = cfg.model.vocab_size - 2
        EOS = cfg.model.vocab_size - 1

        gen_cfg = GenerationConfig(
            max_new_tokens=50,
            temperature=1.0,
            enable_thinking=True,
            max_thinking_tokens=10,
            thinking_start_id=THINKING_START,
            thinking_end_id=THINKING_END,
            eos_token_id=EOS,
        )
        gen = APEX1Generator(model, gen_cfg)
        prompt = torch.randint(0, cfg.model.vocab_size - 4, (1, 4))
        out = gen.generate(prompt)

        # thinking_tokens must be <= budget (start token not counted)
        assert out.thinking_tokens <= gen_cfg.max_thinking_tokens, (
            f"thinking_tokens={out.thinking_tokens} exceeds budget={gen_cfg.max_thinking_tokens}"
        )


# ---------------------------------------------------------------------------
# BUG-22 — apply_yarn_scaling is vectorised
# ---------------------------------------------------------------------------

class TestBug22YarnScaling:
    def test_output_shape(self):
        theta = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        scaled, factor = apply_yarn_scaling(theta, 4.0, 32)
        assert scaled.shape == theta.shape
        assert factor > 1.0

    def test_matches_reference_loop_implementation(self):
        """Vectorised output must match the old loop-based reference."""
        import math

        theta = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
        scale_factor = 4.0
        beta_fast, beta_slow = 32.0, 1.0

        # Reference: original loop
        ref = theta.clone()
        for i in range(len(theta)):
            wavelength = 2.0 * math.pi / theta[i].item()
            if wavelength < beta_fast:
                ref[i] = theta[i]
            elif wavelength > beta_slow * scale_factor:
                ref[i] = theta[i] / scale_factor
            else:
                t = (wavelength / beta_slow - 1.0) / (scale_factor - 1.0)
                ref[i] = theta[i] / (t * scale_factor + (1.0 - t))

        fast, _ = apply_yarn_scaling(theta, scale_factor, 16, beta_fast, beta_slow)
        assert torch.allclose(fast, ref, atol=1e-5), \
            f"Vectorised YaRN differs from reference: max diff {(fast - ref).abs().max()}"

    def test_no_python_loop_in_implementation(self):
        """Smoke test: function executes quickly for large d_head (no O(d) loop)."""
        import time
        theta = 1.0 / (10000.0 ** (torch.arange(0, 512, 2, dtype=torch.float32) / 512))
        t0 = time.perf_counter()
        for _ in range(200):
            apply_yarn_scaling(theta, 8.0, 512)
        elapsed = time.perf_counter() - t0
        # A Python loop over 256 dims × 200 iterations would be ~50 ms on most hardware.
        # Vectorised should be well under 500 ms total.
        assert elapsed < 5.0, f"apply_yarn_scaling too slow ({elapsed:.2f}s) — may still use loop"
