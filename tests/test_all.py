"""Unit tests for all APEX-1 core modules."""

import pytest
import torch

from apex.config import APEXConfig, get_tiny_config
from apex.generation.sampler import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)
from apex.model.apex_model import APEX1Model
from apex.model.attention import GQASlidingWindowAttention, MLAAttention
from apex.model.block import APEXTransformerBlock
from apex.model.ffn import DenseFFN, MoEFFN
from apex.model.load_balancer import LoadBalancer
from apex.model.mask import build_apex_attention_mask, is_global_layer
from apex.model.multi_token_head import MultiTokenHead
from apex.model.norm import RMSNorm
from apex.model.rope import apply_rope, apply_yarn_scaling, precompute_rope_cache, rotate_half
from apex.model.skip_gate import SkipGate
from apex.training.checkpoint import load_checkpoint, save_checkpoint
from apex.training.losses import compute_pretrain_loss, compute_sft_loss
from apex.training.scheduler import CosineWarmupScheduler, get_lr

CFG = get_tiny_config()


# ── RMSNorm ──────────────────────────────────────────────────────────────────
class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == (2, 8, 64)

    def test_unit_scale_identity(self):
        norm = RMSNorm(64)
        x = torch.ones(1, 1, 64)
        out = norm(x)
        assert out.shape == (1, 1, 64)
        assert torch.allclose(out, x, atol=1e-4)

    def test_zero_input(self):
        norm = RMSNorm(64)
        x = torch.zeros(1, 1, 64)
        out = norm(x)
        assert not torch.isnan(out).any()


# ── RoPE ─────────────────────────────────────────────────────────────────────
class TestRoPE:
    def test_cache_shape(self):
        cos, sin = precompute_rope_cache(64, 128)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 8, 64)
        assert rotate_half(x).shape == x.shape

    def test_orthogonality(self):
        cos, sin = precompute_rope_cache(16, 32)
        q = torch.randn(1, 2, 4, 16)
        k = torch.randn(1, 2, 4, 16)
        pos = torch.arange(4)
        qr, kr = apply_rope(q, k, cos, sin, pos)
        q_norm = q.norm(dim=-1)
        qr_norm = qr.norm(dim=-1)
        assert torch.allclose(q_norm, qr_norm, atol=1e-4)

    def test_periodicity(self):
        cos, sin = precompute_rope_cache(16, 1024, rope_base=10.0)
        assert cos.shape == (1024, 16)

    def test_yarn_scaling(self):
        theta = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
        scaled, factor = apply_yarn_scaling(theta, 4.0, 16)
        assert factor > 1.0
        assert scaled.shape == theta.shape

    def test_yarn_no_scaling(self):
        theta = torch.ones(8)
        scaled, factor = apply_yarn_scaling(theta, 1.0, 16)
        assert factor == 1.0
        assert torch.allclose(scaled, theta)


# ── Attention Masks ──────────────────────────────────────────────────────────
class TestMask:
    def test_global_mask_shape(self):
        mask = build_apex_attention_mask(4, 8, 512, is_global_layer=True)
        assert mask.shape == (8, 8)

    def test_prefix_bidirectional(self):
        mask = build_apex_attention_mask(4, 8, 512, is_global_layer=True)
        assert mask[:4, :4].all()

    def test_causal_generation(self):
        mask = build_apex_attention_mask(4, 8, 512, is_global_layer=True)
        assert mask[5, 6] == False
        assert mask[5, 5] == True

    def test_sliding_window(self):
        mask = build_apex_attention_mask(0, 16, 4, is_global_layer=False)
        assert mask[10, 10] == True
        assert mask[10, 6] == False

    def test_is_global_layer(self):
        assert is_global_layer(5, 6) == True
        assert is_global_layer(0, 6) == False
        assert is_global_layer(11, 6) == True
        assert is_global_layer(3, 6) == False


# ── SkipGate ─────────────────────────────────────────────────────────────────
class TestSkipGate:
    def test_output_shape(self):
        gate = SkipGate(64, hidden_dim=16)
        x = torch.randn(2, 8, 64)
        assert gate(x).shape == (2, 8, 1)

    def test_output_range(self):
        gate = SkipGate(64)
        x = torch.randn(2, 8, 64)
        out = gate(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_skip_mask(self):
        gate = SkipGate(64, threshold=0.5)
        x = torch.randn(2, 8, 64)
        mask = gate.get_skip_mask(x)
        assert mask.dtype == torch.bool


# ── DenseFFN ─────────────────────────────────────────────────────────────────
class TestDenseFFN:
    def test_output_shape(self):
        ffn = DenseFFN(CFG)
        x = torch.randn(2, 8, CFG.model.d_model)
        assert ffn(x).shape == x.shape


# ── MoEFFN ───────────────────────────────────────────────────────────────────
class TestMoEFFN:
    def test_output_shape(self):
        ffn = MoEFFN(CFG)
        x = torch.randn(2, 4, CFG.model.d_model)
        assert ffn(x).shape == x.shape

    def test_routing_indices(self):
        ffn = MoEFFN(CFG)
        x = torch.randn(2, 4, CFG.model.d_model)
        ffn(x)
        idx = ffn.get_last_routing_indices()
        assert idx is not None
        assert idx.shape[1] == CFG.moe.n_active

    def test_expert_bias_update(self):
        ffn = MoEFFN(CFG)
        new_bias = torch.ones(CFG.moe.n_experts) * 0.5
        ffn.set_expert_bias(new_bias)
        assert torch.allclose(ffn.expert_bias, new_bias)


# ── LoadBalancer ─────────────────────────────────────────────────────────────
class TestLoadBalancer:
    def test_init(self):
        lb = LoadBalancer(8)
        assert lb.bias.shape == (8,)
        assert (lb.bias == 0).all()

    def test_update(self):
        lb = LoadBalancer(4, alpha=0.1)
        idx = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        lb.update(idx)
        assert lb.bias[2] > 0  # underused
        assert lb.bias[0] < 0  # overused

    def test_clamp(self):
        lb = LoadBalancer(4, alpha=10.0)
        idx = torch.tensor([[0, 1]] * 100)
        for _ in range(100):
            lb.update(idx)
        assert lb.bias.max() <= 1.0
        assert lb.bias.min() >= -1.0

    def test_state_dict(self):
        lb = LoadBalancer(4)
        state = lb.state_dict()
        lb2 = LoadBalancer(4)
        lb2.load_state_dict(state)
        assert torch.allclose(lb.bias, lb2.bias)


# ── Attention ────────────────────────────────────────────────────────────────
class TestAttention:
    def test_mla_output_shape(self):
        mla = MLAAttention(CFG)
        cos, sin = precompute_rope_cache(CFG.model.d_head_rope, 256)
        x = torch.randn(1, 8, CFG.model.d_model)
        pos = torch.arange(8)
        out, kv = mla(x, cos, sin, pos)
        assert out.shape == (1, 8, CFG.model.d_model)

    def test_mla_kv_cache_growth(self):
        mla = MLAAttention(CFG)
        cos, sin = precompute_rope_cache(CFG.model.d_head_rope, 256)
        x1 = torch.randn(1, 4, CFG.model.d_model)
        _, kv1 = mla(x1, cos, sin, torch.arange(4))
        # BUG-01 FIX: kv_cache is now a tuple (c_kv, K_rope), access c_kv
        assert kv1[0].shape[1] == 4
        x2 = torch.randn(1, 1, CFG.model.d_model)
        _, kv2 = mla(x2, cos, sin, torch.arange(4, 5), kv_cache=kv1)
        assert kv2[0].shape[1] == 5

    def test_gqa_output_shape(self):
        gqa = GQASlidingWindowAttention(CFG)
        cos, sin = precompute_rope_cache(CFG.model.d_head, 256)
        x = torch.randn(1, 8, CFG.model.d_model)
        pos = torch.arange(8)
        out, kv = gqa(x, cos, sin, pos)
        assert out.shape == (1, 8, CFG.model.d_model)


# ── TransformerBlock ─────────────────────────────────────────────────────────
class TestBlock:
    def test_local_block(self):
        block = APEXTransformerBlock(0, CFG)
        assert not block.is_global
        cos, sin = precompute_rope_cache(CFG.model.d_head, 256)
        x = torch.randn(1, 8, CFG.model.d_model)
        out, _ = block(x, cos, sin, torch.arange(8))
        assert out.shape == x.shape

    def test_global_block(self):
        block = APEXTransformerBlock(5, CFG)
        assert block.is_global
        cos, sin = precompute_rope_cache(CFG.model.d_head_rope, 256)
        x = torch.randn(1, 8, CFG.model.d_model)
        out, _ = block(x, cos, sin, torch.arange(8))
        assert out.shape == x.shape


# ── MultiTokenHead ───────────────────────────────────────────────────────────
class TestMultiTokenHead:
    def test_output(self):
        head = MultiTokenHead(64, 1000, n_predict=4)
        h = torch.randn(1, 8, 64)
        logits = head(h)
        assert len(logits) == 4
        assert logits[0].shape == (1, 8, 1000)

    def test_draft(self):
        head = MultiTokenHead(64, 1000, n_predict=4)
        h = torch.randn(1, 1, 64)
        drafts = head.draft_tokens(h)
        assert drafts.shape == (1, 4)


# ── Full Model ───────────────────────────────────────────────────────────────
class TestAPEX1Model:
    def test_forward_shape(self):
        model = APEX1Model(CFG)
        x = torch.randint(0, CFG.model.vocab_size, (1, 16))
        out = model(x)
        assert out["logits"].shape == (1, 16, CFG.model.vocab_size)

    def test_weight_tying(self):
        model = APEX1Model(CFG)
        assert model.embedding.weight.data_ptr() == model.embedding.weight.data_ptr()
        logits = torch.matmul(
            torch.randn(1, 1, CFG.model.d_model),
            model.embedding.weight.T,
        )
        assert logits.shape[-1] == CFG.model.vocab_size

    def test_kv_caches_count(self):
        model = APEX1Model(CFG)
        x = torch.randint(0, CFG.model.vocab_size, (1, 8))
        out = model(x)
        assert len(out["kv_caches"]) == CFG.model.n_layers

    def test_spec_logits(self):
        model = APEX1Model(CFG)
        x = torch.randint(0, CFG.model.vocab_size, (1, 8))
        out = model(x)
        assert out["spec_logits"] is not None
        assert len(out["spec_logits"]) == CFG.multi_token_head.n_predict

    def test_param_counts(self):
        model = APEX1Model(CFG)
        assert model.total_parameters() > 0
        assert model.active_parameters() > 0
        assert model.active_parameters() <= model.total_parameters()


# ── Sampler ──────────────────────────────────────────────────────────────────
class TestSampler:
    def test_temperature(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        scaled = apply_temperature(logits, 0.5)
        assert torch.allclose(scaled, logits / 0.5)

    def test_top_k(self):
        logits = torch.randn(100)
        filtered = apply_top_k(logits, 5)
        assert (filtered > float("-inf")).sum() == 5

    def test_top_p(self):
        logits = torch.randn(100)
        filtered = apply_top_p(logits, 0.5)
        assert (filtered > float("-inf")).sum() < 100

    def test_repetition_penalty(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        penalized = apply_repetition_penalty(logits, [0, 1], 1.2)
        assert penalized[0] < logits[0]

    def test_sample(self):
        logits = torch.randn(1000)
        token = sample_next_token(logits, temperature=1.0)
        assert 0 <= token.item() < 1000


# ── Scheduler ────────────────────────────────────────────────────────────────
class TestScheduler:
    def test_warmup(self):
        lr = get_lr(0, 100, 1000, 1e-3)
        assert lr == 0.0
        lr = get_lr(50, 100, 1000, 1e-3)
        assert abs(lr - 5e-4) < 1e-8

    def test_peak(self):
        lr = get_lr(100, 100, 1000, 1e-3)
        assert abs(lr - 1e-3) < 1e-8

    def test_decay(self):
        lr_mid = get_lr(550, 100, 1000, 1e-3)
        lr_end = get_lr(999, 100, 1000, 1e-3)
        assert lr_mid > lr_end


# ── Losses ───────────────────────────────────────────────────────────────────
class TestLosses:
    def test_pretrain_loss(self):
        logits = torch.randn(2, 16, 1000)
        spec = [torch.randn(2, 16, 1000) for _ in range(4)]
        ids = torch.randint(0, 1000, (2, 16))
        loss, metrics = compute_pretrain_loss(logits, spec, ids, 1000)
        assert loss.item() > 0
        assert "loss_main" in metrics

    def test_sft_loss(self):
        logits = torch.randn(2, 16, 1000)
        ids = torch.randint(0, 1000, (2, 16))
        types = torch.zeros(2, 16, dtype=torch.long)
        types[:, 8:] = 2
        loss, metrics = compute_sft_loss(logits, ids, types, 1000)
        assert loss.item() > 0


# ── Checkpoint ───────────────────────────────────────────────────────────────
class TestCheckpoint:
    def test_save_load(self, tmp_path):
        model = APEX1Model(CFG)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        path = tmp_path / "test.pt"
        save_checkpoint(path, model, opt, step=42, loss=1.5)
        model2 = APEX1Model(CFG)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
        info = load_checkpoint(path, model2, opt2)
        assert info["step"] == 42
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


# ── Config ───────────────────────────────────────────────────────────────────
class TestConfig:
    def test_tiny_config(self):
        cfg = get_tiny_config()
        cfg.validate()

    def test_yaml_roundtrip(self, tmp_path):
        cfg = get_tiny_config()
        path = tmp_path / "test.yaml"
        cfg.to_yaml(path)
        cfg2 = APEXConfig.from_yaml(path)
        assert cfg2.model.d_model == cfg.model.d_model
        assert cfg2.moe.n_experts == cfg.moe.n_experts
