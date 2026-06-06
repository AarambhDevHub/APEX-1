"""
APEX-1 Model Configuration.

Defines the complete configuration dataclass for all APEX-1 model sizes,
loadable from YAML config files. Covers model dimensions, attention settings,
MoE parameters, skip gate, multi-token prediction, thinking mode, training
hyperparameters, GRPO alignment settings, optional vision settings, and
optional PEFT / LoRA / QLoRA / DoRA fine-tuning settings.

v2.5.0 adds ``PEFTConfig`` so APEX-1 can attach trainable LoRA adapters to a
frozen base model for efficient supervised fine-tuning. v2.7.0 extends PEFT
with an educational QLoRA-style 4-bit quantized base model path. v2.8.0 adds DoRA/QDoRA. v2.9.0 adds adapter-based DPO preference alignment. v2.8.0 adds DoRA/QDoRA adapters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Core model architecture dimensions."""

    d_model: int = 512
    n_layers: int = 12
    n_heads_q: int = 8
    n_heads_kv: int = 2
    d_head: int = 64
    d_kv_compressed: int = 64
    d_q_compressed: int = 96
    d_head_rope: int = 32
    d_ffn: int = 1376
    vocab_size: int = 151643
    max_seq_len: int = 8192
    rope_base: float = 10000.0
    rope_scaling: float = 1.0
    dropout: float = 0.0


@dataclass
class AttentionConfig:
    """Attention strategy parameters."""

    global_layer_freq: int = 6
    local_window: int = 512
    flash: bool = True


@dataclass
class MoEConfig:
    """Mixture of Experts parameters."""

    enabled: bool = True
    n_experts: int = 8
    n_active: int = 2
    n_shared: int = 1
    moe_layer_freq: int = 2
    balancer_alpha: float = 0.001


@dataclass
class SkipGateConfig:
    """Dynamic skip gate parameters."""

    enabled: bool = True
    hidden_dim: int = 64
    threshold: float = 0.15


@dataclass
class MultiTokenHeadConfig:
    """Multi-token prediction head parameters."""

    enabled: bool = True
    n_predict: int = 4
    lambda_spec: float = 0.1


@dataclass
class ThinkingConfig:
    """Thinking / reasoning mode parameters."""

    enabled: bool = True
    max_thinking_tokens: int = 1024


@dataclass
class VisionConfig:
    """Vision input configuration.

    APEX-1 keeps the language model decoder-only. Images are encoded into
    visual tokens with shape ``[batch, n_visual_tokens, d_model]`` and inserted
    into the text context at the ``<|img|>`` placeholder.
    """

    enabled: bool = False

    # Image preprocessing
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Native educational vision encoder
    encoder_type: str = "native_vit"
    d_vision: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Vision-to-language bridge
    projector_type: str = "perceiver"  # perceiver or mlp
    n_visual_tokens: int = 64
    projector_hidden_dim: int = 1024
    projector_layers: int = 2

    # Token plumbing
    image_token_id: int = 8
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False


@dataclass
class PEFTConfig:
    """Parameter-efficient fine-tuning configuration.

    ``method="lora"`` wraps selected ``nn.Linear`` modules with a frozen base
    projection plus trainable low-rank matrices.

    ``method="qlora"`` first quantizes the frozen base projection to an
    educational 4-bit representation, then trains the same LoRA matrices on top.

    ``method="dora"`` uses Weight-Decomposed LoRA: trainable low-rank direction
    updates plus a trainable per-output magnitude vector.

    ``method="qdora"`` combines the educational 4-bit frozen base with DoRA
    magnitude/direction adapters.
    """

    enabled: bool = False
    method: str = "lora"  # lora, qlora, dora, or qdora
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    freeze_base_model: bool = True

    # QLoRA / quantized-base settings. Used when method == "qlora".
    quantization_bits: int = 4
    quant_type: str = "nf4"  # nf4 or fp4
    double_quant: bool = True
    compute_dtype: str = "float32"  # float32, float16, bfloat16

    # Match child module names. These names cover APEX attention, MLA, GQA,
    # SwiGLU FFN, MoE experts, and MoE router.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "W_Q",
            "W_K",
            "W_V",
            "W_O",
            "W_DKV",
            "W_UK",
            "W_UV",
            "W_DQ",
            "W_UQ",
            "W_KR",
            "W_QR",
            "W_gate",
            "W_up",
            "W_down",
            "router",
        ]
    )

    # Optional module name fragments to keep trainable/saved with adapters.
    # Example: ["embedding", "final_norm"].
    modules_to_save: list[str] = field(default_factory=list)

    # none: train only LoRA matrices
    # all: train every bias in the model
    # lora_only: train biases inside LoRA/QLoRA/DoRA-wrapped modules only
    bias: str = "none"



@dataclass
class AdapterDPOConfig:
    """Adapter-based Direct Preference Optimization configuration.

    DPO trains a policy model directly from preference pairs:

        prompt + chosen response
        prompt + rejected response

    Instead of training a full model, APEX-1 v2.9.0 can train only PEFT adapter
    parameters while keeping the base model frozen. The reference model stays
    frozen and adapter-free by default.
    """

    enabled: bool = False
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False
    length_normalize: bool = False
    max_prompt_len: int = 128
    max_response_len: int = 128
    save_every_steps: int = 0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    seq_len: int = 2048
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"


@dataclass
class GRPOConfig:
    """GRPO alignment parameters."""

    G: int = 8
    beta: float = 0.04
    lambda_prm: float = 0.3
    lambda_cai: float = 0.3
    clip_eps: float = 0.2


@dataclass
class APEXConfig:
    """Complete APEX-1 configuration combining all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    skip_gate: SkipGateConfig = field(default_factory=SkipGateConfig)
    multi_token_head: MultiTokenHeadConfig = field(default_factory=MultiTokenHeadConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    adapter_dpo: AdapterDPOConfig = field(default_factory=AdapterDPOConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "APEXConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info("Loading APEX-1 config from %s", path)

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        config = cls()
        if "model" in raw:
            config.model = _update_dataclass(ModelConfig, raw["model"])
        if "attention" in raw:
            config.attention = _update_dataclass(AttentionConfig, raw["attention"])
        if "moe" in raw:
            config.moe = _update_dataclass(MoEConfig, raw["moe"])
        if "skip_gate" in raw:
            config.skip_gate = _update_dataclass(SkipGateConfig, raw["skip_gate"])
        if "multi_token_head" in raw:
            config.multi_token_head = _update_dataclass(
                MultiTokenHeadConfig, raw["multi_token_head"]
            )
        if "thinking" in raw:
            config.thinking = _update_dataclass(ThinkingConfig, raw["thinking"])
        if "vision" in raw:
            config.vision = _update_dataclass(VisionConfig, raw["vision"])
        if "peft" in raw:
            config.peft = _update_dataclass(PEFTConfig, raw["peft"])
        if "training" in raw:
            config.training = _update_dataclass(TrainingConfig, raw["training"])
        if "grpo" in raw:
            config.grpo = _update_dataclass(GRPOConfig, raw["grpo"])
        if "adapter_dpo" in raw:
            config.adapter_dpo = _update_dataclass(AdapterDPOConfig, raw["adapter_dpo"])

        logger.info(
            "Config loaded: d_model=%d, n_layers=%d, n_experts=%d, vision=%s, peft=%s",
            config.model.d_model,
            config.model.n_layers,
            config.moe.n_experts,
            config.vision.enabled,
            config.peft.enabled,
        )
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import dataclasses

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}
        for section_name in dataclasses.fields(self):
            section_obj = getattr(self, section_name.name)
            data[section_name.name] = dataclasses.asdict(section_obj)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Config saved to %s", path)

    def validate(self) -> None:
        """Validate configuration consistency."""
        m = self.model
        a = self.attention
        v = self.vision
        p = self.peft

        if m.n_heads_q % m.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({m.n_heads_q}) must be divisible by n_heads_kv ({m.n_heads_kv})"
            )

        if m.n_layers % a.global_layer_freq != 0:
            raise ValueError(
                f"n_layers ({m.n_layers}) should be divisible by "
                f"global_layer_freq ({a.global_layer_freq}) for clean layer assignment"
            )

        if m.d_model != m.n_heads_q * m.d_head:
            raise ValueError(
                f"d_model ({m.d_model}) must equal n_heads_q ({m.n_heads_q}) * "
                f"d_head ({m.d_head}) = {m.n_heads_q * m.d_head}. "
                f"Mismatched dimensions will crash the attention output projection."
            )

        if self.moe.enabled and self.moe.n_active > self.moe.n_experts:
            raise ValueError(
                f"n_active ({self.moe.n_active}) cannot exceed n_experts ({self.moe.n_experts})"
            )

        if v.enabled:
            if v.image_size <= 0 or v.patch_size <= 0:
                raise ValueError("vision.image_size and vision.patch_size must be positive")
            if v.image_size % v.patch_size != 0:
                raise ValueError(
                    f"vision.image_size ({v.image_size}) must be divisible by "
                    f"vision.patch_size ({v.patch_size})"
                )
            if v.d_vision % v.n_heads != 0:
                raise ValueError(
                    f"vision.d_vision ({v.d_vision}) must be divisible by "
                    f"vision.n_heads ({v.n_heads})"
                )
            if v.n_visual_tokens <= 0:
                raise ValueError("vision.n_visual_tokens must be positive")
            if v.projector_type not in {"perceiver", "mlp"}:
                raise ValueError("vision.projector_type must be 'perceiver' or 'mlp'")
            if v.encoder_type not in {"native_vit"}:
                raise ValueError("Only vision.encoder_type='native_vit' is implemented")
            max_needed = self.training.seq_len + v.n_visual_tokens
            if max_needed > m.max_seq_len:
                raise ValueError(
                    f"training.seq_len + vision.n_visual_tokens ({max_needed}) exceeds "
                    f"model.max_seq_len ({m.max_seq_len})"
                )

        if p.enabled:
            if p.method not in {"lora", "qlora", "dora", "qdora"}:
                raise ValueError(
                    "Only peft.method='lora', 'qlora', 'dora', or 'qdora' is implemented in v2.9.0"
                )
            if p.r <= 0:
                raise ValueError("peft.r must be positive")
            if p.alpha <= 0:
                raise ValueError("peft.alpha must be positive")
            if not 0.0 <= p.dropout < 1.0:
                raise ValueError("peft.dropout must be in [0.0, 1.0)")
            if p.bias not in {"none", "all", "lora_only"}:
                raise ValueError("peft.bias must be one of: none, all, lora_only")
            if p.method in {"qlora", "qdora"}:
                if p.quantization_bits != 4:
                    raise ValueError("v2.8.0 educational QLoRA/QDoRA currently supports only 4-bit quantization")
                if p.quant_type not in {"nf4", "fp4"}:
                    raise ValueError("peft.quant_type must be one of: nf4, fp4")
                if p.compute_dtype not in {"float32", "float16", "bfloat16"}:
                    raise ValueError("peft.compute_dtype must be float32, float16, or bfloat16")
            if not p.target_modules:
                raise ValueError("peft.target_modules must contain at least one module name")


        dpo = self.adapter_dpo
        if dpo.enabled:
            if not p.enabled:
                raise ValueError("adapter_dpo.enabled=True requires peft.enabled=True")
            if dpo.beta <= 0:
                raise ValueError("adapter_dpo.beta must be positive")
            if not 0.0 <= dpo.label_smoothing < 0.5:
                raise ValueError("adapter_dpo.label_smoothing must be in [0.0, 0.5)")
            if dpo.max_prompt_len <= 0:
                raise ValueError("adapter_dpo.max_prompt_len must be positive")
            if dpo.max_response_len <= 0:
                raise ValueError("adapter_dpo.max_response_len must be positive")

        logger.info("Config validation passed.")


def _update_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    unknown = set(data.keys()) - valid_fields
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, unknown)
    return cls(**filtered)


# ─── Preset Configurations ───────────────────────────────────────────────────


def get_small_config() -> APEXConfig:
    """Return the APEX-1-Small configuration (~100M total params)."""
    return APEXConfig(
        model=ModelConfig(
            d_model=512,
            n_layers=12,
            n_heads_q=8,
            n_heads_kv=2,
            d_head=64,
            d_kv_compressed=64,
            d_q_compressed=96,
            d_head_rope=32,
            d_ffn=1376,
            vocab_size=151643,
            max_seq_len=8192,
            rope_base=10000.0,
            rope_scaling=1.0,
        ),
        attention=AttentionConfig(global_layer_freq=6, local_window=512, flash=True),
        moe=MoEConfig(
            enabled=True,
            n_experts=8,
            n_active=2,
            n_shared=1,
            moe_layer_freq=2,
            balancer_alpha=0.001,
        ),
        skip_gate=SkipGateConfig(enabled=True, hidden_dim=64, threshold=0.15),
        multi_token_head=MultiTokenHeadConfig(enabled=True, n_predict=4, lambda_spec=0.1),
        thinking=ThinkingConfig(enabled=True, max_thinking_tokens=1024),
        vision=VisionConfig(
            enabled=False,
            image_size=224,
            patch_size=16,
            d_vision=512,
            n_layers=6,
            n_heads=8,
            n_visual_tokens=64,
            projector_hidden_dim=1024,
        ),
        training=TrainingConfig(
            batch_size=32,
            seq_len=2048,
            peak_lr=3e-4,
            warmup_steps=1000,
            max_steps=100000,
            grad_clip=1.0,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
        ),
        grpo=GRPOConfig(G=8, beta=0.04, lambda_prm=0.3, lambda_cai=0.3, clip_eps=0.2),
    )


def get_medium_config() -> APEXConfig:
    """Return the APEX-1-Medium configuration (~7B total params)."""
    return APEXConfig(
        model=ModelConfig(
            d_model=2048,
            n_layers=36,
            n_heads_q=16,
            n_heads_kv=4,
            d_head=128,
            d_kv_compressed=256,
            d_q_compressed=384,
            d_head_rope=64,
            d_ffn=5504,
            vocab_size=151643,
            max_seq_len=65536,
            rope_base=500000.0,
            rope_scaling=4.0,
        ),
        attention=AttentionConfig(global_layer_freq=6, local_window=2048, flash=True),
        moe=MoEConfig(
            enabled=True,
            n_experts=64,
            n_active=4,
            n_shared=2,
            moe_layer_freq=2,
            balancer_alpha=0.001,
        ),
        skip_gate=SkipGateConfig(enabled=True, hidden_dim=128, threshold=0.15),
        multi_token_head=MultiTokenHeadConfig(enabled=True, n_predict=4, lambda_spec=0.1),
        thinking=ThinkingConfig(enabled=True, max_thinking_tokens=4096),
        vision=VisionConfig(
            enabled=False,
            image_size=224,
            patch_size=14,
            d_vision=1024,
            n_layers=12,
            n_heads=16,
            n_visual_tokens=128,
            projector_hidden_dim=4096,
        ),
        training=TrainingConfig(
            batch_size=16,
            seq_len=4096,
            peak_lr=1e-4,
            warmup_steps=5000,
            max_steps=500000,
            grad_clip=1.0,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
        ),
        grpo=GRPOConfig(G=8, beta=0.04, lambda_prm=0.3, lambda_cai=0.3, clip_eps=0.2),
    )


def get_large_config() -> APEXConfig:
    """Return the APEX-1-Large configuration (~900B total params)."""
    return APEXConfig(
        model=ModelConfig(
            d_model=7168,
            n_layers=72,
            n_heads_q=128,
            n_heads_kv=8,
            d_head=128,
            d_kv_compressed=512,
            d_q_compressed=768,
            d_head_rope=64,
            d_ffn=18432,
            vocab_size=151643,
            max_seq_len=131072,
            rope_base=1000000.0,
            rope_scaling=8.0,
        ),
        attention=AttentionConfig(global_layer_freq=6, local_window=8192, flash=True),
        moe=MoEConfig(
            enabled=True,
            n_experts=256,
            n_active=8,
            n_shared=4,
            moe_layer_freq=2,
            balancer_alpha=0.001,
        ),
        skip_gate=SkipGateConfig(enabled=True, hidden_dim=256, threshold=0.15),
        multi_token_head=MultiTokenHeadConfig(enabled=True, n_predict=4, lambda_spec=0.1),
        thinking=ThinkingConfig(enabled=True, max_thinking_tokens=8192),
        vision=VisionConfig(
            enabled=False,
            image_size=336,
            patch_size=14,
            d_vision=1280,
            n_layers=24,
            n_heads=16,
            n_visual_tokens=256,
            projector_hidden_dim=8192,
        ),
        training=TrainingConfig(
            batch_size=4,
            seq_len=8192,
            peak_lr=3e-5,
            warmup_steps=15000,
            max_steps=1000000,
            grad_clip=1.0,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
        ),
        grpo=GRPOConfig(G=8, beta=0.04, lambda_prm=0.3, lambda_cai=0.3, clip_eps=0.2),
    )


def get_tiny_config() -> APEXConfig:
    """Return a tiny config for fast unit testing (~1M params)."""
    return APEXConfig(
        model=ModelConfig(
            d_model=64,
            n_layers=6,
            n_heads_q=4,
            n_heads_kv=2,
            d_head=16,
            d_kv_compressed=16,
            d_q_compressed=24,
            d_head_rope=8,
            d_ffn=128,
            vocab_size=1000,
            max_seq_len=256,
            rope_base=10000.0,
            rope_scaling=1.0,
        ),
        attention=AttentionConfig(global_layer_freq=6, local_window=64, flash=False),
        moe=MoEConfig(
            enabled=True,
            n_experts=4,
            n_active=2,
            n_shared=1,
            moe_layer_freq=2,
            balancer_alpha=0.001,
        ),
        skip_gate=SkipGateConfig(enabled=True, hidden_dim=16, threshold=0.15),
        multi_token_head=MultiTokenHeadConfig(enabled=True, n_predict=4, lambda_spec=0.1),
        thinking=ThinkingConfig(enabled=True, max_thinking_tokens=64),
        vision=VisionConfig(
            enabled=False,
            image_size=32,
            patch_size=16,
            d_vision=32,
            n_layers=1,
            n_heads=4,
            n_visual_tokens=4,
            projector_hidden_dim=64,
            projector_layers=2,
        ),
        training=TrainingConfig(
            batch_size=2,
            seq_len=64,
            peak_lr=3e-4,
            warmup_steps=10,
            max_steps=100,
            grad_clip=1.0,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
        ),
        grpo=GRPOConfig(G=4, beta=0.04, lambda_prm=0.3, lambda_cai=0.3, clip_eps=0.2),
    )


def get_tiny_lora_config() -> APEXConfig:
    """Return a tiny CPU-friendly config with LoRA/PEFT enabled."""
    cfg = get_tiny_config()
    cfg.peft.enabled = True
    cfg.peft.r = 4
    cfg.peft.alpha = 8
    cfg.peft.dropout = 0.0
    cfg.training.peak_lr = 1e-4
    cfg.training.max_steps = 20
    return cfg



def get_tiny_qlora_config() -> APEXConfig:
    """Return a tiny CPU-friendly config with QLoRA-style 4-bit PEFT enabled."""
    cfg = get_tiny_lora_config()
    cfg.peft.method = "qlora"
    cfg.peft.quantization_bits = 4
    cfg.peft.quant_type = "nf4"
    cfg.peft.double_quant = True
    cfg.peft.compute_dtype = "float32"
    cfg.training.peak_lr = 1e-4
    cfg.training.max_steps = 20
    return cfg


def get_tiny_dora_config() -> APEXConfig:
    """Return a tiny CPU-friendly config with DoRA PEFT enabled."""
    cfg = get_tiny_lora_config()
    cfg.peft.method = "dora"
    cfg.training.peak_lr = 1e-4
    cfg.training.max_steps = 20
    return cfg


def get_tiny_qdora_config() -> APEXConfig:
    """Return a tiny CPU-friendly config with QDoRA enabled."""
    cfg = get_tiny_qlora_config()
    cfg.peft.method = "qdora"
    cfg.training.peak_lr = 1e-4
    cfg.training.max_steps = 20
    return cfg


def get_tiny_adapter_dpo_config(method: str = "lora") -> APEXConfig:
    """Return a tiny CPU-friendly config for adapter-based DPO alignment.

    Args:
        method: One of ``lora``, ``qlora``, ``dora``, or ``qdora``.
    """
    method = method.lower()
    if method == "lora":
        cfg = get_tiny_lora_config()
    elif method == "qlora":
        cfg = get_tiny_qlora_config()
    elif method == "dora":
        cfg = get_tiny_dora_config()
    elif method == "qdora":
        cfg = get_tiny_qdora_config()
    else:
        raise ValueError("method must be one of: lora, qlora, dora, qdora")

    cfg.peft.enabled = True
    cfg.peft.method = method
    cfg.adapter_dpo.enabled = True
    cfg.adapter_dpo.beta = 0.1
    cfg.adapter_dpo.label_smoothing = 0.0
    cfg.adapter_dpo.reference_free = False
    cfg.adapter_dpo.length_normalize = False
    cfg.adapter_dpo.max_prompt_len = 64
    cfg.adapter_dpo.max_response_len = 64
    cfg.training.batch_size = 1
    cfg.training.peak_lr = 1e-4
    cfg.training.max_steps = 20
    return cfg

def get_tiny_vision_config() -> APEXConfig:
    """Return a tiny CPU-friendly config with vision enabled."""
    cfg = get_tiny_config()
    cfg.vision.enabled = True
    cfg.vision.image_size = 32
    cfg.vision.patch_size = 16
    cfg.vision.d_vision = 32
    cfg.vision.n_layers = 1
    cfg.vision.n_heads = 4
    cfg.vision.n_visual_tokens = 4
    cfg.vision.projector_hidden_dim = 64
    cfg.vision.projector_layers = 2
    cfg.training.seq_len = 64
    return cfg
