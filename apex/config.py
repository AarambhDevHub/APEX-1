"""
APEX-1 Model Configuration.

Defines the complete configuration dataclass for all APEX-1 model sizes,
loadable from YAML config files. Covers model dimensions, attention settings,
MoE parameters, skip gate, multi-token prediction, thinking mode, training
hyperparameters, and GRPO alignment settings.
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
    """Complete APEX-1 configuration combining all sub-configurations.

    This is the top-level config that holds every hyperparameter needed
    to instantiate, train, and run inference on an APEX-1 model.

    Attributes:
        model: Core architecture dimensions.
        attention: Attention strategy settings.
        moe: Mixture of Experts settings.
        skip_gate: Dynamic skip gate settings.
        multi_token_head: Multi-token prediction head settings.
        thinking: Thinking / reasoning mode settings.
        training: Training hyperparameters.
        grpo: GRPO alignment settings.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    skip_gate: SkipGateConfig = field(default_factory=SkipGateConfig)
    multi_token_head: MultiTokenHeadConfig = field(default_factory=MultiTokenHeadConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> APEXConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated APEXConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info("Loading APEX-1 config from %s", path)

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

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
        if "training" in raw:
            config.training = _update_dataclass(TrainingConfig, raw["training"])
        if "grpo" in raw:
            config.grpo = _update_dataclass(GRPOConfig, raw["grpo"])

        logger.info(
            "Config loaded: d_model=%d, n_layers=%d, n_experts=%d",
            config.model.d_model,
            config.model.n_layers,
            config.moe.n_experts,
        )
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Destination file path.
        """
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
        """Validate configuration consistency.

        Raises:
            ValueError: If any config values are inconsistent.
        """
        m = self.model
        a = self.attention

        if m.n_heads_q % m.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({m.n_heads_q}) must be divisible by n_heads_kv ({m.n_heads_kv})"
            )

        if m.n_layers % a.global_layer_freq != 0:
            raise ValueError(
                f"n_layers ({m.n_layers}) should be divisible by "
                f"global_layer_freq ({a.global_layer_freq}) for clean layer assignment"
            )

        # BUG-18 FIX: raise ValueError instead of warning.  A d_model
        # mismatch causes a hard shape error in the attention output
        # projection (W_O), so it must be caught before model construction.
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

        logger.info("Config validation passed.")


def _update_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, ignoring unknown keys.

    Args:
        cls: The dataclass type to instantiate.
        data: Dictionary of field values.

    Returns:
        Instance of cls with provided values.
    """
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
    """Return a tiny config for fast unit testing (~1M params).

    This configuration uses minimal dimensions so tests can run on CPU
    in seconds without consuming significant memory.
    """
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
