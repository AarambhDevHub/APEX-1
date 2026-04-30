# 02 — Configuration: The Model Blueprint

> **Difficulty:** ⭐☆☆☆☆ Beginner  
> **Source file:** `apex/config.py`  
> **You will learn:** How every hyperparameter is organised, validated, and loaded from YAML.

---

## 1. What Is a Configuration?

Before building a house, you need blueprints — plans that specify how many rooms, how tall the walls, where the windows go. The configuration file is the blueprint for APEX-1.

Every dimension, every setting, every hyperparameter is defined here. If you want a bigger model, you change the config. You never change the neural network code itself.

---

## 2. The 8 Sub-Configurations

`APEXConfig` is a container that holds 8 smaller config objects, each responsible for one part of the model:

```
APEXConfig
├── ModelConfig         ← Core dimensions (d_model, n_layers, etc.)
├── AttentionConfig     ← Attention strategy (global_layer_freq, window size)
├── MoEConfig           ← Mixture of Experts settings
├── SkipGateConfig      ← Dynamic skip gate settings
├── MultiTokenHeadConfig← Speculative prediction settings
├── ThinkingConfig      ← Thinking mode settings
├── TrainingConfig      ← Optimizer, LR, batch size, etc.
└── GRPOConfig          ← Alignment (RL) settings
```

---

## 3. ModelConfig — The Core Dimensions

```python
@dataclass
class ModelConfig:
    """Core model architecture dimensions."""

    d_model: int = 512        # Width of every vector in the model
    n_layers: int = 12        # How many transformer blocks stacked
    n_heads_q: int = 8        # Number of query attention heads
    n_heads_kv: int = 2       # Number of key/value heads (GQA)
    d_head: int = 64          # Size of each attention head
    d_kv_compressed: int = 64 # MLA: compressed KV latent dimension
    d_q_compressed: int = 96  # MLA: compressed Q latent dimension
    d_head_rope: int = 32     # RoPE head dimension (for MLA decoupled RoPE)
    d_ffn: int = 1376         # Width of feed-forward networks
    vocab_size: int = 151643  # Number of tokens in vocabulary
    max_seq_len: int = 8192   # Maximum sequence length
    rope_base: float = 10000.0# RoPE frequency base
    rope_scaling: float = 1.0 # YaRN scaling factor (1.0 = no scaling)
    dropout: float = 0.0      # Dropout probability (0 = disabled)
```

### Critical Relationship

The most important constraint is:

$$d_{\text{model}} = n_{\text{heads\_q}} \times d_{\text{head}}$$

**Why?** The model's hidden dimension must equal the total size of all query heads put together. For Small: $8 \times 64 = 512$. For Large: $128 \times 64 = 8192$.

If you break this rule, the model crashes. That is why `validate()` raises a `ValueError` for this (Bug-18 fix).

---

## 4. AttentionConfig

```python
@dataclass
class AttentionConfig:
    """Attention strategy parameters."""

    global_layer_freq: int = 6  # Every 6th layer is a global (MLA) layer
    local_window: int = 512     # Local layers only see the last 512 tokens
    flash: bool = True          # Use Flash Attention (faster on GPU)
```

**global_layer_freq = 6** means: in a 12-layer model, layers 5 and 11 are global (MLA), and layers 0–4 and 6–10 are local (GQA). The pattern is: every 6th layer (counting from 0) — specifically when `layer_idx % 6 == 5`.

---

## 5. MoEConfig

```python
@dataclass
class MoEConfig:
    """Mixture of Experts parameters."""

    enabled: bool = True        # Turn MoE on or off
    n_experts: int = 8          # Total number of routed experts
    n_active: int = 2           # How many experts activate per token
    n_shared: int = 1           # How many experts always activate
    moe_layer_freq: int = 2     # Every 2nd layer uses MoE (odd layers)
    balancer_alpha: float = 0.001 # Load balancer step size
```

**n_active / n_experts** is the "sparsity ratio". For Large: 8 out of 256 = 3.1% of experts active per token. This is what makes large models feasible — you have 900B total parameters but only ~45B are computed per token.

---

## 6. The 4 Model Size Presets

| Field | Tiny (test) | Small | Medium | Large |
|---|---|---|---|---|
| `d_model` | 64 | 512 | 2,048 | 7,168 |
| `n_layers` | 6 | 12 | 36 | 72 |
| `n_heads_q` | 4 | 8 | 16 | 128 |
| `n_experts` | 4 | 8 | 64 | 256 |
| `max_seq_len` | 256 | 8,192 | 65,536 | 131,072 |
| Total params | ~1M | ~100M | ~7B | ~900B |
| Active params | ~0.5M | ~40M | ~2B | ~45B |

---

## 7. Full Annotated Source: `apex/config.py`

```python
"""
APEX-1 Model Configuration.

Defines the complete configuration dataclass for all APEX-1 model sizes,
loadable from YAML config files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field   # 'dataclass' auto-generates __init__
from pathlib import Path
from typing import Any

import yaml   # For reading .yaml config files

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configurations (one per model component)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Core model architecture dimensions."""
    d_model: int = 512          # Hidden size — every vector is this wide
    n_layers: int = 12          # Stack this many transformer blocks
    n_heads_q: int = 8          # Query heads (more = richer attention)
    n_heads_kv: int = 2         # KV heads (fewer = GQA = less memory)
    d_head: int = 64            # Each head processes this many dimensions
    d_kv_compressed: int = 64   # MLA: compressed KV size (<<< d_model)
    d_q_compressed: int = 96    # MLA: compressed Q size
    d_head_rope: int = 32       # RoPE head size for decoupled rope in MLA
    d_ffn: int = 1376           # FFN hidden width (~2.7 × d_model)
    vocab_size: int = 151643    # Qwen3 vocabulary (multilingual + code)
    max_seq_len: int = 8192     # Max tokens in one sequence
    rope_base: float = 10000.0  # RoPE base frequency
    rope_scaling: float = 1.0   # 1.0 = no YaRN; >1.0 = extended context
    dropout: float = 0.0        # 0 = no dropout (standard for large models)


@dataclass
class AttentionConfig:
    """Attention strategy parameters."""
    global_layer_freq: int = 6   # 1 global layer per 6 total
    local_window: int = 512      # Local layers see only last N tokens
    flash: bool = True           # Use Flash Attention on CUDA


@dataclass
class MoEConfig:
    """Mixture of Experts parameters."""
    enabled: bool = True         # Set False for a dense-only model
    n_experts: int = 8           # Total number of routed experts
    n_active: int = 2            # Active experts per token (sparse)
    n_shared: int = 1            # Always-active shared experts
    moe_layer_freq: int = 2      # MoE on odd layers (1 % 2 != 0, etc.)
    balancer_alpha: float = 0.001 # Bias update step size


@dataclass
class SkipGateConfig:
    """Dynamic skip gate parameters."""
    enabled: bool = True         # Set False to always run FFN
    hidden_dim: int = 64         # Gate MLP hidden size
    threshold: float = 0.15     # Gate < this → skip FFN


@dataclass
class MultiTokenHeadConfig:
    """Multi-token prediction head parameters."""
    enabled: bool = True         # Set False for standard single-token LM
    n_predict: int = 4           # How many future tokens to predict
    lambda_spec: float = 0.1    # Weight of spec loss (10% of total)


@dataclass
class ThinkingConfig:
    """Thinking / reasoning mode parameters."""
    enabled: bool = True
    max_thinking_tokens: int = 1024  # Budget for the reasoning scratchpad


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    seq_len: int = 2048
    peak_lr: float = 3e-4        # Maximum learning rate
    min_lr_ratio: float = 0.1   # Minimum LR = peak_lr × this
    warmup_steps: int = 1000    # Linear warmup for this many steps
    max_steps: int = 100000     # Total training steps
    grad_clip: float = 1.0      # Clip gradients to this norm
    weight_decay: float = 0.1   # AdamW weight decay
    optimizer: str = "adamw"
    beta1: float = 0.9          # AdamW momentum parameter
    beta2: float = 0.95         # AdamW second momentum
    eps: float = 1e-8           # AdamW numerical stability
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"


@dataclass
class GRPOConfig:
    """GRPO alignment parameters."""
    G: int = 8                   # Rollouts per prompt
    beta: float = 0.04           # KL penalty strength
    lambda_prm: float = 0.3     # Process reward weight
    lambda_cai: float = 0.3     # Constitutional AI reward weight
    clip_eps: float = 0.2       # PPO clipping epsilon


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config combining all the above
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class APEXConfig:
    """Complete APEX-1 configuration."""

    # field(default_factory=...) means: create a new instance each time
    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    skip_gate: SkipGateConfig = field(default_factory=SkipGateConfig)
    multi_token_head: MultiTokenHeadConfig = field(default_factory=MultiTokenHeadConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "APEXConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)   # Parse YAML → dict

        config = cls()   # Start with all defaults
        
        # Override each section if present in the YAML file
        if "model" in raw:
            config.model = _update_dataclass(ModelConfig, raw["model"])
        if "attention" in raw:
            config.attention = _update_dataclass(AttentionConfig, raw["attention"])
        # ... (same pattern for all sections)
        
        return config

    def validate(self) -> None:
        """Check that all config values are consistent."""
        m = self.model
        a = self.attention

        # n_heads_q must be divisible by n_heads_kv (for GQA)
        if m.n_heads_q % m.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({m.n_heads_q}) must be divisible by n_heads_kv ({m.n_heads_kv})"
            )

        # n_layers must be divisible by global_layer_freq (clean layer assignment)
        if m.n_layers % a.global_layer_freq != 0:
            raise ValueError(...)

        # BUG-18 FIX: This must be an error, not a warning.
        # If d_model != n_heads_q * d_head, the output projection W_O crashes.
        if m.d_model != m.n_heads_q * m.d_head:
            raise ValueError(
                f"d_model ({m.d_model}) must equal n_heads_q * d_head "
                f"= {m.n_heads_q * m.d_head}"
            )

        # Cannot activate more experts than exist
        if self.moe.enabled and self.moe.n_active > self.moe.n_experts:
            raise ValueError(...)


def _update_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Create a dataclass from a dict, ignoring unknown keys.
    
    This is used when loading from YAML — if the YAML has an unknown key,
    we log a warning and ignore it rather than crashing.
    """
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    unknown = set(data.keys()) - valid_fields
    if unknown:
        logger.warning("Ignoring unknown config keys for %s: %s", cls.__name__, unknown)
    return cls(**filtered)
```

---

## 8. How a YAML Config Maps to Python

The file `configs/apex1_tiny.yaml` might look like:

```yaml
model:
  d_model: 64
  n_layers: 6
  n_heads_q: 4
  d_head: 16

moe:
  n_experts: 4
  n_active: 2
```

When loaded with `APEXConfig.from_yaml("configs/apex1_tiny.yaml")`, each section becomes a Python dataclass:

```python
config.model.d_model   # → 64
config.moe.n_experts   # → 4
config.training.peak_lr # → 3e-4  (default, not in yaml)
```

---

## 9. Why This Design Is Smart

1. **Single source of truth**: All hyperparameters in one place. You never hunt through code to find a magic number.
2. **Validation catches bugs early**: A misconfigured model fails immediately with a clear message, not halfway through training.
3. **Easy experimentation**: Switch from tiny to small by changing one YAML file.
4. **Dataclasses auto-generate** `__init__`, `__repr__`, and `__eq__` — less boilerplate.

---

**Next:** [03 — Tokenizer →](03-tokenizer.md)
