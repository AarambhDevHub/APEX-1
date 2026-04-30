# 01 — Project Structure: Every File Explained

> **Difficulty:** ⭐☆☆☆☆ Beginner  
> **Time to read:** ~10 minutes  
> **You will learn:** What every file and folder in the APEX-1 project does.

---

## 1. Why Understanding Structure Matters

Before reading code, you need to know **where things live**. Think of the project structure as a kitchen map — you need to know where the knives, pots, and ingredients are before you can cook.

---

## 2. The Full Directory Tree

```
APEX-Model/
│
├── apex/                          ← Main Python package (the "brain")
│   ├── __init__.py                ← Package marker; sets version = "2.2.0"
│   ├── config.py                  ← ALL settings and hyperparameters
│   │
│   ├── model/                     ← Core neural network components
│   │   ├── norm.py                ← RMSNorm (layer normalisation)
│   │   ├── rope.py                ← Rotary Positional Encoding + YaRN
│   │   ├── mask.py                ← Attention mask builder
│   │   ├── attention.py           ← MLA + GQA attention mechanisms
│   │   ├── ffn.py                 ← Feed-Forward Networks (Dense + MoE)
│   │   ├── skip_gate.py           ← Dynamic skip gate
│   │   ├── load_balancer.py       ← Auxiliary-loss-free MoE load balancer
│   │   ├── multi_token_head.py    ← Speculative prediction heads
│   │   ├── block.py               ← One complete transformer block
│   │   └── apex_model.py          ← The complete APEX-1 model
│   │
│   ├── tokenizer/                 ← Text ↔ token conversion
│   │   ├── tokenizer.py           ← BPE tokenizer with special tokens
│   │   └── train_tokenizer.py     ← Script to train a new tokenizer
│   │
│   ├── generation/                ← Text generation (inference)
│   │   ├── sampler.py             ← Temperature, top-p, top-k, repetition penalty
│   │   └── generator.py           ← Full generation engine with KV cache
│   │
│   ├── training/                  ← Training infrastructure
│   │   ├── losses.py              ← Loss functions
│   │   ├── trainer.py             ← PreTrainer and SFTTrainer
│   │   ├── scheduler.py           ← Cosine warmup LR schedule
│   │   └── checkpoint.py          ← Save/load checkpoints
│   │
│   ├── alignment/                 ← Safety and helpfulness
│   │   ├── reward_model.py        ← Scores responses
│   │   ├── dpo.py                 ← Direct Preference Optimization
│   │   ├── grpo.py                ← Group Relative Policy Optimization
│   │   ├── prm.py                 ← Process Reward Model
│   │   ├── constitutional.py      ← Constitutional AI
│   │   └── combined_reward.py     ← All signals combined
│   │
│   ├── data/                      ← Data loading
│   │   ├── dataset.py             ← Dataset classes
│   │   └── data_loader.py         ← DataLoader factories
│   │
│   └── utils/                     ← Helper tools
│       ├── param_counter.py       ← Count parameters
│       ├── shape_checker.py       ← Verify tensor shapes
│       └── flops.py               ← Estimate FLOPs
│
├── configs/                       ← YAML config presets
│   ├── apex1_tiny.yaml            ← ~1M params (tests)
│   ├── apex1_small.yaml           ← ~100M params
│   ├── apex1_medium.yaml          ← ~7B params
│   └── apex1_large.yaml           ← ~900B params
│
├── tests/                         ← Automated tests (86 passing)
│   ├── test_all.py                ← Integration tests
│   └── test_bugfixes.py           ← Regression tests for 24 bugs
│
├── examples/                      ← Quick demo scripts
│   ├── forward_pass_demo.py
│   ├── generation_demo.py
│   ├── thinking_mode_demo.py
│   └── mask_visualization.py
│
├── scripts/                       ← CLI entry points
│   ├── train.py                   ← Training CLI
│   └── generate.py                ← Generation CLI
│
├── pyproject.toml                 ← Package config, dependencies, tool settings
├── Makefile                       ← Shortcut commands
├── Dockerfile                     ← Container definition
├── CHANGELOG.md                   ← History of every change
└── APEX-1-Model-Architecture.md   ← Full technical design doc
```

---

## 3. The Dependency Flow

Files import from each other in this order (top = most fundamental):

```
config.py
    ↓
model/norm.py     model/rope.py
    ↓                   ↓
model/mask.py     model/attention.py    model/ffn.py
    ↓
model/block.py  (uses: attention + ffn + skip_gate + norm)
    ↓
model/apex_model.py  (uses: block × n_layers + multi_token_head)
    ↓
generation/generator.py    training/trainer.py    alignment/*.py
```

---

## 4. Reading Order for Beginners

| Step | File | Why This Order |
|---|---|---|
| 1 | `config.py` | Defines all dimensions — needed to understand tensor shapes |
| 2 | `model/norm.py` | Simplest component; just 10 lines |
| 3 | `model/rope.py` | Positional encoding; used in every attention layer |
| 4 | `model/mask.py` | Attention mask; needed before reading attention |
| 5 | `model/attention.py` | Core of the transformer |
| 6 | `model/ffn.py` | Second major component |
| 7 | `model/skip_gate.py` | Optional gate |
| 8 | `model/load_balancer.py` | Pure Python, easy to follow |
| 9 | `model/multi_token_head.py` | Small add-on |
| 10 | `model/block.py` | Combines attention + FFN |
| 11 | `model/apex_model.py` | Final assembly |
| 12–16 | `training/` | How the model learns |
| 17–18 | `generation/` | How text is generated |
| 19–24 | `alignment/` | Safety and alignment |

---

## 5. Key Design Pattern

Every file in `apex/model/` follows this pattern:

```python
class SomeThing(nn.Module):        # Always inherits nn.Module
    def __init__(self, config):    # Takes config, builds layers
        super().__init__()
        self.layer = nn.Linear(...)
    
    def forward(self, x):          # The actual computation
        return self.layer(x)
```

`nn.Module` is PyTorch's base class for any learnable component. It tracks parameters, allows GPU placement, and enables saving/loading.

---

**Next:** [02 — Configuration →](02-configuration.md)
