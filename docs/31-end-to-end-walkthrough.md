# 31 — End-to-End Walkthrough: Raw Text → Trained Chatbot

> **Difficulty:** ⭐⭐⭐⭐☆ Advanced  
> **You will learn:** The complete lifecycle of training and running APEX-1, from raw data to a conversational AI assistant.

---

## Overview

This document walks through the **complete pipeline** for building a working APEX-1 chatbot:

| Stage | What Happens | Time (Tiny model) |
|---|---|---|
| 1. Install | Set up environment | 2 min |
| 2. Configure | Create a tiny test config | 1 min |
| 3. Tokenise | Build the vocabulary | 1 min |
| 4. Data prep | Prepare training data | 2 min |
| 5. Pretrain | Train on raw text | 10 min |
| 6. SFT | Fine-tune on conversations | 5 min |
| 7. Alignment | GRPO + DPO | 5 min |
| 8. Generate | Run the chatbot | immediate |

---

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/AarambhDevHub/APEX-1.git
cd APEX-1

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install all dependencies (including training tools)
pip install -e ".[all]"

# Verify installation
python -c "from apex import APEX1Model, APEXConfig; print('APEX-1 installed!')"
```

---

## Step 2: Create a Tiny Configuration

For learning and testing, we use the smallest possible config (1M params):

```python
# create_tiny_config.py
import yaml
from pathlib import Path

tiny_config = {
    "model": {
        "d_model": 64,
        "n_layers": 6,
        "n_heads_q": 4,
        "n_heads_kv": 2,
        "d_head": 16,
        "d_kv_compressed": 16,
        "d_q_compressed": 24,
        "d_head_rope": 8,
        "d_ffn": 172,
        "vocab_size": 151643,
        "max_seq_len": 256,
        "rope_base": 10000.0,
        "rope_scaling": 1.0,
        "dropout": 0.0,
    },
    "attention": {
        "global_layer_freq": 6,
        "local_window": 64,
        "flash": False,  # Use manual attention on CPU
    },
    "moe": {
        "enabled": True,
        "n_experts": 4,
        "n_active": 2,
        "n_shared": 1,
        "moe_layer_freq": 2,
        "balancer_alpha": 0.001,
    },
    "skip_gate": {"enabled": True, "hidden_dim": 16, "threshold": 0.15},
    "multi_token_head": {"enabled": True, "n_predict": 4, "lambda_spec": 0.1},
    "thinking": {"enabled": True, "max_thinking_tokens": 128},
    "training": {
        "batch_size": 4,
        "seq_len": 128,
        "peak_lr": 3e-4,
        "warmup_steps": 100,
        "max_steps": 1000,
        "grad_clip": 1.0,
        "weight_decay": 0.1,
        "mixed_precision": "no",  # CPU training: no mixed precision
    },
}

Path("configs/apex1_tiny.yaml").write_text(yaml.dump(tiny_config))
print("Config saved to configs/apex1_tiny.yaml")
```

Run it:
```bash
python create_tiny_config.py
```

---

## Step 3: Verify Model Architecture

```python
# verify_model.py
from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.utils.shape_checker import verify_shapes
from apex.utils.param_counter import count_parameters
from apex.utils.flops import estimate_model_flops

# Load config
config = APEXConfig.from_yaml("configs/apex1_tiny.yaml")
config.validate()   # Check for inconsistencies

# Build model
model = APEX1Model(config)
print("Model built successfully!")

# Count parameters
params = count_parameters(model)
print(f"\nParameters:")
print(f"  Total:   {params['total_params_M']:.2f}M")
print(f"  Active:  {params['active_params_M']:.2f}M")

# Verify all tensor shapes
shapes = verify_shapes(config, model=model, seq_len=32)
print(f"\nShapes verified:")
for name, shape in shapes.items():
    print(f"  {name}: {shape}")

# Estimate FLOPs
flops = estimate_model_flops(config, seq_len=128)
print(f"\nFLOPs per token: {flops['total_flops_per_token']:.0f}")
```

---

## Step 4: Prepare Training Data

```python
# prepare_data.py
"""
For pretraining: we need raw text files.
For SFT: we need JSON conversation files.
"""

import json
from pathlib import Path

# Create a tiny pretraining corpus (normally: gigabytes of text)
pretrain_text = """
The sun rises in the east and sets in the west.
Water consists of hydrogen and oxygen molecules.
Python is a programming language created by Guido van Rossum.
Neural networks learn by adjusting weights through backpropagation.
The moon orbits the Earth approximately every 27 days.
Mathematics is the language of the universe.
Transformers are neural network architectures based on attention mechanisms.
""" * 100   # Repeat to get ~3000 words

Path("data").mkdir(exist_ok=True)
Path("data/pretrain.txt").write_text(pretrain_text)
print(f"Pretrain data: {len(pretrain_text)} chars")

# Create SFT data (chat conversations)
sft_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the sun?"},
            {"role": "assistant", "content": "The sun is a star at the center of our solar system. It provides light and heat to Earth."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language known for its clear syntax and versatility. It is widely used for data science, web development, and AI."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is a transformer model?"},
            {"role": "assistant", "content": "A transformer is a type of neural network that uses attention mechanisms to process sequences. It is the foundation of modern large language models like GPT and BERT."},
        ]
    },
]

with open("data/sft.jsonl", "w") as f:
    for sample in sft_data * 20:   # Repeat for more data
        f.write(json.dumps(sample) + "\n")

print(f"SFT data: {len(sft_data) * 20} samples")
```

---

## Step 5: Run Pretraining

```python
# pretrain.py
"""
Full pretraining run on the tiny corpus.
"""

import torch
from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.tokenizer.tokenizer import APEX1Tokenizer
from apex.data.dataset import StreamingPretrainDataset
from apex.data.data_loader import create_pretrain_loader
from apex.training.trainer import PreTrainer

# ── Setup ─────────────────────────────────────────────────────────────
config = APEXConfig.from_yaml("configs/apex1_tiny.yaml")
config.validate()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# ── Tokenizer ─────────────────────────────────────────────────────────
# Load the pre-trained tokenizer (Qwen3 vocabulary)
# For a real run: download tokenizer files first
tokenizer = APEX1Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")   # or local path

# ── Model ─────────────────────────────────────────────────────────────
model = APEX1Model(config)
model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ── Dataset ───────────────────────────────────────────────────────────
dataset = StreamingPretrainDataset(
    file_paths=["data/pretrain.txt"],
    tokenizer=tokenizer,
    seq_len=config.training.seq_len,
    shuffle_files=False,
)
loader = create_pretrain_loader(
    dataset,
    batch_size=config.training.batch_size,
    num_workers=0,   # Single worker for small test
)

# ── Trainer ───────────────────────────────────────────────────────────
trainer = PreTrainer(
    model=model,
    config=config,
    train_loader=loader,
    device=device,
    checkpoint_dir="checkpoints/pretrain",
)

# ── Train! ────────────────────────────────────────────────────────────
print("Starting pretraining...")
trainer.train(max_steps=config.training.max_steps)
print("Pretraining complete!")

# Save checkpoint
from apex.training.checkpoint import save_checkpoint
save_checkpoint(
    checkpoint_dir="checkpoints/pretrain",
    model=model,
    optimizer=trainer.optimizer,
    scheduler=trainer.scheduler,
    global_step=trainer.global_step,
    epoch=0,
    loss=0.0,
    tag="final",
)
```

---

## Step 6: Supervised Fine-Tuning (SFT)

```python
# sft.py
"""
SFT on conversation data to teach the model to follow instructions.
"""

from apex.data.dataset import SFTDataset
from apex.data.data_loader import create_sft_loader
from apex.training.trainer import SFTTrainer
from apex.training.checkpoint import load_checkpoint

# Load pretrained checkpoint
config = APEXConfig.from_yaml("configs/apex1_tiny.yaml")
model = APEX1Model(config).to(device)
load_checkpoint("checkpoints/pretrain/checkpoint_final.pt", model)

# Load SFT data
sft_dataset = SFTDataset.from_jsonl("data/sft.jsonl", tokenizer, max_seq_len=config.model.max_seq_len)
sft_loader = create_sft_loader(sft_dataset, batch_size=2)

# SFT trainer uses lower LR and assistant-only loss
sft_trainer = SFTTrainer(model=model, config=config, train_loader=sft_loader, device=device)

print("Starting SFT...")
sft_trainer.train(max_steps=500)

save_checkpoint("checkpoints/sft", model, sft_trainer.optimizer, sft_trainer.scheduler, 
                global_step=500, epoch=0, loss=0.0, tag="final")
print("SFT complete!")
```

---

## Step 7: Generate Text

```python
# generate.py
"""
Generate a response using the trained model.
"""

import torch
from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.tokenizer.tokenizer import APEX1Tokenizer
from apex.generation.generator import APEX1Generator, GenerationConfig
from apex.training.checkpoint import load_checkpoint

# Load SFT checkpoint
config = APEXConfig.from_yaml("configs/apex1_tiny.yaml")
tokenizer = APEX1Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = APEX1Model(config)
load_checkpoint("checkpoints/sft/checkpoint_final.pt", model)
model.eval()

# Build generator
gen_config = GenerationConfig(
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    enable_thinking=True,
    max_thinking_tokens=64,
)
generator = APEX1Generator(model, gen_config)

# Format a conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a transformer model?"},
]
input_ids = tokenizer.encode_chat(messages, add_generation_prompt=True)
input_tensor = torch.tensor([input_ids], dtype=torch.long)

# Generate!
print("Generating response...")
output = generator.generate(input_tensor, prefix_len=len(input_ids))

response = tokenizer.decode(output.token_ids)
print(f"\nResponse: {response}")
print(f"Tokens generated: {output.total_tokens}")
print(f"Thinking tokens used: {output.thinking_tokens}")
```

---

## Step 8: Run the Demo Scripts

```bash
# Quick forward pass (no training needed)
python examples/forward_pass_demo.py

# Text generation
python examples/generation_demo.py

# Thinking mode
python examples/thinking_mode_demo.py

# Visualise the attention mask pattern
python examples/mask_visualization.py
```

---

## Appendix: The Complete Data Flow

```
Raw text: "The sun rises in the east."
            │
            ▼ tokenizer.encode()
Token IDs: [785, 7777, 35532, 304, 279, 1973, 13]
            │
            ▼ nn.Embedding + √d_model
Embeddings: [[0.23, -0.12, ...], ...]   # [7, d_model]
            │
            ▼ RoPE positional encoding (rotate Q and K)
            │
            ▼ Block 0 (GQA+SW attention, Dense FFN)
            │   → tokens attend to nearby context
            │   → FFN processes each token independently
            │
            ▼ Block 1 (GQA+SW attention, MoE FFN)
            │   → router selects 2 of 4 experts per token
            │
            ▼ ...Blocks 2–4...
            │
            ▼ Block 5 (MLA attention, MoE FFN)
            │   → global context — all tokens see all others
            │
            ▼ Final RMSNorm
            │
            ▼ LM Head (= Embedding.weight.T)
Logits:    [p(word_0), p(word_1), ..., p(word_151642)]
            │
            ▼ softmax + sample
Next token: "east"  (maybe... depends on sampling)
```

---

## Congratulations!

You have read through the complete APEX-1 documentation. You now understand:

- ✅ What a large language model is and how it is structured
- ✅ How tokens, embeddings, and attention work mathematically
- ✅ The innovations: MLA, MoE, GQA+SW, SwiGLU, Skip Gate, RoPE+YaRN
- ✅ How to train: loss functions, AdamW, cosine schedule, gradient clipping
- ✅ How to generate: KV cache, sampling strategies, speculative decoding
- ✅ How to align: DPO, GRPO, PRM, Constitutional AI, combined reward
- ✅ All 24 bugs that were found and fixed, and why they mattered

**What to explore next:**
- Read the [APEX-1 Architecture Document](../APEX-1-Model-Architecture.md) for the complete technical design
- Run the [examples/](../examples/) scripts to see it working live
- Modify `configs/apex1_tiny.yaml` and observe how changing hyperparameters affects training
- Read the mathematical reference documents in `docs/` for deeper formal derivations

---

*Built with ❤️ by AarambhDevHub — Teaching AI from the ground up.*
