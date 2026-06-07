
# APEX-1 Model Card

## Model Name

**APEX-1**

## Version

**v3.0.0 — Course-Ready Stable Release**

## Project Type

APEX-1 is an educational from-scratch large language + vision model architecture. It is designed for learning how modern LLM/VLM systems are built internally.

It is not distributed as a large pretrained production assistant model.

## Intended Use

APEX-1 is intended for:

- Learning how language models predict the next token.
- Understanding transformer architecture from code.
- Studying modern attention mechanisms such as MLA, GQA, and sliding-window attention.
- Learning Mixture of Experts, dynamic skip gates, and multi-token prediction.
- Understanding training loops, losses, optimizers, checkpointing, and datasets.
- Learning generation techniques like sampling, KV cache, thinking mode, and speculative decoding.
- Understanding vision-token insertion and multimodal model structure.
- Studying PEFT methods such as LoRA, QLoRA, DoRA, QDoRA, and adapter-DPO.

## Not Intended For

APEX-1 v3.0.0 is not intended to be used as:

- A production chatbot without proper training.
- A safety-critical decision system.
- A replacement for large pretrained frontier models.
- A medical, legal, financial, or security decision engine.
- A model for generating high-quality real-world answers without real training data and evaluation.

## Training Status

APEX-1 does **not** ship with a large pretrained checkpoint.

The tiny CPU demos are designed to verify architecture, tensor shapes, training mechanics, adapter workflows, and inference paths. Real language understanding or high-quality generation requires:

- a trained base checkpoint,
- large-scale data,
- adequate compute,
- evaluation,
- safety testing,
- and deployment-specific validation.

## Architecture Summary

APEX-1 combines educational implementations of:

- Decoder-only transformer modeling.
- Token embeddings and RMSNorm.
- RoPE and YaRN-style positional handling.
- Multi-Head Latent Attention.
- Grouped Query Attention.
- Sliding-window attention.
- SwiGLU feed-forward networks.
- Mixture of Experts.
- Auxiliary-loss-free load balancing.
- Dynamic skip gate.
- Multi-token prediction heads.
- Thinking mode.
- Vision encoder and visual-token bridge.
- LoRA / QLoRA / DoRA / QDoRA adapters.
- Adapter-DPO preference alignment.

## Inputs

Depending on the example or config, APEX-1 can accept:

- text token IDs,
- optional image tensors,
- prompt / response pairs,
- supervised fine-tuning samples,
- preference pairs for DPO-style training.

## Outputs

Depending on the mode, APEX-1 can output:

- logits over the vocabulary,
- generated token sequences,
- hidden states,
- visual-token-augmented outputs,
- adapter checkpoints,
- merged checkpoints,
- evaluation / benchmark summaries.

## Limitations

- The default tiny model is for learning and smoke tests, not quality generation.
- No large pretrained checkpoint is included.
- Real image understanding requires vision-language training data and trained weights.
- Adapter fine-tuning demos verify mechanics, but meaningful adaptation requires a trained base model.
- Alignment demos teach DPO mechanics but do not produce a production-safe aligned assistant.
- Performance depends heavily on hardware, configuration, data, and training scale.

## Safety Notes

APEX-1 is a teaching project. Any model trained from this code should be evaluated for:

- hallucinations,
- bias,
- unsafe content,
- privacy leakage,
- prompt injection behavior,
- misuse risk,
- dataset contamination,
- and deployment-specific failure modes.

## License

Apache-2.0. See `LICENSE`.
