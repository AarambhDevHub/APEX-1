"""
APEX-1 v2.6.0 LoRA inference and merge smoke demo.

This demo is CPU-friendly and does not require a pretrained checkpoint.
It proves the complete adapter lifecycle works:

1. build tiny LoRA model
2. save adapter-only checkpoint
3. load adapter for generation
4. run a tiny generation call
5. merge and unload LoRA wrappers
6. save a plain merged checkpoint

Run:

    python examples/lora_generation_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from apex.config import get_tiny_config, get_tiny_lora_config
from apex.generation.generator import APEX1Generator, GenerationConfig
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    count_lora_modules,
    has_lora_adapters,
    save_lora_adapters,
)
from apex.model.lora_inference import load_lora_model_for_inference, load_merge_and_unload_lora_model
from apex.tokenizer.tokenizer import APEX1Tokenizer


def main() -> None:
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        adapter_path = tmp_dir / "adapter_final.pt"
        merged_path = tmp_dir / "merged_model.pt"

        # Create a tiny LoRA model and save adapter weights.
        lora_config = get_tiny_lora_config()
        lora_model = APEX1Model(lora_config)
        save_lora_adapters(lora_model, adapter_path, peft_config=lora_config.peft)

        print(f"Saved adapter: {adapter_path}")
        print(f"LoRA modules in training model: {count_lora_modules(lora_model)}")

        # Load the adapter into a fresh inference model.
        infer_config = get_tiny_lora_config()
        result = load_lora_model_for_inference(
            config=infer_config,
            adapter=adapter_path,
            checkpoint=None,
            device="cpu",
            merge_for_runtime=True,
        )

        print(f"Loaded adapter format: {result.adapter_info.get('format')}")
        print(f"Inference LoRA modules: {result.lora_modules}")
        print(f"Merged for runtime: {result.merged_for_runtime}")

        # Run a tiny generation call.
        tokenizer = APEX1Tokenizer()
        generator = APEX1Generator(
            result.model,
            GenerationConfig(
                max_new_tokens=8,
                temperature=0.8,
                top_p=0.9,
                top_k=20,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        input_ids = torch.tensor([tokenizer.encode("Hello", add_special_tokens=True)])
        output = generator.generate(input_ids)

        print(f"Generated token ids: {output.token_ids}")
        print(f"Generated token count: {output.total_tokens}")

        # Load, merge, unload, and save a plain merged checkpoint.
        merge_config = get_tiny_lora_config()
        merged_result = load_merge_and_unload_lora_model(
            config=merge_config,
            adapter=adapter_path,
            checkpoint=None,
            device="cpu",
        )
        assert not has_lora_adapters(merged_result.model)
        torch.save(
            {
                "format": "apex_merged_lora_checkpoint",
                "version": "2.6.0",
                "model_state_dict": merged_result.model.state_dict(),
            },
            merged_path,
        )

        # Show that the merged state dict can load into a plain APEX model.
        plain_config = get_tiny_config()
        plain_config.peft.enabled = False
        plain_model = APEX1Model(plain_config)
        checkpoint = torch.load(merged_path, map_location="cpu", weights_only=False)
        plain_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        print(f"Saved merged checkpoint: {merged_path}")
        print("Merged checkpoint loaded into plain APEX-1 model successfully")


if __name__ == "__main__":
    main()
