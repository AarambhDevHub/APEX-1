"""
CPU smoke demo for APEX-1 v2.9.0 Adapter-DPO alignment.

This demo creates a tiny preference dataset, trains only LoRA adapter parameters
for a couple of steps, and saves an adapter checkpoint.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from torch.utils.data import DataLoader

from apex.alignment.adapter_dpo import AdapterDPOTrainer, PreferenceJSONLDataset, preference_collate
from apex.config import get_tiny_adapter_dpo_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import count_trainable_parameters
from apex.tokenizer import APEX1Tokenizer


def main() -> None:
    config = get_tiny_adapter_dpo_config(method="lora")
    config.training.max_steps = 3
    config.adapter_dpo.beta = 0.1
    config.adapter_dpo.max_prompt_len = 32
    config.adapter_dpo.max_response_len = 32
    config.validate()

    tokenizer = APEX1Tokenizer()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        data_path = tmpdir / "tiny_preference.jsonl"
        rows = [
            {
                "prompt": "Explain Rust ownership simply.",
                "chosen": "Rust ownership means each value has one owner, and borrowing lets code use it safely.",
                "rejected": "Rust ownership is random and does not matter.",
            },
            {
                "prompt": "What is LoRA?",
                "chosen": "LoRA trains small low-rank adapter matrices while the base model stays frozen.",
                "rejected": "LoRA means training every model weight from scratch.",
            },
        ]
        with data_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        policy_model = APEX1Model(config)

        reference_config = copy.deepcopy(config)
        reference_config.peft.enabled = False
        reference_config.adapter_dpo.enabled = False
        reference_model = APEX1Model(reference_config)

        dataset = PreferenceJSONLDataset(
            data_path,
            tokenizer=tokenizer,
            max_prompt_len=config.adapter_dpo.max_prompt_len,
            max_response_len=config.adapter_dpo.max_response_len,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=preference_collate)

        trainer = AdapterDPOTrainer(policy_model, reference_model, config, loader)
        result = trainer.train(max_steps=3, output_dir=tmpdir / "adapter_dpo")

        print(f"Trainable adapter parameters: {count_trainable_parameters(policy_model):,}")
        print(f"Final DPO loss: {result.get('loss', 0.0):.4f}")
        print(f"Reward margin: {result.get('reward_margin', 0.0):.4f}")
        print(f"Saved adapter: {result['adapter_path']}")


if __name__ == "__main__":
    main()
