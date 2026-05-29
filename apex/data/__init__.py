"""
APEX-1 Data package.

Dataset tools for pretraining, SFT, and preference data:
- Streaming dataset for large corpora
- Data packing into fixed-length sequences
- SFT formatting with role markers
- Preference data formatting for RLHF/DPO/GRPO
"""

from apex.data.data_loader import create_pretrain_loader, create_sft_loader
from apex.data.dataset import PreferenceDataset, PretrainDataset, SFTDataset
from apex.data.vision_dataset import VisionInstructionDataset

__all__ = [
    "PretrainDataset",
    "SFTDataset",
    "PreferenceDataset",
    "VisionInstructionDataset",
    "create_pretrain_loader",
    "create_sft_loader",
]
