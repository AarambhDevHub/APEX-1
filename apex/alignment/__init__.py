"""
APEX-1 Alignment package.

Implements Phase 3 alignment methods:
- Reward Model with Bradley-Terry loss
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- Process Reward Model (PRM)
- Constitutional AI critique loop
- Combined reward function
"""

from apex.alignment.reward_model import RewardModel
from apex.alignment.dpo import dpo_loss
from apex.alignment.grpo import grpo_training_step
from apex.alignment.prm import ProcessRewardModel
from apex.alignment.constitutional import ConstitutionalAI
from apex.alignment.combined_reward import combined_reward

__all__ = [
    "RewardModel",
    "dpo_loss",
    "grpo_training_step",
    "ProcessRewardModel",
    "ConstitutionalAI",
    "combined_reward",
]
