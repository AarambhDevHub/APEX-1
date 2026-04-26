"""
Multi-Token Prediction Head for APEX-1.

Implements speculative prediction heads that predict the next N tokens
simultaneously during training. This provides N× richer gradient signal
and enables speculative decoding at inference time for 2-3× throughput.

Training loss with multi-token prediction:
    L = L_main + λ × mean(L_offset_k for k in 1..n_predict)
    λ = 0.1 — speculative heads contribute 10% of gradient signal

At inference, the speculative heads draft tokens that the main model
can verify in a single forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTokenHead(nn.Module):
    """Multi-token prediction head for speculative training and decoding.

    Predicts the next N tokens simultaneously using separate linear
    projection heads. Each head k predicts the token at offset k
    from the current position.

    Args:
        d_model: Model hidden dimension.
        vocab_size: Vocabulary size.
        n_predict: Number of future tokens to predict (default: 4).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_predict: int = 4,
    ) -> None:
        super().__init__()
        self.n_predict = n_predict
        self.d_model = d_model
        self.vocab_size = vocab_size

        # One small projection per future offset (1, 2, 3, 4 steps ahead)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_predict)]
        )

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Compute speculative logits for each future offset.

        Args:
            hidden_states: Final hidden states ``[batch, seq_len, d_model]``.

        Returns:
            List of ``n_predict`` logit tensors, each ``[batch, seq, vocab_size]``.
        """
        return [head(hidden_states) for head in self.heads]

    def draft_tokens(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Draft speculative tokens for speculative decoding.

        Takes the last position's hidden state and generates draft
        token predictions for the next n_predict positions.

        Args:
            hidden_states: Hidden states, typically ``[batch, 1, d_model]``
                          for the last generated position.
            temperature: Sampling temperature.

        Returns:
            Draft token IDs ``[batch, n_predict]``.
        """
        drafts = []
        for head in self.heads:
            logits = head(hidden_states[:, -1:, :])  # [batch, 1, vocab]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                token = logits.squeeze(1).argmax(dim=-1, keepdim=True)
            drafts.append(token)
        return torch.cat(drafts, dim=-1)  # [batch, n_predict]

    def extra_repr(self) -> str:
        """Return string representation."""
        return f"n_predict={self.n_predict}, d_model={self.d_model}, vocab_size={self.vocab_size}"
