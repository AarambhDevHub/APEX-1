# 10 — Mixture of Experts (MoE)

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/model/ffn.py` — class `MoEFFN`  
> **You will learn:** What MoE is, the 3-tier hierarchy, token routing, and the BUG-08 dispatch fix.

---

## 1. The Problem: Dense Models Are Expensive

In a dense model, every token passes through the same FFN every layer. For a 900B parameter model, every single token activates all 900B parameters — enormously expensive.

**Key insight:** Different tokens need different processing. A code token and a poetry token should not use the same weights!

---

## 2. What Is Mixture of Experts?

**MoE** replaces a single FFN with many **experts** (each is a separate DenseFFN). A **router** selects which experts to activate for each token:

```
Token x → Router → picks expert #3 and #7 → output = weighted sum
```

**The power of MoE:**
- 256 experts total → 900B total parameters
- Only 2 experts active per token → ~45B active parameters
- Same inference cost as a 45B dense model!
- But knowledge capacity of 900B parameters

---

## 3. APEX-1's 3-Tier Hierarchy

APEX-1 uses a 3-tier expert structure:

```
MoEFFN(x)
├── Shared Experts (always active)    ← n_shared = 1
│   └── DenseFFN (processes ALL tokens)
│
└── Routed Experts (conditionally active)
    ├── Router: Linear(d_model, n_experts) → scores
    ├── TopK(scores) → picks n_active = 2 experts
    └── Each selected expert processes x, weighted sum
```

**Tier 1 — Shared experts:** Always active. Handle common, domain-general knowledge.

**Tier 2 — Routed experts:** Only 2 of 256 activate per token. Each specialises in different content types (code, math, language, etc.).

**Tier 3 — Router:** A tiny linear layer that decides which experts to use.

---

## 4. The Routing Math

Given token representation $x \in \mathbb{R}^{d_{model}}$:

**Step 1: Compute routing scores**
$$s = W_{router}\, x + b_{bias} \in \mathbb{R}^{n_{experts}}$$

where $b_{bias}$ is the load balancer's bias term (to prevent expert collapse).

**Step 2: Select top-k experts**
$$\text{indices} = \text{TopK}(s,\, k=n_{active})$$

**Step 3: Compute weights via softmax over selected scores**
$$w_i = \frac{e^{s_i}}{\sum_{j \in \text{indices}} e^{s_j}}$$

**Step 4: Weighted sum of expert outputs**
$$\text{output} = \sum_{i \in \text{indices}} w_i \cdot \text{Expert}_i(x) + \text{SharedExpert}(x)$$

---

## 5. The BUG-08 Story: Dispatch Logic

When multiple tokens in a batch select the **same** expert, they must be processed together efficiently.

The original dispatch code called `DenseFFN` with a fake batch dimension created by stacking each token:

```python
# ORIGINAL (broken for multiple tokens per expert):
for expert_idx, tokens in grouped:
    out = expert(tokens.unsqueeze(0))   # Always [1, 1, d_model]
    # Crashed if 2+ tokens routed to same expert!
```

**The fix** reshapes the token batch correctly:

```python
# FIXED: reshape to [1, n_tokens_for_this_expert, d_model]
out = expert(tokens.unsqueeze(0))  # [1, n_tokens, d_model]
out = out.squeeze(0)               # [n_tokens, d_model]
```

This correctly handles 0, 1, or many tokens routed to the same expert.

---

## 6. Full Annotated Source: `MoEFFN`

```python
class MoEFFN(nn.Module):
    """3-tier Mixture of Experts Feed-Forward Network.
    
    Args:
        d_model:    Model hidden dimension.
        d_ffn:      Per-expert FFN dimension.
        n_experts:  Total number of routed experts.
        n_active:   Experts activated per token (sparse).
        n_shared:   Always-active shared experts.
        dropout:    Dropout probability.
    """

    def __init__(self, d_model, d_ffn, n_experts, n_active, n_shared, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.n_shared = n_shared
        self.d_model = d_model

        # Router: tiny linear → scores over all experts
        # No bias in the router itself (bias comes from load balancer)
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # n_experts independent DenseFFNs (each is a specialist)
        self.experts = nn.ModuleList([
            DenseFFN(d_model, d_ffn, dropout) for _ in range(n_experts)
        ])

        # n_shared always-active DenseFFNs (domain-general knowledge)
        self.shared_experts = nn.ModuleList([
            DenseFFN(d_model, d_ffn, dropout) for _ in range(n_shared)
        ])

        # Expert bias: adjusted by LoadBalancer (not a trainable parameter!)
        # Starts at zero; nudged up/down by the balancer at each training step
        self.expert_bias = torch.zeros(n_experts)

    def set_expert_bias(self, bias: torch.Tensor) -> None:
        """Update expert bias from the load balancer.
        
        Moves bias to the same device as the router weights
        (important: GPU training needs everything on the same device).
        """
        self.expert_bias = bias.to(self.router.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        B, S, D = x.shape

        # ── Step 1: Flatten batch+seq for routing ────────────────────────
        # Each token is an independent routing decision
        x_flat = x.view(B * S, D)   # [N, d_model] where N = B × S

        # ── Step 2: Compute routing scores ───────────────────────────────
        scores = self.router(x_flat)   # [N, n_experts]

        # Add load balancer bias (not a gradient — just a nudge)
        if self.expert_bias.device != scores.device:
            self.expert_bias = self.expert_bias.to(scores.device)
        scores = scores + self.expert_bias.unsqueeze(0)   # broadcast over N

        # ── Step 3: Select top-k experts ─────────────────────────────────
        # topk_vals: [N, n_active] — scores of selected experts
        # topk_idx:  [N, n_active] — which experts were selected
        topk_vals, topk_idx = scores.topk(self.n_active, dim=-1)

        # Convert scores to weights (softmax over selected experts only)
        topk_weights = torch.softmax(topk_vals, dim=-1)   # [N, n_active]

        # ── Step 4: Dispatch tokens to experts ───────────────────────────
        # output accumulator
        output = torch.zeros_like(x_flat)   # [N, d_model]

        for e in range(self.n_experts):
            # Find which tokens selected this expert, and at which rank
            # token_mask: [N, n_active] — True where topk_idx == e
            token_mask = topk_idx == e   # boolean [N, n_active]

            # Get the row indices of tokens that selected expert e
            # and which slot (0=first choice, 1=second choice, etc.)
            token_rows, slot_ranks = token_mask.nonzero(as_tuple=True)

            if token_rows.numel() == 0:
                continue   # No token chose this expert — skip

            # Gather the tokens destined for expert e
            # tokens_for_e: [n_tokens_for_e, d_model]
            tokens_for_e = x_flat[token_rows]

            # BUG-08 FIX: reshape to [1, n_tokens, d_model] for the DenseFFN
            # (DenseFFN expects [batch, seq, d_model]; we use batch=1)
            expert_out = self.experts[e](tokens_for_e.unsqueeze(0))  # [1, n_t, d_model]
            expert_out = expert_out.squeeze(0)                         # [n_t, d_model]

            # Gather the routing weight for each token-expert pair
            weights_e = topk_weights[token_rows, slot_ranks].unsqueeze(-1)  # [n_t, 1]

            # Add weighted expert output to accumulator
            output.index_add_(0, token_rows, expert_out * weights_e)

        # ── Step 5: Add shared expert contributions ─────────────────────
        # Shared experts always process ALL tokens
        for shared_expert in self.shared_experts:
            # shared_expert expects [batch, seq, d_model]
            shared_input = x_flat.unsqueeze(0)         # [1, N, d_model]
            shared_out = shared_expert(shared_input)   # [1, N, d_model]
            output = output + shared_out.squeeze(0)    # add to all tokens

        # ── Step 6: Reshape back to [batch, seq, d_model] ────────────────
        return output.view(B, S, D)
```

---

## 7. Active vs Total Parameters

| Config | Total Params | Active per Token | Sparsity |
|---|---|---|---|
| Tiny | ~1M | ~0.5M | 50% |
| Small | ~100M | ~40M | 40% |
| Medium | ~7B | ~2B | 29% |
| Large | ~900B | ~45B | 5% |

The Large model has 20× more parameters than a 45B dense model but costs the same to run per token — that is the MoE magic.

---

## 8. Which Layers Use MoE?

With `moe_layer_freq = 2`, MoE is used on all **odd-indexed** layers:

```
Layer 0: Dense FFN   (0 % 2 == 0)
Layer 1: MoE FFN     (1 % 2 != 0)  ← expert routing
Layer 2: Dense FFN
Layer 3: MoE FFN
...
```

This gives roughly half dense + half MoE layers, balancing quality and cost.

---

**Next:** [11 — Dynamic Skip Gate →](11-skip-gate.md)
