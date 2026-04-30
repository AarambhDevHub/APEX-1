# 17 — Optimizer & Learning Rate Scheduler

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/training/scheduler.py`  
> **You will learn:** What AdamW is, why learning rate scheduling matters, and how cosine warmup works.

---

## 1. What Is an Optimizer?

During training, after computing the loss, we calculate **gradients** — the direction in which each parameter should move to reduce the loss. The optimizer applies those gradients to update the parameters.

### The Hill-Climbing Analogy

Imagine you are blindfolded on a hilly landscape, trying to reach the lowest valley (minimum loss). You can only feel which direction is downhill at your current position (that is the gradient). The **optimizer** decides:
- How big a step to take (learning rate)
- Whether to remember previous steps (momentum)
- How to handle parameters that rarely get updated (adaptive learning rate)

---

## 2. AdamW — Adaptive Momentum with Weight Decay

AdamW (Adam with decoupled weight decay) is the standard optimizer for modern LLMs.

### Step 1: Gradient Momentum (First Moment)

Instead of using the raw gradient $g_t$, maintain an exponential moving average:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$\beta_1 = 0.9$ means: 90% past momentum + 10% new gradient. This smooths out noisy gradients.

### Step 2: Gradient Variance (Second Moment)

Track how "spiky" each gradient is:

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

$\beta_2 = 0.95$. Large $v_t$ → high variance → gradient is noisy → take smaller step.

### Step 3: Bias Correction

Early in training, $m_t$ and $v_t$ start at 0 and are biased toward 0. Correct:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

### Step 4: Parameter Update with Weight Decay

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} - \alpha \cdot \lambda \cdot \theta_{t-1}$$

where:
- $\alpha$ = learning rate
- $\varepsilon = 10^{-8}$ = numerical stability
- $\lambda$ = weight decay (e.g., 0.1) — shrinks parameters toward zero each step

**Why weight decay?** Without it, parameters can grow very large — this is called overfitting. Weight decay acts as regularisation, keeping parameters small and forcing the model to use its capacity efficiently.

**AdamW vs Adam:** In vanilla Adam, weight decay is applied to the adaptive update (wrong — it interacts with the variance estimate). AdamW applies it directly to the parameter (correct, independent of the gradient scale).

---

## 3. Learning Rate Scheduling

The learning rate $\alpha$ should not be constant throughout training.

### Why Warmup?

Early in training, the model's parameters are random. With a high learning rate, random gradients cause chaotic, large updates that destabilise training.

**Warmup:** Slowly increase $\alpha$ from 0 to $\alpha_{peak}$ over the first `warmup_steps` steps:

$$\alpha_t = \alpha_{peak} \times \frac{t}{\text{warmup\_steps}}, \quad t \leq \text{warmup\_steps}$$

### Why Cosine Decay?

After warmup, the model is in a good region. Now we want to make progressively finer adjustments. Cosine decay smoothly reduces the learning rate:

$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{peak} - \alpha_{min})\left(1 + \cos\!\left(\pi \cdot \frac{t - \text{warmup}}{\text{max\_steps} - \text{warmup}}\right)\right)$$

This curves from $\alpha_{peak}$ at the start of decay to $\alpha_{min}$ at the end. The cosine shape decays slowly at first (when the model is learning large patterns) and slowly at the end (when fine-tuning small details).

$\alpha_{min} = \alpha_{peak} \times r$ where $r = 0.1$ (the model never goes below 10% of peak LR).

---

## 4. Full Annotated Source: `apex/training/scheduler.py`

```python
"""
Cosine Warmup Learning Rate Scheduler for APEX-1.

Phase 1: Linear warmup from 0 to peak_lr over warmup_steps steps.
Phase 2: Cosine decay from peak_lr to min_lr over remaining steps.
"""

import math
import torch


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup + cosine decay.
    
    Args:
        optimizer:    The PyTorch optimizer to control.
        peak_lr:      Maximum learning rate (reached at end of warmup).
        warmup_steps: Number of steps for linear warmup.
        max_steps:    Total training steps.
        min_lr_ratio: Final LR = peak_lr × min_lr_ratio. Default: 0.1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = peak_lr * min_lr_ratio

        # Current step counter (starts at 0)
        self.current_step = 0

    def get_lr(self) -> float:
        """Compute the learning rate for the current step."""
        t = self.current_step

        if t < self.warmup_steps:
            # Phase 1: Linear warmup
            # LR grows from 0 to peak_lr over warmup_steps steps
            # At t=0: LR=0, at t=warmup_steps-1: LR≈peak_lr
            return self.peak_lr * t / max(self.warmup_steps, 1)

        elif t >= self.max_steps:
            # After training: hold at min_lr
            return self.min_lr

        else:
            # Phase 2: Cosine decay
            # progress goes from 0 (start of decay) to 1 (end of training)
            progress = (t - self.warmup_steps) / max(
                self.max_steps - self.warmup_steps, 1
            )
            # Cosine: starts at 1, decays to 0
            # Range: [0, π] → cos goes from 1 to -1, halved and shifted to [1, 0]
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale to [min_lr, peak_lr]
            return self.min_lr + cosine_factor * (self.peak_lr - self.min_lr)

    def step(self) -> None:
        """Advance one training step and update optimizer LR."""
        self.current_step += 1
        lr = self.get_lr()

        # Update all parameter groups in the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        self.current_step = state["current_step"]
        self.peak_lr = state["peak_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps = state["max_steps"]
        self.min_lr = state["min_lr"]
```

---

## 5. The LR Curve Visualised

```
LR
 |
peak_lr ──────────── ╮
 |                    ╰──── cosine decay
 |      /                              ╰──
 | warmup                                  ──── min_lr
 |
 └────────────────────────────────────────────→ step
      warmup_steps                   max_steps
```

---

## 6. Choosing Hyperparameters

| Parameter | APEX-1 Default | Why |
|---|---|---|
| `peak_lr` | 3e-4 | Standard for 512d models; scale down for larger |
| `warmup_steps` | 1000 | 1–2% of training steps |
| `weight_decay` | 0.1 | Moderate regularisation |
| `beta1` | 0.9 | Standard momentum |
| `beta2` | 0.95 | Slightly more aggressive than default (0.999) |
| `eps` | 1e-8 | Standard numerical stability |
| `min_lr_ratio` | 0.1 | Final LR = 10% of peak |

---

**Next:** [18 — Training Pipeline →](18-training-pipeline.md)
