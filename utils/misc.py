# Padding/masking utilities, STE ops, simple planners and helpers
# utils/misc.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_sequences(seqs: List[torch.Tensor], pad_value: float = 0.0):
    """
    Pad a list of variable-length tensors (B list) to a single tensor with mask.
    Each tensor is shape (L_i, D). Returns (B, L_max, D), mask (B, L_max) where True means padded.
    """
    B = len(seqs)
    L_max = max(s.size(0) for s in seqs)
    D = seqs[0].size(1)
    device = seqs[0].device
    out = torch.full((B, L_max, D), pad_value, device=device, dtype=seqs[0].dtype)
    mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        L = s.size(0)
        out[i, :L] = s
        mask[i, :L] = False
    return out, mask

def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None):
    """
    lengths: (B,) integer lengths
    returns mask: (B, L_max) with True for positions beyond length (i.e., masked)
    """
    B = lengths.size(0)
    L = int(max_len) if max_len is not None else int(lengths.max().item())
    ar = torch.arange(L, device=lengths.device).unsqueeze(0).expand(B, -1)
    return ar >= lengths.unsqueeze(1)

def ste_round(x: torch.Tensor):
    """
    Straight-through estimator rounding: forward round, backward identity.
    """
    return (x - x.detach()) + x.detach().round()

def ste_clamp01(x: torch.Tensor):
    """
    Clamp to [0,1] with STE on boundaries.
    """
    y = torch.clamp(x, 0.0, 1.0)
    return x + (y - x).detach()

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-Scaled Cross Entropy Loss (InfoNCE) for two modalities.
    Expects already normalized embeddings v and t (B x D). Computes symmetric loss.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, v: torch.Tensor, t: torch.Tensor):
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)
        logits_vt = (v @ t.t()) / self.t  # B x B
        logits_tv = (t @ v.t()) / self.t  # B x B
        targets = torch.arange(v.size(0), device=v.device)
        loss = F.cross_entropy(logits_vt, targets) + F.cross_entropy(logits_tv, targets)
        return loss * 0.5

class CEMPlanner:
    """
    Cross-Entropy Method planner for action-conditioned inference.
    Optimizes a sequence of actions a_{0:H-1} in R^{H x A} to minimize a differentiable energy function.

    Expected energy_fn signature (vectorized over samples):
      energy_fn(actions: Tensor[B, H, A]) -> Tensor[B]
      Lower is better. The function must run under torch.no_grad() during planning.

    Notes:
    - Works with action_posttrain.py by providing a mean action sequence as the plan.
    - Deterministic seeding can be set via utils.misc.set_seed before calling minimize_energy.
    """
    def __init__(
        self,
        samples: int = 256,
        elites: int = 32,
        iters: int = 8,
        action_dim: int = 7,
        init_std: float = 0.5,
        clamp_actions: bool = True,
        action_low: float = -1.0,
        action_high: float = 1.0,
        momentum: float = 0.0
    ):
        assert elites > 0 and samples >= elites, "CEM: samples must be >= elites and elites > 0"
        self.samples = samples
        self.elites = elites
        self.iters = iters
        self.action_dim = action_dim
        self.init_std = init_std
        self.clamp_actions = clamp_actions
        self.action_low = action_low
        self.action_high = action_high
        self.momentum = momentum  # optional smoothing of mean updates

    def minimize_energy(
        self,
        energy_fn,
        horizon: int = 10,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        device=None
    ) -> torch.Tensor:
        """
        Run CEM to find an action sequence minimizing energy_fn.

        Args:
          energy_fn: callable mapping (S, H, A) -> (S,) energies (lower is better).
          horizon: planning horizon H.
          mean: initial mean (H, A). Defaults to zeros.
          std: initial std (H, A). Defaults to init_std.
          device: torch device.

        Returns:
          plan: Tensor (H, A) mean action sequence after optimization.
        """
        device = device or get_device()
        mean = mean if mean is not None else torch.zeros(horizon, self.action_dim, device=device)
        std = std if std is not None else torch.full((horizon, self.action_dim), self.init_std, device=device)

        prev_mean = mean.clone()

        for _ in range(self.iters):
            # Sample actions: (S, H, A)
            noise = torch.randn(self.samples, horizon, self.action_dim, device=device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            if self.clamp_actions:
                actions = actions.clamp(self.action_low, self.action_high)

            # Evaluate energies
            with torch.no_grad():
                costs = energy_fn(actions)  # (S,)
                if costs.dim() != 1 or costs.size(0) != self.samples:
                    raise ValueError("energy_fn must return (samples,) tensor")

            # Select elites
            k = min(self.elites, self.samples)
            vals, idx = torch.topk(-costs, k=k, largest=True)  # lowest cost -> highest negative
            elite = actions[idx]  # (k, H, A)

            # Update mean/std
            elite_mean = elite.mean(dim=0)  # (H, A)
            elite_std = elite.std(dim=0).clamp_min(1e-4)  # (H, A)

            if self.momentum > 0.0:
                mean = self.momentum * mean + (1.0 - self.momentum) * elite_mean
                std = self.momentum * std + (1.0 - self.momentum) * elite_std
            else:
                mean, std = elite_mean, elite_std

            # Optional damping to avoid oscillations
            mean = 0.5 * mean + 0.5 * prev_mean
            prev_mean = mean.clone()

        if self.clamp_actions:
            mean = mean.clamp(self.action_low, self.action_high)

        return mean  # (H, A)

class CrossAttentionFuser(nn.Module):
    """
    Cross-attention fuser for conditioning (actions->video tokens or vice versa).
    Provided here for generic usage; specific trainers may override.
    """
    def __init__(self, dim=256, num_heads=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, query_tokens, key_value_tokens):
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(key_value_tokens)
        fused, _ = self.attn(q, kv, kv)
        return self.out(fused)

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
