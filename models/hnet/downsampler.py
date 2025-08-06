# Semantic-aware pooling (LearnedAttentionPooling with centroids/sizes computation)
# models/hnet/downsampler.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
    # Differentiable Bernoulli via Gumbel noise
    u1 = torch.rand_like(logits)
    u2 = torch.rand_like(logits)
    g1 = -torch.log(-torch.log(u1.clamp_min(1e-6)))
    g2 = -torch.log(-torch.log(u2.clamp_min(1e-6)))
    y = torch.sigmoid((logits + g1 - g2) / tau)
    if hard:
        y_hard = (y > 0.5).to(y.dtype)
        y = y_hard.detach() - y.detach() + y
    return y

class LearnedAttentionPooling(nn.Module):
    """
    Variable-length semantic-aware pooling with optional ratio prediction and ratio regularizer.
    Backward-compatible: if enable_variable=False, behaves like fixed M_out pooling.
    """
    def __init__(
        self,
        D_chunk: int = 256,
        M_out: int = 196,
        num_heads: int = 4,
        enable_variable: bool = True,
        max_queries: Optional[int] = None,  # if None -> use M_out as upper bound
        ratio_pred_dim: Optional[int] = None,  # if provided, enables ratio predictor MLP
        ratio_target: Optional[float] = None,  # desired M/M' (compression) or M'/M (see ratio_mode)
        ratio_mode: str = "keep_frac",  # "keep_frac" -> target is M'/M; "compress" -> target is M/M'
        ratio_loss_weight: float = 0.0,
        gumbel_tau: float = 1.0,
        hard_select: bool = False,
    ):
        super(LearnedAttentionPooling, self).__init__()
        self.M_out = M_out
        self.D_chunk = D_chunk
        self.num_heads = num_heads
        self.enable_variable = enable_variable
        self.max_queries = max_queries if max_queries is not None else M_out

        # Query pool is sized to max_queries; we will select a variable subset per batch.
        self.query_tokens = nn.Parameter(torch.randn(1, self.max_queries, D_chunk))
        self.attention = nn.MultiheadAttention(embed_dim=D_chunk, num_heads=num_heads, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(D_chunk, D_chunk),
            nn.GELU(),
            nn.Linear(D_chunk, D_chunk)
        )

        # Optional ratio predictor that conditions on global chunk stats
        self.ratio_pred_dim = ratio_pred_dim
        if ratio_pred_dim is not None:
            self.ratio_predictor = nn.Sequential(
                nn.Linear(ratio_pred_dim, D_chunk),
                nn.GELU(),
                nn.Linear(D_chunk, 1)  # predicts keep_frac in (0,1) via sigmoid
            )
        else:
            self.ratio_predictor = None

        # Ratio regularizer settings
        self.ratio_target = ratio_target
        self.ratio_mode = ratio_mode
        self.ratio_loss_weight = ratio_loss_weight
        self.gumbel_tau = gumbel_tau
        self.hard_select = hard_select

    def _compute_side_outputs(
        self,
        attn_weights: torch.Tensor,  # B x Q x M
        assignments: Optional[torch.Tensor],
        original_positions: Optional[torch.Tensor],
        B: int,
        Q: int,
        M: int,
        device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if original_positions is not None and assignments is not None:
            assign_norm = assignments.sum(dim=1, keepdim=True).clamp(min=1e-6)  # B x 1 x M
            chunk_centroids_input = torch.einsum('bnm,bn3->bm3', assignments, original_positions) / assign_norm.unsqueeze(-1)  # B x M x 3
            chunk_sizes_input = assignments.sum(dim=1)  # B x M
            centroids = torch.einsum('bqm,bm3->bq3', attn_weights, chunk_centroids_input)  # B x Q x 3
            sizes = torch.einsum('bqm,bm->bq', attn_weights, chunk_sizes_input)  # B x Q
        else:
            centroids = torch.zeros(B, Q, 3, device=device)
            sizes = torch.ones(B, Q, device=device)
        return centroids, sizes

    def _ratio_loss(self, keep_frac: torch.Tensor, M: int) -> torch.Tensor:
        # keep_frac: B in (0,1)
        if self.ratio_target is None or self.ratio_loss_weight <= 0:
            return keep_frac.new_zeros(())
        if self.ratio_mode == "keep_frac":
            target = torch.tensor(self.ratio_target, device=keep_frac.device, dtype=keep_frac.dtype)
            loss = F.l1_loss(keep_frac, target.expand_as(keep_frac))
        elif self.ratio_mode == "compress":
            # target provided as compression ratio r = M / M'
            target = torch.tensor(self.ratio_target, device=keep_frac.device, dtype=keep_frac.dtype)
            # keep_frac = M' / M -> desired keep_frac = 1 / r
            desired = (1.0 / target).clamp(min=1e-3, max=1.0)
            loss = F.l1_loss(keep_frac, desired.expand_as(keep_frac))
        else:
            # Fallback: KL between Bernoulli(keep_frac) and Bernoulli(desired)
            desired = torch.tensor(self.ratio_target, device=keep_frac.device, dtype=keep_frac.dtype)
            eps = 1e-6
            p = keep_frac.clamp(eps, 1 - eps)
            q = desired.clamp(eps, 1 - eps)
            loss = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()
        return loss * self.ratio_loss_weight

    def forward(
        self,
        z_chunks: torch.Tensor,
        chunk_masks: Optional[torch.Tensor] = None,
        original_positions: Optional[torch.Tensor] = None,
        assignments: Optional[torch.Tensor] = None,
        ratio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inputs:
          z_chunks: B x M x D_chunk
          chunk_masks: (optional) B x M, True for PAD to be ignored by attention (PyTorch key_padding_mask semantics)
          original_positions: (optional) B x N_f x 3
          assignments: (optional) B x N_f x M soft assignment from router
          ratio_features: (optional) B x ratio_pred_dim features to condition the ratio predictor

        Returns:
          pooled_chunks: B x Q' x D_chunk (Q' variable if enable_variable else M_out)
          centroids: B x Q' x 3
          sizes: B x Q'
          aux: dict with keys {attn_weights (B x Q x M), keep_mask (B x Q), ratio_loss (scalar), keep_frac (B)}
        """
        B, M, D = z_chunks.shape
        Q = self.max_queries
        device = z_chunks.device

        queries = self.query_tokens.expand(B, -1, -1)  # B x Q x D
        pooled_all, attn_weights = self.attention(
            query=queries,  # B x Q x D
            key=z_chunks,   # B x M x D
            value=z_chunks, # B x M x D
            key_padding_mask=chunk_masks  # B x M (True means ignore)
        )  # pooled_all: B x Q x D, attn: B x Q x M
        pooled_all = self.output_mlp(pooled_all)

        aux: Dict[str, torch.Tensor] = {}
        aux["attn_weights"] = attn_weights

        if not self.enable_variable:
            # Fixed output path: take first M_out queries
            Q_out = min(self.M_out, Q)
            pooled_chunks = pooled_all[:, :Q_out]
            attn_sel = attn_weights[:, :Q_out]
            centroids, sizes = self._compute_side_outputs(attn_sel, assignments, original_positions, B, Q_out, M, device)
            aux["keep_mask"] = torch.ones(B, Q_out, device=device, dtype=pooled_chunks.dtype)
            aux["ratio_loss"] = torch.zeros((), device=device)
            aux["keep_frac"] = torch.full((B,), fill_value=Q_out / max(M, 1), device=device, dtype=pooled_chunks.dtype)
            return pooled_chunks, centroids, sizes, aux

        # Variable path: score per-query, form a differentiable keep mask
        # Use attention summary as saliency: s = mean over keys of attn weights
        saliency = attn_weights.mean(dim=-1)  # B x Q

        # Optionally modulate threshold/target via ratio predictor
        if self.ratio_predictor is not None and ratio_features is not None:
            pred_logit = self.ratio_predictor(ratio_features).squeeze(-1)  # B
            pred_keep_frac = torch.sigmoid(pred_logit).clamp(1e-3, 1.0)  # desired M'/M
        else:
            pred_keep_frac = None

        # Map saliency to logits for Bernoulli via temperatured logit transform
        # Normalize per-batch for stability
        saliency_norm = (saliency - saliency.mean(dim=-1, keepdim=True)) / (saliency.std(dim=-1, keepdim=True) + 1e-6)
        logits = saliency_norm  # B x Q

        # If predictor exists, shift logits to approach target keep fraction by biasing threshold
        if pred_keep_frac is not None:
            # Convert desired keep_frac to approximate logit threshold
            # For a standard logistic with mean 0, std 1, 0 keepfrac ~ threshold around quantile
            # Approximate with bias b so that sigmoid(-b) ~ desired; use inverse-sigmoid
            desired = pred_keep_frac.clamp(1e-3, 1-1e-3)
            b = -torch.log(desired.reciprocal() - 1.0)  # logit(desired)
            logits = logits - b.unsqueeze(-1)

        keep_mask = gumbel_sigmoid(logits, tau=self.gumbel_tau, hard=self.hard_select)  # B x Q in (0,1)
        aux["keep_mask"] = keep_mask

        # Compute expected kept count and keep fraction for loss
        kept = keep_mask.sum(dim=-1)  # B
        keep_frac = kept / max(M, 1)
        aux["keep_frac"] = keep_frac
        ratio_loss = self._ratio_loss(keep_frac, M)
        aux["ratio_loss"] = ratio_loss

        # Apply soft selection to pooled tokens and attn to retain differentiability
        keep_mask_exp = keep_mask.unsqueeze(-1)  # B x Q x 1
        pooled_soft = pooled_all * keep_mask_exp  # softly zero-out dropped tokens
        attn_soft = attn_weights * keep_mask.unsqueeze(-1)  # B x Q x M

        # Optionally, cap to top-K by expected count (still differentiable by straight-through on indices if hard)
        # Here we keep all with non-zero weights to preserve differentiability; consumers can handle masks.

        # Compute side outputs on soft-attn, then trim zeros by dynamic packing downstream
        centroids, sizes = self._compute_side_outputs(attn_soft, assignments, original_positions, B, Q, M, device)

        # Return pooled_soft and keep_mask so downstream can pack variable sequences (ops/varlen_pack.py)
        return pooled_soft, centroids, sizes, aux
