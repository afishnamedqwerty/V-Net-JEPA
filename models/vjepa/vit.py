# ViT encoder (adapt from V-JEPA2 GitHub, 12 layers, 768 dim, EMA target)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hnet.encoder import LowLevelEncoder  # Lowest-level 3D CNN
from models.hnet.routing import SparseRouting  # Dynamic chunking with sparse attention
from models.hnet.downsampler import LearnedAttentionPooling  # Semantic-aware pooling
from models.hnet.pos_enc import AdaptivePosEnc  # Adaptive positional encoding
from typing import Dict
from utils.distrib import zigzag_ring_attention, checkpoint_sequential, get_world_size, is_main_process, barrier

# Helper function to compute positions (grid coords for features)
def compute_positions(shape):
    t, h, w = shape
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing='ij')
    return torch.stack([tt, hh, ww], dim=-1).view(-1, 3).float()  # N_f x 3

# Helper to compute centroids and sizes from assignments p (B x N_f x M) and positions (B x N_f x 3)
def compute_centroids_sizes(p, positions):
    # Centroids: weighted avg positions (B x M x 3)
    centroids = torch.einsum('bnk,bn3->bk3', p, positions) / p.sum(dim=1, keepdim=True).clamp(min=1e-6).unsqueeze(-1)
    # Sizes: sum of assignments per chunk (B x M)
    sizes = p.sum(dim=1)
    return centroids, sizes

# Standard ViT Backbone (adapted for video chunks, with self-attention)
class ViT(nn.Module):
    def __init__(self, embed_dim=256, num_layers=12, num_heads=8, mlp_ratio=4, drop_rate=0.1, checkpoint_layers=True, checkpoint_mode:str="all"):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(num_layers)
        ])
        self.checkpoint_layers = checkpoint_layers
        # Modes: "all", "none", "interval:k", "first:n"
        self.checkpoint_mode = checkpoint_mode

    def _build_layer_fns(self, mask):
        fns = []
        for layer in self.layers:
            def make_fn(l):
                def fn(t):
                    return l(t, mask)
                return fn
            fns.append(make_fn(layer))
        return fns

    def forward(self, x, mask=None):
        if not self.checkpoint_layers or self.checkpoint_mode == "none":
            for layer in self.layers:
                x = layer(x, mask)
            return x

        mode = self.checkpoint_mode
        if mode == "all":
            fns = self._build_layer_fns(mask)
            return checkpoint_sequential(fns, x, use_checkpoint=True)

        if mode.startswith("interval:"):
            try:
                k = int(mode.split(":")[1])
            except Exception:
                k = 2
            for i, layer in enumerate(self.layers):
                if i % k == 0:
                    x = torch.utils.checkpoint.checkpoint(lambda t, l=layer: l(t, mask), x, use_reentrant=False)
                else:
                    x = layer(x, mask)
            return x

        if mode.startswith("first:"):
            try:
                n = int(mode.split(":")[1])
            except Exception:
                n = len(self.layers) // 2
            for i, layer in enumerate(self.layers):
                if i < n:
                    x = torch.utils.checkpoint.checkpoint(lambda t, l=layer: l(t, mask), x, use_reentrant=False)
                else:
                    x = layer(x, mask)
            return x

        # Fallback to all if mode unrecognized
        fns = self._build_layer_fns(mask)
        return checkpoint_sequential(fns, x, use_checkpoint=True)

# Transformer Layer (standard self-attention + MLP)
class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        # x: B x S x D; MultiheadAttention is batch_first
        qkv = self.norm1(x)
        # If distributed and more than 1 rank, enable zig-zag ring attention across ranks along sequence/context
        use_ring = (get_world_size() > 1) and (int(torch.cuda.current_device()) >= 0)
        # Optional env/cfg flags to control ring attention
        try:
            import os
            if "USE_RING_ATTENTION" in os.environ:
                use_ring = use_ring and (os.environ["USE_RING_ATTENTION"] == "1")
        except Exception:
            pass

        if use_ring:
            B, S, D = qkv.shape
            H = self.heads
            hd = D // H
            q = qkv.view(B, S, H, hd).permute(0, 2, 1, 3)  # B H S D_h
            k = q
            v = q
            # Build additive mask shaped [B, H, S, S] if provided as key_padding_mask (B x S), expand to attn mask
            attn_mask = None
            if mask is not None:
                # mask: True means pad; convert to additive -inf on padded keys
                key_mask = mask.unsqueeze(1).unsqueeze(1).expand(B, H, S, S)  # mask broadcast on key positions
                attn_mask = torch.zeros(B, H, S, S, device=x.device, dtype=x.dtype)
                attn_mask[key_mask] = float('-inf')
            out = zigzag_ring_attention(q, k, v, attn_mask)  # B H S D_h
            attn_out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        else:
            attn_out, _ = self.attn(qkv, qkv, qkv, key_padding_mask=mask)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

# Integrated H-Net ViT Encoder (main class, with EMA target)
class HNetViT(nn.Module):
    def __init__(self, embed_dim=256, num_layers=12, num_heads=8, mlp_ratio=4, drop_rate=0.1, momentum=0.996):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = LowLevelEncoder()  # Lowest-level 3D CNN
        self.routing = SparseRouting(D=embed_dim)  # Dynamic chunking
        # Use correct constructor args for pooling (D_chunk and M_out)
        # Enable variable-length pooling (configurable). Defaults safe/backward compatible at inference since ViT accepts masks.
        self.downsampler = LearnedAttentionPooling(
            D_chunk=embed_dim,
            M_out=196,
            num_heads=num_heads,
            enable_variable=True,
            max_queries=256,
            ratio_pred_dim=None,
            ratio_target=None,
            ratio_mode="keep_frac",
            ratio_loss_weight=0.0,
            gumbel_tau=1.0,
            hard_select=False,
        )  # Semantic pooling
        self.pos_enc = AdaptivePosEnc(D=embed_dim)  # Adaptive pos based on centroids/sizes
        # Enable checkpointing in ViT layers by default for activation recomputation
        self.vit = ViT(embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, checkpoint_layers=True)

        # EMA Target Encoder (for stable targets in SSL)
        self.target_encoder = None  # Initialized in update_target
        self.momentum = momentum

    def forward(self, x, is_target=False):
        if is_target and self.target_encoder is not None:
            return self.target_encoder(x)  # Use EMA version for targets

        h = self.encoder(x)  # Dense feature map: B T' H' W' D
        h_flat = h.view(x.size(0), -1, h.size(-1))  # B N_f D
        positions = compute_positions(h.shape[1:4]).to(h.device).expand(x.size(0), -1, -1)  # B N_f 3
        z, p = self.routing(h_flat, positions)  # Chunks: B M D, assignments: B N_f M
        # Pool with awareness of assignments and positions to align centroids/sizes with M'
        # Expect downsampler to optionally provide pooled mask; if not, create zeros mask
        out = self.downsampler(
            z_chunks=z,
            chunk_masks=None,
            original_positions=positions,
            assignments=p
        )
        # Support both old (3-tuple) and new (4-tuple) API. New API returns aux dict instead of mask.
        pooled_mask = None
        aux: Dict[str, torch.Tensor] = {}
        if isinstance(out, tuple) and len(out) == 4:
            z_pooled, centroids, sizes, aux = out
            # Build key_padding_mask for kept tokens if keep_mask provided (True means PAD)
            keep_mask = aux.get("keep_mask", None)  # B x Q in [0,1]
            if keep_mask is not None:
                # consider kept if weight>0; pad if near zero
                pooled_mask = (keep_mask <= 1e-4)
        else:
            z_pooled, centroids, sizes = out
        if pooled_mask is None:
            pooled_mask = torch.zeros(z_pooled.size(0), z_pooled.size(1), dtype=torch.bool, device=z_pooled.device)

        z_pos = self.pos_enc(z_pooled, centroids, sizes)  # Add adaptive pos: B M' D
        return self.vit(z_pos, mask=pooled_mask)  # Contextualized: B M' D

    def update_target(self):
        # Initialize EMA target if needed
        if self.target_encoder is None:
            self.target_encoder = HNetViT(embed_dim=self.vit.embed_dim, num_layers=len(self.vit.layers))
            self.target_encoder.load_state_dict(self.state_dict())

        # Momentum update on all ranks
        with torch.no_grad():
            for param_q, param_k in zip(self.parameters(), self.target_encoder.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        # Ensure all ranks have identical EMA after update
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for p in self.target_encoder.parameters():
                torch.distributed.broadcast(p.data, src=0)
            barrier()

        self.target_encoder.eval()  # Always eval mode
