# Adaptive positional (MLP on centroid/size, RoPE adaptation)
# models/hnet/pos_enc.py
import torch
import torch.nn as nn

class AdaptivePosEnc(nn.Module):
    def __init__(self, D=256, rope_dim=32):  # rope_dim for RoPE adaptation
        super(AdaptivePosEnc, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, D),  # centroid (3) + size (1)
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.rope_dim = rope_dim  # For RoPE integration

    def forward(self, z_pooled, centroids, sizes):
        # centroids: B x M' x 3, sizes: B x M'
        inputs = torch.cat([centroids, sizes.unsqueeze(-1)], dim=-1)  # B x M' x 4
        base_pos = self.mlp(inputs)  # B x M' x D

        # Adapt RoPE: Scale rotations by size, apply to first rope_dim
        if self.rope_dim > 0:
            # Simple RoPE simulation: rotate based on centroid coords
            theta = 10000 ** (-torch.arange(0, self.rope_dim//2, device=centroids.device) / self.rope_dim)
            angles = centroids.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # B x M' x 3 x (rope_dim//2)
            sin, cos = angles.sin(), angles.cos()
            # Scale by size
            scale = sizes.unsqueeze(-1).unsqueeze(-1) / sizes.max(dim=1, keepdim=True)[0]  # Normalize scale
            sin *= scale
            cos *= scale
            # Apply to z_pooled's first rope_dim dims (example rotation on pairs)
            x1 = z_pooled[..., :self.rope_dim:2]
            x2 = z_pooled[..., 1:self.rope_dim:2]
            rotated = torch.cat([x1 * cos.mean(dim=2) - x2 * sin.mean(dim=2),
                                 x1 * sin.mean(dim=2) + x2 * cos.mean(dim=2)], dim=-1)
            z_pooled = torch.cat([rotated, z_pooled[..., self.rope_dim:]], dim=-1)

        return z_pooled + base_pos
