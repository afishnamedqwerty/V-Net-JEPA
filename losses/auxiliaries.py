# Ratio loss (from H-Net train.py), entropy/boundary reg
import torch
import math

def ratio_loss(chunk_sizes, target_ratio=4.0):
    """
    Ratio loss to encourage average chunk size close to target_ratio.
    chunk_sizes: Tensor of chunk lengths (B, M)
    """
    mean_size = chunk_sizes.float().mean()
    return torch.abs(mean_size - target_ratio)

def entropy_reg(p):
    """
    Entropy regularization to promote diverse assignments.
    p: Assignment probabilities (B, N_f, K)
    """
    return -torch.sum(p * torch.log(p + 1e-8), dim=-1).mean()

def boundary_reg(z):
    """
    Boundary consistency regularization: L2 diff between adjacent chunks.
    z: Chunk embeddings (B, M, D)
    """
    if z.size(1) < 2:
        return torch.tensor(0.0, device=z.device)
    diff = z[:, 1:] - z[:, :-1]
    return torch.mean(diff ** 2)
