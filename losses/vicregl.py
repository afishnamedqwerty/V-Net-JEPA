# VICRegL loss (variance/invariance/covariance) adapted for local chunk embeddings
# losses/vicregl.py
import torch
import torch.nn as nn

class VICRegLLoss(nn.Module):
    """
    Computes VICRegL between two chunk-level embeddings.
    Inputs:
      z1, z2: tensors of shape (B, M, D) or (B*N, D)
    Loss terms:
      - invariance: MSE(z1, z2)
      - variance: ensures per-dimension std >= gamma
      - covariance: decorrelates features (off-diagonal of covariance)
    """
    def __init__(self, inv_weight=25.0, var_weight=25.0, cov_weight=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.inv_w = inv_weight
        self.var_w = var_weight
        self.cov_w = cov_weight
        self.gamma = gamma
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # Accept (B, M, D) or (B*N, D); flatten to (N, D)
        if z1.dim() == 3:
            N, D = z1.size(0) * z1.size(1), z1.size(2)
            z1f = z1.reshape(N, D)
            z2f = z2.reshape(N, D)
        elif z1.dim() == 2:
            z1f, z2f = z1, z2
            D = z1f.size(1)
        else:
            raise ValueError("z1/z2 must be (B,M,D) or (N,D)")

        # Invariance (MSE)
        inv = self.mse(z1f, z2f)

        # Variance term: encourage std per dim >= gamma
        def var_loss(z):
            z = z - z.mean(dim=0, keepdim=True)
            std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
            return torch.mean(torch.relu(self.gamma - std))
        var = var_loss(z1f) + var_loss(z2f)

        # Covariance term: off-diagonal of covariance matrix
        def cov_loss(z):
            z = z - z.mean(dim=0, keepdim=True)
            N = z.size(0)
            cov = (z.T @ z) / (N - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            return (off_diag.pow(2).sum()) / D
        cov = cov_loss(z1f) + cov_loss(z2f)

        total = self.inv_w * inv + self.var_w * var + self.cov_w * cov
        return total, {"inv": inv.detach(), "var": var.detach(), "cov": cov.detach(), "total": total.detach()}
