# Energy function (chunk rep, computation, norm, noisy handling; Huber robust)
# losses/energy.py
import torch
import torch.nn as nn

def energy(f_x, g_y, w, mask=None, robust=True):
    diff = f_x - g_y
    if robust:
        delta = diff.abs()
        loss = torch.where(delta <= 1, 0.5 * delta**2, delta - 0.5)  # Huber
    else:
        loss = diff**2
    loss = loss.mean(dim=-1)  # Per chunk
    
    if mask is not None:
        loss = loss * mask
    
    w_norm = torch.softmax(w, dim=-1)
    e = (w_norm * loss).sum(dim=-1).mean()  # Batch mean
    
    return e

def entropy_reg(p):
    return -p * torch.log(p + 1e-6)

def boundary_reg(z):
    return (z[:,1:] - z[:, :-1])**2 .mean()

# In trainer: L = energy(...) + gamma * entropy_reg(p).mean() + delta * boundary_reg(z)


class EnergyLoss(nn.Module):
    """
    Flexible wrapper for energy-style losses.
    Supports calls:
      - forward(pred, target)
      - forward(context, pred, target)
      - optional mask keyword for (pred, target)
    """
    def __init__(self, robust: bool = True, **kwargs):
        super().__init__()
        self.robust = robust

    def _pair_energy(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        if self.robust:
            delta = diff.abs()
            loss = torch.where(delta <= 1, 0.5 * delta**2, delta - 0.5)
        else:
            loss = diff**2
        return loss.mean(dim=-1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        mask = kwargs.pop('mask', None)
        if len(args) == 2:
            z_pred, z_tgt = args
            loss = self._pair_energy(z_pred, z_tgt)
            if mask is not None:
                loss = loss * mask
            return loss.mean()
        elif len(args) == 3:
            # context, pred, target -> ignore context in base energy; extension point for conditioned variants
            _, z_pred, z_tgt = args
            loss = self._pair_energy(z_pred, z_tgt)
            return loss.mean()
        else:
            raise TypeError("EnergyLoss.forward expects (pred, target) or (context, pred, target)")
