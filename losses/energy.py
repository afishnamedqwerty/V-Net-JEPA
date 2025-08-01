# Energy function (chunk rep, computation, norm, noisy handling; Huber robust)
# losses/energy.py (Bonus for completeness)
import torch

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