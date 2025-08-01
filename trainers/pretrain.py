# SSv2 masked prediction loop (Energy + VICRegL regularization)
# trainers/pretrain.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.vjepa.vit import HNetViT
from models.vjepa.predictor import Predictor
from losses.energy import EnergyLoss
from losses.vicregl import VICRegLLoss
from losses.auxiliaries import RatioLoss, EntropyReg, BoundaryReg
# from data.ssv2 import SSv2Dataset
from utils.logging import Logger
from utils.misc import get_device

def random_chunk_mask(z_chunks, mask_ratio=0.4, generator: torch.Generator = None):
    # z_chunks: B x M' x D
    B, M, D = z_chunks.shape
    num_mask = max(1, int(M * mask_ratio))
    perms = []
    for _ in range(B):
        perm = torch.randperm(M, generator=generator, device=z_chunks.device)
        perms.append(perm[:num_mask])
    mask_idx = torch.stack(perms, dim=0)  # B x num_mask
    return mask_idx

def gather_indices(x, idx):
    # x: B x M x D, idx: B x K
    B, M, D = x.shape
    K = idx.size(1)
    idx_exp = idx.unsqueeze(-1).expand(B, K, D)
    return torch.gather(x, 1, idx_exp)

def pretrain(cfg):
    device = get_device()
    embed_dim = getattr(cfg, 'embed_dim', 256)

    model = HNetViT(embed_dim=embed_dim).to(device)
    predictor = Predictor(embed_dim=embed_dim,
                          num_layers=getattr(cfg, 'pred_layers', 6),
                          num_heads=getattr(cfg, 'pred_heads', 4)).to(device)

    params = list(model.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=getattr(cfg, 'lr', 1e-4), weight_decay=getattr(cfg, 'weight_decay', 0.05))

    # dataset = SSv2Dataset(mode='train')
    loader = cfg.loader if hasattr(cfg, 'loader') else DataLoader(getattr(cfg, 'dataset'), batch_size=getattr(cfg, 'batch_size', 8), shuffle=True)

    energy_loss = EnergyLoss().to(device)
    vicregl = VICRegLLoss(inv_weight=getattr(cfg, 'vic_inv', 25.0),
                          var_weight=getattr(cfg, 'vic_var', 25.0),
                          cov_weight=getattr(cfg, 'vic_cov', 1.0))
    ratio_loss = RatioLoss(target_ratio=getattr(cfg, 'compression_ratio', 0.5))
    entropy_reg = EntropyReg()
    boundary_reg = BoundaryReg()

    for epoch in range(getattr(cfg, 'epochs', 200)):
        # Deterministic generator seeded per-epoch for identical masks across ranks
        gen = torch.Generator(device=device)
        seed_base = getattr(cfg, 'seed', 42)
        gen.manual_seed(seed_base + epoch)

        for batch in loader:
            x = batch['video'].to(device)  # B T H W C

            # Forward through H-Net ViT to get contextualized chunks (B x M' x D)
            z_ctx = model(x)                 # contextualized chunks
            # Prepare EMA target
            model.update_target()
            with torch.no_grad():
                z_tgt = model.target_encoder(x)  # stopgrad target

            # Create random + optional semantic mask
            mask_ratio = getattr(cfg, 'mask_ratio', 0.4)
            mask_idx = random_chunk_mask(z_ctx, mask_ratio=mask_ratio, generator=gen)  # B x K

            # Gather masked targets and provide context to predictor
            z_tgt_masked = gather_indices(z_tgt, mask_idx)  # B x K x D

            # Predictor forecasts masked tokens conditioned on full context
            z_pred_masked = predictor(z_ctx, mask_idx)      # B x K x D

            # Energy between predicted and target masked embeddings
            loss_e = energy_loss(z_pred_masked, z_tgt_masked)

            # VICRegL on two stochastic views of chunks (use dropout/noise implicitly from model; here reuse ctx vs tgt)
            loss_vic, _ = vicregl(z_ctx, z_tgt)

            # Optional routing regularizers if available from model (placeholders: zero if not exposed)
            loss_ratio = torch.tensor(0.0, device=device)
            loss_ent = torch.tensor(0.0, device=device)
            loss_bound = torch.tensor(0.0, device=device)

            total = loss_e + getattr(cfg, 'lambda_vic', 1.0) * loss_vic \
                    + getattr(cfg, 'alpha', 0.0) * loss_ratio \
                    + getattr(cfg, 'gamma', 0.0) * loss_ent \
                    + getattr(cfg, 'delta', 0.0) * loss_bound

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        Logger.log(f"Epoch {epoch}: E={loss_e.item():.4f}, VICRegL={loss_vic.item():.4f}, Total={total.item():.4f}")
