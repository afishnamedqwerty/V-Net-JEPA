# Saliency probe (BCE + Energy regularizer)
# trainers/probe_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.vjepa.vit import HNetViT
from losses.energy import EnergyLoss
# Expect a dataset providing chunk-level saliency labels aligned to pooled chunks M'
# from data.ssv2 import SSv2DatasetWithSaliency

def probe_train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load HNetViT backbone; optionally freeze or allow light finetuning of routing
    model = HNetViT(embed_dim=getattr(cfg, 'embed_dim', 256),
                    down_kwargs=getattr(cfg, 'down_kwargs', None)).to(device)
    if getattr(cfg, 'freeze_backbone', True):
        for p in model.parameters():
            p.requires_grad = False

    probe_head = nn.Linear(256, 1).to(device)  # Saliency per chunk

    optimizer = torch.optim.AdamW(
        probe_head.parameters(),
        lr=getattr(cfg, 'lr', 1e-4),
        weight_decay=getattr(cfg, 'weight_decay', 0.05)
    )

    # Wiring: expect caller to provide dataset/loader via cfg
    loader = cfg.loader if hasattr(cfg, 'loader') else DataLoader(getattr(cfg, 'dataset'), batch_size=getattr(cfg, 'batch_size', 8))

    bce = nn.BCEWithLogitsLoss()
    energy = EnergyLoss(mode='saliency').to(device)  # Encourage coherent low energy on salient chunks

    # Optional: configure downsampler ratio if not provided via down_kwargs
    if hasattr(model, 'downsampler'):
        if getattr(model.downsampler, 'ratio_target', None) is None and hasattr(cfg, 'compression_ratio'):
            model.downsampler.ratio_target = getattr(cfg, 'compression_ratio', 0.5)
        if getattr(model.downsampler, 'ratio_loss_weight', 0.0) <= 0 and hasattr(cfg, 'alpha'):
            model.downsampler.ratio_loss_weight = getattr(cfg, 'alpha', 0.0)

    for epoch in range(getattr(cfg, 'epochs', 20)):
        for batch in loader:
            video = batch['video'].to(device)           # B T H W C
            saliency = batch['saliency'].to(device)     # B x M' (binary labels per pooled chunk)

            z_chunks = model(video)                     # B x M' x D
            logits = probe_head(z_chunks).squeeze(-1)   # B x M'

            loss_cls = bce(logits, saliency.float())

            # Energy regularizer on salient chunks: select via mask per batch
            with torch.no_grad():
                mask = saliency > 0.5                   # B x M'
            if mask.any():
                z_sel = z_chunks[mask]                  # (#salient, D)
                loss_e = energy(z_sel, z_sel)           # identity energy minimization for coherence
            else:
                loss_e = torch.tensor(0.0, device=device)

            loss = loss_cls + getattr(cfg, 'lambda_energy', 0.1) * loss_e

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
