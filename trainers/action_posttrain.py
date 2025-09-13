# Droid post-training (action fusion, conditioned Energy, CEM inference)
# trainers/action_posttrain.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from losses.energy import EnergyLoss
from models.vjepa.vit import HNetViT
from models.vjepa.predictor import Predictor
# from data.droid import DroidDataset
from utils.misc import CEMPlanner

class CrossAttentionFuser(nn.Module):
    """
    Simple cross-attention: actions as queries over chunk embeddings.
    """
    def __init__(self, dim=256, num_heads=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, action_tokens, chunk_tokens):
        # action_tokens: B x A x D, chunk_tokens: B x M' x D
        q = self.norm_q(action_tokens)
        kv = self.norm_kv(chunk_tokens)
        fused, _ = self.attn(q, kv, kv)
        return self.proj(fused)  # B x A x D

def shift_chunks(z, shift=1):
    # Shift along sequence dimension as simple future target
    return torch.roll(z, shifts=-shift, dims=1)

def action_posttrain(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Backbone and predictor
    model = HNetViT(embed_dim=getattr(cfg, 'embed_dim', 256),
                    down_kwargs=getattr(cfg, 'down_kwargs', None)).to(device)
    if getattr(cfg, 'freeze_encoder', True):
        for p in model.encoder.parameters():
            p.requires_grad = False
    predictor_cond = Predictor(embed_dim=256,
                               num_layers=getattr(cfg, 'pred_layers', 6),
                               num_heads=getattr(cfg, 'pred_heads', 4)).to(device)

    # Action embedding and cross-attention fuser
    action_embed = nn.Linear(getattr(cfg, 'action_dim', 7), 256).to(device)
    fuser = CrossAttentionFuser(dim=256, num_heads=getattr(cfg, 'fuser_heads', 4)).to(device)

    params = list(action_embed.parameters()) + list(fuser.parameters()) + list(predictor_cond.parameters())
    optimizer = torch.optim.AdamW(params, lr=getattr(cfg, 'lr', 1e-4), weight_decay=getattr(cfg, 'weight_decay', 0.05))

    # Dataset/loader wiring expected from cfg
    loader = cfg.loader if hasattr(cfg, 'loader') else DataLoader(getattr(cfg, 'dataset'), batch_size=getattr(cfg, 'batch_size', 8), shuffle=True)

    energy = EnergyLoss(conditioned=True).to(device)

    # Optional: configure downsampler ratio if not provided via down_kwargs
    if hasattr(model, 'downsampler'):
        if getattr(model.downsampler, 'ratio_target', None) is None and hasattr(cfg, 'compression_ratio'):
            model.downsampler.ratio_target = getattr(cfg, 'compression_ratio', 0.5)
        if getattr(model.downsampler, 'ratio_loss_weight', 0.0) <= 0 and hasattr(cfg, 'alpha'):
            model.downsampler.ratio_loss_weight = getattr(cfg, 'alpha', 0.0)

    for epoch in range(getattr(cfg, 'epochs', 50)):
        for batch in loader:
            video = batch['video'].to(device)         # B T H W C
            actions = batch['actions'].to(device)     # B A 7 (sequence of actions)

            # Encode video into chunk tokens
            z_video = model(video)                    # B x M' x D

            # Embed actions and fuse via cross-attention
            z_action = action_embed(actions)          # B x A x D
            z_fused = fuser(z_action, z_video)        # B x A x D

            # Predict future chunk tokens conditioned on actions (use A==M' alignment if needed)
            # Pool fused actions to a single token per video to condition predictions
            z_cond = z_fused.mean(dim=1, keepdim=True)            # B x 1 x D
            z_cond_expanded = z_cond.expand(-1, z_video.size(1), -1)  # B x M' x D

            # Combine condition with video tokens (concat then project or simple sum)
            z_context = z_video + z_cond_expanded                  # B x M' x D

            # Future target: next-step chunks as proxy
            target_future = shift_chunks(z_video, shift=1)         # B x M' x D

            # Predict future from conditioned context; predict all tokens, then compare
            pred_future = predictor_cond(z_context, mask_indices=torch.arange(z_context.size(1), device=device))

            # Conditioned energy between prediction and target future
            loss = energy(z_context, pred_future, target_future)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Inference planning via CEM (sampling in action space to minimize energy)
    planner = CEMPlanner(samples=getattr(cfg, 'cem_samples', 100))
    # Example usage (to be integrated in eval pipeline):
    # plan = planner.minimize_energy(model, initial_obs, goal_spec)
