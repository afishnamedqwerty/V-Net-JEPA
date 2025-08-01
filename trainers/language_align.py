# Contrastive alignment (InfoNCE + Energy regularizer)
# trainers/language_align.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.vjepa.vit import HNetViT
from losses.energy import EnergyLoss
from utils.misc import NTXentLoss  # Custom contrastive providing InfoNCE over batches

# Optional simple text encoder placeholder (replace with CLIP/BERT as needed)
class DummyTextEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.out_dim = out_dim
        self.token_proj = nn.Embedding(30522, out_dim)  # mimic BERT vocab size

    def forward(self, texts):
        # texts: list of token id tensors or raw strings -> here we fake pooled embedding
        if isinstance(texts, (list, tuple)):
            B = len(texts)
        else:
            B = texts.size(0)
        return torch.randn(B, self.out_dim, device=self.token_proj.weight.device)

def language_align(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HNetViT(embed_dim=256).to(device)  # H-Net ViT backbone
    text_encoder = DummyTextEncoder(out_dim=768).to(device)  # Replace with real BERT/CLIP encoder

    proj_video = nn.Linear(256, 256).to(device)
    proj_text = nn.Linear(768, 256).to(device)  # Project text to shared space

    params = list(model.parameters()) + list(proj_video.parameters()) + list(proj_text.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=getattr(cfg, 'lr', 1e-4), weight_decay=getattr(cfg, 'weight_decay', 0.05))

    # Expect an external dataset/loader wiring; here assumed provided by cfg
    loader = cfg.loader if hasattr(cfg, 'loader') else DataLoader(getattr(cfg, 'dataset'), batch_size=getattr(cfg, 'batch_size', 8))

    energy = EnergyLoss(mode='contrastive').to(device)
    ntxent = NTXentLoss(temperature=getattr(cfg, 'temperature', 0.07))

    model.train()
    for epoch in range(getattr(cfg, 'epochs', 50)):
        for batch in loader:
            video = batch['video'].to(device)  # B T H W C
            texts = batch['text']             # list of strings or token ids

            # Forward H-Net ViT to get chunk-level embeddings, then pool to a video-level descriptor via mean
            z_chunks = model(video)           # B x M' x D
            z_vid = z_chunks.mean(dim=1)      # B x D

            # Text encoding and projection
            z_text = text_encoder(texts)      # B x 768
            v = nn.functional.normalize(proj_video(z_vid), dim=-1)  # B x 256
            t = nn.functional.normalize(proj_text(z_text), dim=-1)  # B x 256

            # Contrastive alignment + energy compatibility regularizer
            loss_contrast = ntxent(v, t)
            loss_energy = energy(v, t)
            loss = loss_contrast + getattr(cfg, 'lambda_energy', 0.1) * loss_energy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
