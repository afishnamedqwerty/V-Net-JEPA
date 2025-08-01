# Smaller ViT predictor for masked chunks
import torch
import torch.nn as nn

# Predictor for Masked Latent Feature Prediction (smaller ViT, operates on chunk embeddings)
class Predictor(nn.Module):
    def __init__(self, embed_dim=256, num_layers=6, num_heads=4, mlp_ratio=4, drop_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)  # Predict latent features

    def forward(self, x_context, mask_indices):
        # x_context: B M' D (contextualized chunks from encoder)
        # mask_indices: Indices of masked chunks to predict
        x = x_context
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        predictions = self.head(x[:, mask_indices])  # Predict only masked: B num_masked D
        return predictions

# Reuse TransformerLayer from vit.py (assume imported or duplicated)
class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop):
        super().__init__()
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
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

# Adaptations: Handles dynamic chunks (variable M'); self-attention for context-aware prediction of masked latents, per JEPA specs for video.
