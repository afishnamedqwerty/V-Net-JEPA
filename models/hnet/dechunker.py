# Smoothing (EMA) + upsampler (repeat + STE confidence)
# models/hnet/dechunker.py (Bonus for completeness)
import torch
import torch.nn as nn

class Dechunker(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_processed, p, positions):
        B, M, D = z_processed.shape
        N_f = p.size(1)
        
        # Chunk lengths from p sum
        chunk_lengths = p.sum(dim=1).clamp(min=1).long()  # B x M
        
        # Smoothing: EMA across chunks
        alpha = 0.9  # EMA factor
        smoothed = z_processed.clone()
        for m in range(1, M):
            smoothed[:, m] = alpha * smoothed[:, m] + (1 - alpha) * smoothed[:, m-1]
        
        # Upsample: Repeat by lengths
        upsampled = torch.repeat_interleave(smoothed, chunk_lengths, dim=1)  # B x ~N_f x D (approx)
        
        # Confidence: Max p per position
        c = p.max(dim=2)[0].clamp(0.01, 0.99)  # B x N_f
        c_ste = c.detach().round() + c - c.detach()  # STE
        
        upsampled = upsampled * c_ste.unsqueeze(-1)
        
        return upsampled
