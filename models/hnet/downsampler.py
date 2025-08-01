# Semantic-aware pooling (LearnedAttentionPooling with centroids/sizes computation)
# models/hnet/downsampler.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedAttentionPooling(nn.Module):
    def __init__(self, D_chunk=256, M_out=196, num_heads=4):
        super(LearnedAttentionPooling, self).__init__()
        self.M_out = M_out
        self.D_chunk = D_chunk
        self.num_heads = num_heads
        self.query_tokens = nn.Parameter(torch.randn(1, M_out, D_chunk))
        self.attention = nn.MultiheadAttention(embed_dim=D_chunk, num_heads=num_heads, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(D_chunk, D_chunk),
            nn.GELU(),
            nn.Linear(D_chunk, D_chunk)
        )

    def forward(self, z_chunks, chunk_masks=None, original_positions=None, assignments=None):
        # z_chunks: B x M x D_chunk
        # assignments: p from routing, B x N_f x M (K=M here)
        # original_positions: B x N_f x 3
        B, M, D = z_chunks.shape
        queries = self.query_tokens.expand(B, -1, -1)
        pooled_chunks, attn_weights = self.attention(
            query=queries,
            key=z_chunks,
            value=z_chunks,
            key_padding_mask=chunk_masks
        )
        pooled_chunks = self.output_mlp(pooled_chunks)

        if original_positions is not None and assignments is not None:
            # Chunk centroids input: weighted avg positions per chunk
            assign_norm = assignments.sum(dim=1, keepdim=True).clamp(min=1e-6)  # B x 1 x M
            chunk_centroids_input = torch.einsum('bnm,bn3->bm3', assignments, original_positions) / assign_norm.unsqueeze(-1)  # B x M x 3
            chunk_sizes_input = assignments.sum(dim=1)  # B x M

            # Propagate to pooled via attn_weights: B x M_out x M
            centroids = torch.einsum('bom,bm3->bo3', attn_weights, chunk_centroids_input)  # B x M_out x 3
            sizes = torch.einsum('bom,bm->bo', attn_weights, chunk_sizes_input)  # B x M_out
        else:
            centroids = torch.zeros(B, self.M_out, 3, device=z_chunks.device)
            sizes = torch.ones(B, self.M_out, device=z_chunks.device)

        return pooled_chunks, centroids, sizes
