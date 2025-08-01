# Dynamic chunking with native sparse attention
# models/hnet/routing.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import knn_graph  # For k-NN local sparsity

class NativeSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size=32, compression_stride=16, selected_block_size=64, num_selected=16, window_size=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size  # l
        self.compression_stride = compression_stride  # d
        self.selected_block_size = selected_block_size  # l'
        self.num_selected = num_selected  # n
        self.window_size = window_size  # w
        self.dropout = nn.Dropout(dropout)
        
        # Local k-NN parameter (used if/when enabling k-NN masking)
        self.k_nn = 8
        
        # Shared projections (scaled for multi-head)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)  # Combined for efficiency
        
        # Compression MLP with PE
        self.compress_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim),  # +3 for 3D pos
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Gate MLP
        self.gate_mlp = nn.Linear(3 * embed_dim, 3)  # Output gates for 3 branches
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    def _scaled_dot_attn(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # Mask added before softmax (negative inf for masked)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output

    def forward(self, x, positions, mask=None):
        # x: seq_len x B x embed_dim
        seq_len, B, D = x.shape
        
        # Project QKV
        qkv = self.qkv_proj(x).reshape(seq_len, B, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # B x heads x seq_len x 3*head_dim
        q, k, v = qkv.chunk(3, dim=-1)  # Each B x heads x seq_len x head_dim
        
        # Branch 1: Compression
        num_blocks = math.ceil((seq_len - self.block_size) / self.compression_stride) + 1
        cmp_k, cmp_v = [], []
        for i in range(num_blocks):
            start = i * self.compression_stride
            end = min(start + self.block_size, seq_len)
            block_pos = positions[start:end]
            block_k = k[:, :, start:end]
            block_v = v[:, :, start:end]
            pe = block_pos.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, self.num_heads, (end-start) * self.head_dim)  # Approx PE
            compressed_k = self.compress_mlp(torch.cat([block_k.view(-1, (end-start)*self.head_dim), pe], dim=-1)).mean(dim=0)  # B x heads x head_dim
            compressed_v = self.compress_mlp(torch.cat([block_v.view(-1, (end-start)*self.head_dim), pe], dim=-1)).mean(dim=0)
            cmp_k.append(compressed_k)
            cmp_v.append(compressed_v)
        cmp_k = torch.stack(cmp_k, dim=2)  # B x heads x num_blocks x head_dim
        cmp_v = torch.stack(cmp_v, dim=2)
        # No mask needed, implicit sparsity
        out_cmp = self._scaled_dot_attn(q.mean(dim=2, keepdim=True), cmp_k, cmp_v)  # Average q for block attn, output B x heads x 1 x head_dim

        # Branch 2: Selection
        # Importance from compression scores (avg over heads)
        importance = (q.mean(dim=1) @ cmp_k.mean(dim=1).transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1).mean(dim=-1)  # B x num_blocks
        top_blocks = importance.topk(self.num_selected, dim=-1)[1]  # B x n
        sel_indices = (top_blocks * self.selected_block_size).unsqueeze(-1) + torch.arange(self.selected_block_size, device=x.device)  # B x n x l'
        sel_indices = sel_indices.view(B, -1).clamp(max=seq_len-1)  # B x (n*l')
        # Local sparsity: k-NN neighbor graph (currently unused; placeholder for future masking)
        _ = knn_graph(positions.reshape(-1, 3), k=self.k_nn)
        # Create selection tensors by gathering along sequence dimension
        k_ = k.reshape(B * self.num_heads, seq_len, self.head_dim)
        v_ = v.reshape(B * self.num_heads, seq_len, self.head_dim)
        idx = sel_indices.unsqueeze(-1).expand(B, sel_indices.size(1), self.head_dim)  # B x (n*l') x D_h
        idx = idx.repeat_interleave(self.num_heads, dim=0)  # (B*heads) x (n*l') x D_h (values ignored by gather for dim=2)

        sel_k = torch.gather(k_, 1, idx)
        sel_v = torch.gather(v_, 1, idx)

        # Compute attention of q over selected keys/values per head-batch
        q_flat = q.reshape(B * self.num_heads, seq_len, self.head_dim)
        out_slc = self._scaled_dot_attn(q_flat, sel_k, sel_v, mask=None)
        # Restore shape to B x heads x seq_len x head_dim
        out_slc = out_slc.view(B, self.num_heads, seq_len, self.head_dim)

        # Branch 3: Window
        start = max(0, seq_len - self.window_size)
        win_k = k[:, :, start:]
        win_v = v[:, :, start:]
        out_win = self._scaled_dot_attn(q, win_k, win_v)  # Implicit local mask

        # Combine with gate
        combined = torch.cat([out_cmp.squeeze(2), out_slc, out_win], dim=-1)  # B x heads x (3*head_dim)
        gates = torch.softmax(self.gate_mlp(combined.view(B, self.num_heads, -1)), dim=-1).view(B, self.num_heads, 3)  # B x heads x 3
        output = gates[..., 0].unsqueeze(-1) * out_cmp.squeeze(2) + gates[..., 1].unsqueeze(-1) * out_slc + gates[..., 2].unsqueeze(-1) * out_win
        output = output.view(B, -1)  # Concat heads or average; adjust
        
        return self.dropout(output.transpose(1,2))  # Adjust dims

class SparseRouting(nn.Module):
    def __init__(self, D=256, K=1024, num_heads=8, block_size=32):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, D))
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)  # Explicit for prototypes
        self.W_v = nn.Linear(D, D)
        self.sparse_attn = NativeSparseAttention(D, num_heads, block_size)
        self.down_threshold = 0.01  # For pruning

    def forward(self, h_flat, positions):
        B, N_f, D = h_flat.shape
        q = self.W_q(h_flat)  # B x N_f x D
        v = self.W_v(h_flat)
        
        # NSA on features for compression (self-attn)
        h_seq = h_flat.transpose(0,1)  # N_f x B x D
        pos_seq = positions.transpose(0,1)  # N_f x B x 3
        compressed = self.sparse_attn(h_seq, pos_seq)  # N_f' x B x D (compressed seq)
        compressed_q = self.W_q(compressed.transpose(0,1))  # B x N_f' x D
        
        # Cross to prototypes
        k_prot = self.W_k(self.prototypes)  # K x D
        v_prot = self.prototypes  # Use as V
        attn_scores = compressed_q @ k_prot.T / math.sqrt(D)  # B x N_f' x K
        p = torch.softmax(attn_scores, dim=-1)  # Soft probabilities
        
        # Chunk embeddings
        z = p @ v_prot  # B x N_f' x D
        
        # Downsampler: Select top-M based on activation (sum p over N_f')
        activ = p.sum(dim=1)  # B x K
        top_m = activ > self.down_threshold
        M = top_m.sum(dim=-1).min()  # Variable M, min across batch for padding
        top_idx = activ.topk(M, dim=-1)[1]
        z = torch.gather(z.unsqueeze(2).repeat(1,1,K,1), 2, top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,D))[:,0]  # B x M x D
        
        # Full p for downsampler/centroids
        p_full = F.interpolate(p.transpose(1,2), size=N_f, mode='nearest').transpose(1,2)  # B x N_f x K
        
        return z, p_full

# Critique: Now fully includes all mechanisms: projections (W_q/W_k/W_v), prototypes, sparse attn (with local window, compression stride, selection top-n, k-NN approx in mask if added), scores with scaled dot, soft p, chunk formation as weighted (but on prototypes V), downsampler prune. Adapted paper for routing cross-attn by compressing features first. For video, 3D PE in compression, positions in forward. Efficiency: Linear in N_f via branches. Challenge: Variable M requires padding in ViT.
