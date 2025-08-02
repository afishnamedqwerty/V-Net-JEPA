import torch
from typing import List, Tuple, Optional


def pack_varlen(seqs: List[torch.Tensor], device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a list of [L_i, H, D] tensors (per-sequence tokens already projected to heads)
    into a single packed tensor [T, H, D] with cu_seqlens [B+1].

    Args:
        seqs: list of tensors with shapes [L_i, H, D], same dtype and device (or will be moved).
        device: optional device to place outputs.

    Returns:
        packed: [T, H, D] concatenated tensor
        cu_seqlens: [B+1] cumulative sequence lengths, cu_seqlens[0] = 0, cu_seqlens[-1] = T
    """
    if len(seqs) == 0:
        raise ValueError("seqs must be non-empty")

    if device is None:
        device = seqs[0].device

    dtype = seqs[0].dtype
    H = seqs[0].shape[-2]
    D = seqs[0].shape[-1]
    for t in seqs:
        if t.dim() != 3:
            raise ValueError(f"each seq must be [L, H, D], got {t.shape}")
        if t.shape[-2] != H or t.shape[-1] != D:
            raise ValueError("head or dim mismatch among sequences")
        if t.dtype != dtype:
            raise ValueError("dtype mismatch among sequences")

    lengths = [t.shape[0] for t in seqs]
    T = sum(lengths)
    B = len(seqs)
    cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
    if B > 0:
        cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)

    packed = torch.empty((T, H, D), dtype=dtype, device=device)
    offset = 0
    for t in seqs:
        L = t.shape[0]
        packed[offset:offset + L].copy_(t.to(device))
        offset += L
    return packed, cu


def unpack_varlen(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> List[torch.Tensor]:
    """
    Unpack a packed [T, H, D] tensor into a list of [L_i, H, D] tensors using cu_seqlens.

    Args:
        packed: [T, H, D]
        cu_seqlens: [B+1] int32 cumulative lengths

    Returns:
        list of tensors on the same device/dtype as packed
    """
    if packed.dim() != 3:
        raise ValueError("packed must be [T, H, D]")
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be 1D [B+1]")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be int32 or int64")

    outs = []
    B = cu_seqlens.numel() - 1
    for i in range(B):
        s = int(cu_seqlens[i].item())
        e = int(cu_seqlens[i + 1].item())
        outs.append(packed[s:e])
    return outs


def balance_sequences_by_tokens(lengths: List[int], world_size: int) -> List[int]:
    """
    Greedy assignment of sequence indices to world_size bins to balance sum of lengths per rank.
    Returns an assignment list 'rank_of_seq' of size B.
    """
    import heapq
    # min-heap of (current_load, rank)
    heap = [(0, r) for r in range(world_size)]
    heapq.heapify(heap)
    # pair (length, idx) and sort descending
    order = sorted([(l, i) for i, l in enumerate(lengths)], reverse=True)
    rank_of_seq = [0] * len(lengths)
    for l, i in order:
        load, r = heapq.heappop(heap)
        rank_of_seq[i] = r
        heapq.heappush(heap, (load + l, r))
    return rank_of_seq


def build_per_rank_packing(
    seqs: List[torch.Tensor],
    world_size: int,
    device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[int]]]:
    """
    Partition sequences across ranks for balanced total tokens; produce per-rank packed tensors and cu_seqlens.
    Returns:
        packed_per_rank: list length world_size, each [T_r, H, D]
        cu_per_rank: list length world_size, each [N_r+1]
        seq_index_per_rank: mapping list per rank of original sequence indices included
    """
    lengths = [int(t.shape[0]) for t in seqs]
    assign = balance_sequences_by_tokens(lengths, world_size)
    per_rank_lists: List[List[torch.Tensor]] = [[] for _ in range(world_size)]
    per_rank_idx: List[List[int]] = [[] for _ in range(world_size)]
    for i, r in enumerate(assign):
        per_rank_lists[r].append(seqs[i])
        per_rank_idx[r].append(i)
    packed_per_rank: List[torch.Tensor] = []
    cu_per_rank: List[torch.Tensor] = []
    for r in range(world_size):
        if len(per_rank_lists[r]) == 0:
            # create empty packed
            if device is None and len(seqs) > 0:
                device = seqs[0].device
            dtype = seqs[0].dtype
            H = seqs[0].shape[-2]
            D = seqs[0].shape[-1]
            packed_per_rank.append(torch.empty((0, H, D), dtype=dtype, device=device))
            cu_per_rank.append(torch.zeros(1, dtype=torch.int32, device=device))
        else:
            packed, cu = pack_varlen(per_rank_lists[r], device=device)
            packed_per_rank.append(packed)
            cu_per_rank.append(cu)
    return packed_per_rank, cu_per_rank, per_rank_idx
