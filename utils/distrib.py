# Distributed utilities (DDP/DeepSpeed helpers, NCCL ring attention)
# utils/distrib.py
import math
import os
import torch
import torch.distributed as dist
from typing import Any, Optional, List, Callable

def init_distributed(backend: str = "nccl", seed: int = 42):
    if dist.is_available() and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        seed_all(seed, local_rank)

def seed_all(seed: int = 42, rank_offset: int = 0):
    import random, numpy as np
    random.seed(seed + rank_offset)
    np.random.seed(seed + rank_offset)
    torch.manual_seed(seed + rank_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank_offset)

def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def broadcast_object(obj: Any, src: int = 0):
    if not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

def build_sampler(dataset):
    if dist.is_initialized():
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
    return None

def maybe_all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    world = get_world_size()
    if world == 1:
        return t
    gather_list = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gather_list, t.contiguous())
    return torch.cat(gather_list, dim=0)

def ring_neighbors(rank: int, world: int):
    left = (rank - 1 + world) % world
    right = (rank + 1) % world
    return left, right

def get_ring_neighbors(rank: int, world: int, step: int):
    """
    Zig-zag neighbors for a given step:
      - even steps go clockwise (send to right, receive from left)
      - odd steps go counter-clockwise (send to left, receive from right)
    Returns (prev_rank, next_rank)
    """
    if world <= 1:
        return rank, rank
    clockwise = (step % 2) == 0
    if clockwise:
        prev_rank = (rank - 1 + world) % world
        next_rank = (rank + 1) % world
    else:
        prev_rank = (rank + 1) % world
        next_rank = (rank - 1 + world) % world
    return prev_rank, next_rank

def zigzag_direction(step: int) -> str:
    """Return 'clockwise' on even steps, 'counter' on odd steps."""
    return "clockwise" if (step % 2) == 0 else "counter"

def post_isend(tensor: torch.Tensor, dst: int, tag: int = 0):
    """
    Thin wrapper for nonblocking send. Returns a Work handle or None if not distributed.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return None
    return dist.isend(tensor.contiguous(), dst=dst, tag=tag)

def post_irecv(tensor: torch.Tensor, src: int, tag: int = 0):
    """
    Thin wrapper for nonblocking recv. Returns a Work handle or None if not distributed.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return None
    return dist.irecv(tensor, src=src, tag=tag)

def ring_exchange(tensor: torch.Tensor, peer: int, tag: int = 0) -> torch.Tensor:
    """
    Non-blocking send/recv to a single peer. Returns received tensor (same shape).
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    recv_buf = torch.empty_like(tensor)
    req_send = dist.isend(tensor.contiguous(), dst=peer, tag=tag)
    req_recv = dist.irecv(recv_buf, src=peer, tag=tag)
    req_send.wait()
    req_recv.wait()
    return recv_buf

def zigzag_ring_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          steps: Optional[int] = None,
                          alternate: bool = True) -> torch.Tensor:
    """
    Zig-zag ring attention across ranks with optional step cap and light overlap.
    q: [B, H, S, D_h], k/v: local [B, H, S, D_h].
    We circulate k,v around the ring and accumulate attention outputs, alternating directions.
    Environment overrides:
      - RING_STEPS: int, limit steps below world size for latency testing.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    import os
    rank = get_rank()
    world = get_world_size()
    left, right = ring_neighbors(rank, world)

    # Step control from env
    max_steps_env = os.environ.get("RING_STEPS", None)
    max_steps_env = int(max_steps_env) if max_steps_env is not None else None

    steps = steps if steps is not None else world
    if max_steps_env is not None:
        steps = min(steps, max_steps_env)

    k_cur = k
    v_cur = v
    out = torch.zeros_like(q)

    # Pre-post exchange handles
    for s in range(steps):
        # Compute local attention
        scores = torch.matmul(q, k_cur.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = out + torch.matmul(attn, v_cur)

        # Determine direction
        go_right = (s % 2 == 0) if alternate else True
        peer = right if go_right else left

        # Light overlap: kick off sends before compute in next iter (best-effort)
        # Use nonblocking isend/irecv here
        if s < steps - 1:
            recv_k = torch.empty_like(k_cur)
            recv_v = torch.empty_like(v_cur)
            req_send_k = dist.isend(k_cur.contiguous(), dst=peer, tag=100 + s)
            req_send_v = dist.isend(v_cur.contiguous(), dst=peer, tag=200 + s)
            req_recv_k = dist.irecv(recv_k, src=peer, tag=100 + s)
            req_recv_v = dist.irecv(recv_v, src=peer, tag=200 + s)
            # Wait to ensure buffers are safe to reuse
            req_send_k.wait(); req_send_v.wait()
            req_recv_k.wait(); req_recv_v.wait()
            k_cur, v_cur = recv_k, recv_v

    out = out / float(steps)
    return out

# Activation recomputation helpers
def checkpoint_sequential(functions: List[Callable], x: torch.Tensor, *args, use_checkpoint: bool = True, **kwargs):
    """
    Apply a list of functions sequentially with optional torch.utils.checkpoint on each.
    Each function f: (x, *args, **kwargs) -> x
    """
    for f in functions:
        if use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(f, x, *args, use_reentrant=False, **kwargs)
        else:
            x = f(x, *args, **kwargs)
    return x
