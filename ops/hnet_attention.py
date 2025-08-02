import torch
import torch.distributed as dist
from typing import Optional, Tuple

import hnet_attention_cuda  # built by hnet_attention/setup.py


class _Streams:
    def __init__(self):
        self.comp = torch.cuda.Stream()
        self.comm = torch.cuda.Stream()
        self._events = {
            "recv_ready": torch.cuda.Event(enable_timing=False, blocking=False),
            "comp_done": torch.cuda.Event(enable_timing=False, blocking=False),
            "send_done": torch.cuda.Event(enable_timing=False, blocking=False),
        }

    def record(self, name: str, stream: torch.cuda.Stream):
        ev = self._events.get(name)
        if ev is None:
            ev = torch.cuda.Event(enable_timing=False, blocking=False)
            self._events[name] = ev
        ev.record(stream)

    def wait(self, name: str, stream: torch.cuda.Stream):
        ev = self._events.get(name)
        if ev is not None:
            stream.wait_event(ev)


def _assert_tensors(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor):
    if not (q.is_cuda and k.is_cuda and v.is_cuda and cu_seqlens.is_cuda):
        raise ValueError("q,k,v,cu_seqlens must be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q,k,v must have the same dtype")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q,k,v shapes must match [T,H,D]")
    if cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be [B+1]")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be int32 or int64")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("q,k,v must be contiguous")


def _world_info() -> Tuple[int, int, bool]:
    if not dist.is_available() or not dist.is_initialized():
        return 1, 0, False
    return dist.get_world_size(), dist.get_rank(), True


def _ring_neighbors(rank: int, world: int, step: int) -> Tuple[int, int]:
    """Zig-zag neighbors for a given step: even steps go clockwise, odd steps counter-clockwise."""
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


def _post_irecv(tensor: torch.Tensor, src: int):
    if world_size := (dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1):
        if world_size > 1:
            return dist.irecv(tensor, src=src)
    return None


def _post_isend(tensor: torch.Tensor, dst: int):
    if world_size := (dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1):
        if world_size > 1:
            return dist.isend(tensor, dst=dst)
    return None


class HNetAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                cu_seqlens: torch.Tensor,
                scale: float = 1.0,
                training: bool = False,
                attn_mask: Optional[torch.Tensor] = None,
                dropout_mask: Optional[torch.Tensor] = None,
                dropout_p: float = 0.0) -> torch.Tensor:
        """
        q,k,v: [T,H,D] packed varlen (same dtype/device/contiguous)
        cu_seqlens: [B+1] cumulative lengths
        """
        _assert_tensors(q, k, v, cu_seqlens)
        world_size, rank, dist_on = _world_info()
        del rank  # not used yet for dense path

        # Placeholders for ring overlap: dual streams + events
        streams = _Streams()

        world_size, rank, dist_on = _world_info()

        # If distributed, execute a zig-zag N-1 ring overlap with staged tiles.
        # This Python path orchestrates comm/computation; compute still uses CUDA kernels.
        if dist_on and world_size > 1:
            # Allocate staging buffers (same dtype/device)
            T, H, D = q.shape
            # Workspace for streaming softmax accumulation (FP32 for stability)
            stats = torch.empty((T, H, 2), device=q.device, dtype=torch.float32)
            stats[..., 0].fill_(-float("inf"))  # m_i
            stats[..., 1].zero_()               # s_i
            y_accum = torch.zeros((T, H, D), device=q.device, dtype=torch.float32)

            # Tile size: simple heuristic; can be tuned. Use full tensor per step for now.
            Tk = T

            # Stage buffers for ring K/V exchange
            recv_k = torch.empty_like(k)
            recv_v = torch.empty_like(v)
            cur_k = k
            cur_v = v

            streams = _Streams()
            # Consume local tile first
            with torch.cuda.stream(streams.comp):
                hnet_attention_cuda.tile_consume_forward(
                    q, cur_k, cur_v, cu_seqlens, stats, y_accum, float(scale)
                )
                streams.record("comp_done", streams.comp)

            # N-1 ring steps: zig-zag direction
            for step in range(world_size - 1):
                prev_rank, next_rank = _ring_neighbors(rank, world_size, step)

                # 1) Comm stream: irecv next K/V
                with torch.cuda.stream(streams.comm):
                    req_rk = _post_irecv(recv_k, src=prev_rank)
                    req_rv = _post_irecv(recv_v, src=prev_rank)
                    streams.record("recv_ready", streams.comm)

                # 2) Compute stream: consume current tile while receiving the next
                with torch.cuda.stream(streams.comp):
                    # Wait for prior compute if needed; for sequential steps we are fine
                    # Since we already consumed cur_k/cur_v for step 0, in subsequent steps
                    # we consume currently held cur_k/cur_v which was received last iteration.
                    hnet_attention_cuda.tile_consume_forward(
                        q, cur_k, cur_v, cu_seqlens, stats, y_accum, float(scale)
                    )
                    streams.record("comp_done", streams.comp)

                # 3) Comm stream: send current K/V to next rank
                with torch.cuda.stream(streams.comm):
                    req_sk = _post_isend(cur_k, dst=next_rank)
                    req_sv = _post_isend(cur_v, dst=next_rank)
                    streams.record("send_done", streams.comm)

                # 4) Ensure recv finished before rotating buffers
                if req_rk is not None:
                    req_rk.wait()
                if req_rv is not None:
                    req_rv.wait()

                # Rotate: the newly received K/V become current for next iteration
                cur_k, recv_k = recv_k, cur_k
                cur_v, recv_v = recv_v, cur_v

            # Finalize: out = y_accum / s_i, cast to q dtype
            out = torch.empty_like(q)
            with torch.cuda.stream(streams.comp):
                hnet_attention_cuda.finalize_forward(y_accum, stats, out)
                streams.record("comp_done", streams.comp)

            # finalize default stream sync
            torch.cuda.current_stream().wait_stream(streams.comp)
            torch.cuda.current_stream().wait_stream(streams.comm)
        else:
            # Single-rank path: use tiled streaming-softmax forward, or tile-consume+finalize for parity
            streams = _Streams()
            # For single-rank, prefer kernel path for speed
            out = hnet_attention_cuda.forward(
                q, k, v, cu_seqlens,
                attn_mask, float(scale), bool(training),
                dropout_mask, float(dropout_p)
            )[0]

        # Save for backward
        ctx.save_for_backward(q, k, v, cu_seqlens)
        ctx.scale = float(scale)
        ctx.training = bool(training)
        ctx.has_dist = dist_on
        ctx._saved_streams = streams  # kept for symmetry; not used by current CUDA path
        ctx.dropout_mask = dropout_mask if training and dropout_p > 0.0 else None
        ctx.dropout_p = float(dropout_p)
        # Save stats if distributed path used tile-consume; else None
        ctx._saved_stats = stats if (dist_on and 'stats' in locals()) else None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, cu_seqlens = ctx.saved_tensors
        _assert_tensors(q, k, v, cu_seqlens)
        if not grad_out.is_cuda or not grad_out.is_contiguous():
            raise ValueError("grad_out must be CUDA and contiguous")

        # Prefer tile-wise backward if we have stats from distributed tile-consume forward
        dist_on = getattr(ctx, "has_dist", False)
        stats = getattr(ctx, "_saved_stats", None)
        if dist_on and (stats is not None) and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            T, H, D = q.shape

            # Allocate FP32 accumulators
            dq_acc = torch.zeros((T, H, D), device=q.device, dtype=torch.float32)
            dk_acc = torch.zeros_like(k, dtype=torch.float32)
            dv_acc = torch.zeros_like(v, dtype=torch.float32)

            # Stage buffers for ring K/V exchange (mirror forward)
            recv_k = torch.empty_like(k)
            recv_v = torch.empty_like(v)
            cur_k = k
            cur_v = v

            streams = _Streams()

            # Consume local tile first
            with torch.cuda.stream(streams.comp):
                hnet_attention_cuda.tile_consume_backward(
                    grad_out, q, cur_k, cur_v, cu_seqlens, stats,
                    dq_acc, dk_acc, dv_acc, float(ctx.scale)
                )
                streams.record("comp_done", streams.comp)

            # N-1 ring steps: zig-zag
            for step in range(world_size - 1):
                prev_rank, next_rank = _ring_neighbors(rank, world_size, step)

                # 1) Comm: irecv next K/V
                with torch.cuda.stream(streams.comm):
                    req_rk = _post_irecv(recv_k, src=prev_rank)
                    req_rv = _post_irecv(recv_v, src=prev_rank)
                    streams.record("recv_ready", streams.comm)

                # 2) Compute: consume current tile with grad_out
                with torch.cuda.stream(streams.comp):
                    hnet_attention_cuda.tile_consume_backward(
                        grad_out, q, cur_k, cur_v, cu_seqlens, stats,
                        dq_acc, dk_acc, dv_acc, float(ctx.scale)
                    )
                    streams.record("comp_done", streams.comp)

                # 3) Comm: send current K/V to next
                with torch.cuda.stream(streams.comm):
                    req_sk = _post_isend(cur_k, dst=next_rank)
                    req_sv = _post_isend(cur_v, dst=next_rank)
                    streams.record("send_done", streams.comm)

                # 4) Ensure recv finished before rotation
                if req_rk is not None:
                    req_rk.wait()
                if req_rv is not None:
                    req_rv.wait()

                # Rotate K/V for next iteration
                cur_k, recv_k = recv_k, cur_k
                cur_v, recv_v = recv_v, cur_v

            # Cast accumulators back to input dtype
            dq = dq_acc.to(q.dtype)
            dk = dk_acc.to(k.dtype)
            dv = dv_acc.to(v.dtype)

            torch.cuda.current_stream().wait_stream(streams.comp)
            torch.cuda.current_stream().wait_stream(streams.comm)
        else:
            # Fallback dense backward
            dq, dk, dv = hnet_attention_cuda.backward(
                grad_out, q, k, v, cu_seqlens,
                None, float(ctx.scale),
                ctx.dropout_mask, float(ctx.dropout_p)
            )

        # Non-tensor returns must align with inputs (q,k,v,cu_seqlens, scale, training, attn_mask, dropout_mask, dropout_p)
        return dq, dk, dv, None, None, None, None, None, None


def hnet_attention(q: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   cu_seqlens: torch.Tensor,
                   scale: float = 1.0,
                   training: bool = False,
                   attn_mask: Optional[torch.Tensor] = None,
                   dropout_p: float = 0.0) -> torch.Tensor:
    """
    Convenience wrapper. Dropout mask is generated outside if needed to guarantee reproducibility.
    """
    dropout_mask = None
    if training and dropout_p > 0.0:
        # Placeholder: dropout is to be fused in CUDA kernels; Python-side mask ensures deterministic tests if needed.
        T = q.shape[0]
        dropout_mask = torch.rand((T,), device=q.device, dtype=torch.float32)
    return HNetAttention.apply(q.contiguous(), k.contiguous(), v.contiguous(),
                               cu_seqlens.contiguous(), float(scale), bool(training),
                               attn_mask, dropout_mask, float(dropout_p))
