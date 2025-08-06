#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// Kernel launchers (implemented in attention_kernel.cu)
void zigzag_ring_dynamic_attention_forward_launcher(
    const at::Tensor& q,   // [T, H, D] packed
    const at::Tensor& k,   // [T, H, D] packed
    const at::Tensor& v,   // [T, H, D] packed
    const at::Tensor& cu_seqlens, // [B+1] int32
    at::Tensor& out,       // [T, H, D]
    const c10::optional<at::Tensor>& attn_mask,  // optional [T] or None (reserved)
    const double scale,
    const bool training,
    const c10::optional<at::Tensor>& dropout_mask, // optional, same T
    const double dropout_p
);

// Tile-consume (per K/V tile) forward accumulation (updates stats and y_accum)
void tile_consume_forward_launcher(
    const at::Tensor& q,             // [T,H,D]
    const at::Tensor& k_tile,        // [Tk,H,D]
    const at::Tensor& v_tile,        // [Tk,H,D]
    const at::Tensor& cu_seqlens,    // [B+1] int32
    at::Tensor& stats,               // [T,H,2] float32 (m_i, s_i)
    at::Tensor& y_accum,             // [T,H,D] float32
    const double scale
);

// Finalize forward: out = y_accum / s_i cast to q.dtype
void finalize_forward_launcher(
    const at::Tensor& y_accum,       // [T,H,D] float32
    const at::Tensor& stats,         // [T,H,2] float32
    at::Tensor& out                  // [T,H,D] dtype like q
);

void zigzag_ring_dynamic_attention_backward_launcher(
    const at::Tensor& grad_out,  // [T, H, D]
    const at::Tensor& q,         // [T, H, D]
    const at::Tensor& k,         // [T, H, D]
    const at::Tensor& v,         // [T, H, D]
    const at::Tensor& cu_seqlens,// [B+1]
    const c10::optional<at::Tensor>& attn_mask,
    const double scale,
    const c10::optional<at::Tensor>& dropout_mask,
    const double dropout_p,
    at::Tensor& dq,              // [T, H, D]
    at::Tensor& dk,              // [T, H, D]
    at::Tensor& dv               // [T, H, D]
);

// Backward tile-consume launcher (implemented in .cu)
void tile_consume_backward_launcher(
    const at::Tensor& grad_out,  // [T,H,D] float or castable
    const at::Tensor& q,         // [T,H,D]
    const at::Tensor& k_tile,    // [Tk,H,D]
    const at::Tensor& v_tile,    // [Tk,H,D]
    const at::Tensor& cu_seqlens,// [B+1]
    const at::Tensor& stats,     // [T,H,2] float32
    at::Tensor& dq,              // [T,H,D] float32 accum
    at::Tensor& dk_tile,         // [Tk,H,D] float32 accum
    at::Tensor& dv_tile,         // [Tk,H,D] float32 accum
    const double scale
);

void check_inputs(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_dtype(const at::Tensor& t) {
  auto dt = t.scalar_type();
  TORCH_CHECK(dt == at::kFloat || dt == at::kHalf || dt == at::kBFloat16,
              "dtype must be float32/float16/bfloat16");
}

} // namespace

std::vector<at::Tensor> hnet_attention_forward(
    const at::Tensor& q,          // [T,H,D]
    const at::Tensor& k,          // [T,H,D]
    const at::Tensor& v,          // [T,H,D]
    const at::Tensor& cu_seqlens, // [B+1]
    c10::optional<at::Tensor> attn_mask, // optional
    double scale,
    bool training,
    c10::optional<at::Tensor> dropout_mask,
    double dropout_p,
    c10::optional<at::Tensor> workspace_kv,      // optional preallocated workspace for staged K/V
    c10::optional<at::Tensor> workspace_stats,   // optional workspace for softmax stats
    c10::optional<int64_t> capacity_tokens,      // optional capacity guard
    c10::optional<bool> use_tiled                // optional: prefer tiled kernel if supported
) {
  at::cuda::CUDAGuard guard(q.device());
  check_inputs(q, "q"); check_inputs(k, "k"); check_inputs(v, "v");
  check_dtype(q); check_dtype(k); check_dtype(v);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q,k,v shape mismatch");
  TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1D");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt || cu_seqlens.scalar_type() == at::kLong,
              "cu_seqlens must be int32 or int64");

  // Normalize cu_seqlens to int32 for device kernels
  at::Tensor cu_i32 = cu_seqlens.scalar_type() == at::kInt ? cu_seqlens : cu_seqlens.to(at::kInt);
  auto out = at::empty_like(q);

  // Capacity guard if provided
  const int64_t T = q.size(0);
  if (capacity_tokens.has_value()) {
    TORCH_CHECK(T <= *capacity_tokens, "Total tokens exceed provided workspace capacity");
  }

  // Forward launcher (dense/tiled baseline uses no workspaces directly)
  zigzag_ring_dynamic_attention_forward_launcher(
      q, k, v, cu_i32, out, attn_mask, scale, training, dropout_mask, dropout_p
  );
  return {out};
}

std::vector<at::Tensor> hnet_attention_backward(
    const at::Tensor& grad_out,   // [T,H,D]
    const at::Tensor& q,          // [T,H,D]
    const at::Tensor& k,          // [T,H,D]
    const at::Tensor& v,          // [T,H,D]
    const at::Tensor& cu_seqlens, // [B+1]
    c10::optional<at::Tensor> attn_mask,
    double scale,
    c10::optional<at::Tensor> dropout_mask,
    double dropout_p,
    c10::optional<at::Tensor> workspace_grad   // optional workspace for grads/temp
) {
  at::cuda::CUDAGuard guard(grad_out.device());
  check_inputs(grad_out, "grad_out");
  check_inputs(q, "q"); check_inputs(k, "k"); check_inputs(v, "v");
  check_dtype(grad_out); check_dtype(q); check_dtype(k); check_dtype(v);

  // Normalize cu_seqlens to int32 for device kernels
  at::Tensor cu_i32 = cu_seqlens.scalar_type() == at::kInt ? cu_seqlens : cu_seqlens.to(at::kInt);

  auto dq = at::empty_like(q);
  auto dk = at::empty_like(k);
  auto dv = at::empty_like(v);
  zigzag_ring_dynamic_attention_backward_launcher(
      grad_out, q, k, v, cu_i32, attn_mask, scale, dropout_mask, dropout_p, dq, dk, dv
  );
  return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hnet_attention_forward, "H-Net ZigZag Ring Attention Forward",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_seqlens"),
        py::arg("attn_mask") = c10::nullopt, py::arg("scale") = 1.0, py::arg("training") = false,
        py::arg("dropout_mask") = c10::nullopt, py::arg("dropout_p") = 0.0,
        py::arg("workspace_kv") = c10::nullopt, py::arg("workspace_stats") = c10::nullopt,
        py::arg("capacity_tokens") = c10::nullopt, py::arg("use_tiled") = c10::nullopt);

  m.def("backward", &hnet_attention_backward, "H-Net ZigZag Ring Attention Backward",
        py::arg("grad_out"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_seqlens"),
        py::arg("attn_mask") = c10::nullopt, py::arg("scale") = 1.0,
        py::arg("dropout_mask") = c10::nullopt, py::arg("dropout_p") = 0.0,
        py::arg("workspace_grad") = c10::nullopt);

  // Expose tile-consume and finalize for Phase 3 ring orchestration
  m.def("tile_consume_forward", &tile_consume_forward_launcher,
        "Consume a staged K/V tile to update running stats and y_accum",
        py::arg("q"), py::arg("k_tile"), py::arg("v_tile"), py::arg("cu_seqlens"),
        py::arg("stats"), py::arg("y_accum"), py::arg("scale"));

  m.def("finalize_forward", &finalize_forward_launcher,
        "Finalize forward: out = y_accum / s_i",
        py::arg("y_accum"), py::arg("stats"), py::arg("out"));

  // Expose tile-wise backward consume
  m.def("tile_consume_backward", &tile_consume_backward_launcher,
        "Consume a K/V tile to accumulate dQ/dK/dV using saved stats",
        py::arg("grad_out"), py::arg("q"), py::arg("k_tile"), py::arg("v_tile"),
        py::arg("cu_seqlens"), py::arg("stats"),
        py::arg("dq"), py::arg("dk_tile"), py::arg("dv_tile"),
        py::arg("scale"));
}
