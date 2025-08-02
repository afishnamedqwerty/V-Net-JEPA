# Unit tests (e.g., routing sparsity, energy deriv)
# # tests/test_hnet.py
import unittest
import torch
from models.hnet.encoder import LowLevelEncoder
from models.hnet.routing import SparseRouting
from models.hnet.downsampler import LearnedAttentionPooling
from models.hnet.pos_enc import AdaptivePosEnc
from models.hnet.dechunker import Dechunker  # Assume impl

class TestHNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.T, self.H, self.W = 32, 224, 224
        self.D = 256
        self.x = torch.randn(self.batch_size, self.T, self.H, self.W, 3)

    def test_encoder(self):
        encoder = LowLevelEncoder()
        h = encoder(self.x)
        self.assertEqual(h.shape, (self.batch_size, 32, 56, 56, self.D))  # Expected downsample

    def test_routing(self):
        h_flat = torch.randn(self.batch_size, 100000, self.D)
        positions = torch.randn(100000, 3)
        routing = SparseRouting()
        z, p = routing(h_flat, positions)
        self.assertEqual(z.shape[-1], self.D)

    def test_downsampler(self):
        z = torch.randn(self.batch_size, 1024, self.D)
        down = LearnedAttentionPooling()
        z_pooled = down(z)
        self.assertEqual(z_pooled.shape[1], 196)

    def test_pos_enc(self):
        z_pooled = torch.randn(self.batch_size, 196, self.D)
        centroids = torch.randn(self.batch_size, 196, 3)
        sizes = torch.randn(self.batch_size, 196)
        pos_enc = AdaptivePosEnc()
        z_pos = pos_enc(z_pooled, centroids, sizes)
        self.assertEqual(z_pos.shape, z_pooled.shape)

    def test_dechunker(self):
        z_proc = torch.randn(self.batch_size, 196, self.D)
        p = torch.softmax(torch.randn(self.batch_size, 100000, 196), dim=-1)
        positions = torch.randn(100000, 3)
        dechunk = Dechunker()
        recon = dechunk(z_proc, p, positions)
        self.assertEqual(recon.shape[1], 100000)  # Reconstructed

    @pytest.mark.skipif(not HAS_ATTENTION, reason="hnet_attention extension not built")
    def test_attention_forward_parity_fp32(self):
        # Build a small packed-varlen batch
        torch.manual_seed(0)
        H = 2
        D = 16
        lens = [3, 5, 4]  # B=3
        cu = [0]
        for L in lens:
            cu.append(cu[-1] + L)
        T = cu[-1]
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")

        q = torch.randn(T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Our op (single-rank path)
        out = hnet_attention(q, k, v, cu_seqlens, scale=1.0, training=False, dropout_p=0.0)

        # Reference: per-sequence scaled dot product attention
        ref = torch.empty_like(out)
        start = 0
        for L in lens:
            sl = slice(start, start + L)
            q_s = q[sl]               # [L,H,D]
            k_s = k[sl]               # [L,H,D]
            v_s = v[sl]               # [L,H,D]
            # Compute scores: [L,H,L]
            # Use explicit math to avoid SDPA availability differences
            scores = torch.empty(L, H, L, device=q.device, dtype=torch.float32)
            for t in range(L):
                for j in range(L):
                    # dot over D for each head
                    dot = (q_s[t] * k_s[j]).sum(-1)  # [H]
                    scores[t, :, j] = dot
            scores = scores / math.sqrt(D)
            probs = torch.softmax(scores, dim=-1)  # [L,H,L]
            out_s = torch.empty(L, H, D, device=q.device, dtype=torch.float32)
            for t in range(L):
                # [H,D] = sum_j probs[t,:,j][:,None] * v_s[j]
                out_s[t] = (probs[t].unsqueeze(-1) * v_s).sum(dim=0)
            ref[sl] = out_s
            start += L

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=3e-5)

    @pytest.mark.skipif(not HAS_ATTENTION, reason="hnet_attention extension not built")
    def test_attention_backward_parity_fp32_small(self):
        # Small shapes for gradient parity
        torch.manual_seed(1)
        H = 2
        D = 8
        lens = [2, 3]  # T=5
        cu = [0]
        for L in lens:
            cu.append(cu[-1] + L)
        T = cu[-1]
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device="cuda")

        q = torch.randn(T, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        # Our forward
        out = hnet_attention(q, k, v, cu_seqlens, scale=1.0, training=False, dropout_p=0.0)
        loss = out.sum()
        loss.backward()
        dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

        # Reference autograd with per-sequence masking
        q.grad = None; k.grad = None; v.grad = None
        start = 0
        out_ref = torch.empty_like(out)
        for L in lens:
            sl = slice(start, start + L)
            q_s = q[sl]  # [L,H,D]
            k_s = k[sl]
            v_s = v[sl]
            scores = torch.empty(L, H, L, device=q.device, dtype=torch.float32)
            for t in range(L):
                for j in range(L):
                    scores[t, :, j] = (q_s[t] * k_s[j]).sum(-1)
            scores = scores / math.sqrt(D)
            probs = torch.softmax(scores, dim=-1)
            out_s = torch.empty(L, H, D, device=q.device, dtype=torch.float32)
            for t in range(L):
                out_s[t] = (probs[t].unsqueeze(-1) * v_s).sum(dim=0)
            out_ref[sl] = out_s
            start += L
        loss_ref = out_ref.sum()
        loss_ref.backward()
        dq_ref, dk_ref, dv_ref = q.grad, k.grad, v.grad

        torch.testing.assert_close(dq, dq_ref, rtol=2e-3, atol=5e-4)
        torch.testing.assert_close(dk, dk_ref, rtol=2e-3, atol=5e-4)
        torch.testing.assert_close(dv, dv_ref, rtol=2e-3, atol=5e-4)

if __name__ == '__main__':
    unittest.main()
