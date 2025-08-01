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

if __name__ == '__main__':
    unittest.main()
