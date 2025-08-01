# Unit tests (e.g., routing sparsity, energy deriv)
# tests/test_energy.py
import unittest
import torch
from losses.energy import EnergyLoss

class TestEnergy(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.M = 196
        self.D = 256
        self.energy = EnergyLoss(robust=True)  # Huber

    def test_computation(self):
        f_x = torch.randn(self.batch_size, self.M, self.D)
        g_y = f_x + 0.1 * torch.randn_like(f_x)  # Noisy target
        w = torch.ones(self.batch_size, self.M) / self.M
        loss = self.energy(f_x, g_y, w)
        self.assertGreater(loss.item(), 0)

    def test_normalization(self):
        f_x = torch.randn(self.batch_size, self.M, self.D)
        g_y = f_x.clone()
        w_uneven = torch.softmax(torch.randn(self.batch_size, self.M), dim=-1)
        loss = self.energy(f_x, g_y, w_uneven)
        self.assertAlmostEqual(loss.item(), 0, places=5)  # Zero mismatch

    def test_noisy_handling(self):
        f_x = torch.randn(self.batch_size, self.M, self.D)
        g_y = f_x + torch.cat([torch.zeros(self.batch_size, self.M//2, self.D), 10*torch.randn(self.batch_size, self.M//2, self.D)], dim=1)  # Outliers
        loss_huber = self.energy(f_x, g_y)
        loss_mse = self.energy(f_x, g_y, robust=False)
        self.assertLess(loss_huber.item(), loss_mse.item())  # Huber caps

if __name__ == '__main__':
    unittest.main()
