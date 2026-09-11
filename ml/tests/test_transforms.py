import unittest
import torch
from ml.transforms.dct import dct_1d, idct_1d, compress_dct
from ml.transforms.hadamard import fwht, ifwht, compress_hadamard
from ml.transforms import apply_transform


class TestTransforms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dct_invertibility(self):
        x = torch.randn(8, 256, device=self.device)
        X = dct_1d(x, norm="ortho")
        x_recon = idct_1d(X, norm="ortho")
        self.assertTrue(torch.allclose(x, x_recon, atol=1e-5), f"Max diff: {torch.max(torch.abs(x - x_recon)).item()}")

    def test_dct_orthonormality(self):
        x = torch.randn(4, 512, device=self.device)
        X = dct_1d(x, norm="ortho")
        energy_x = torch.sum(x ** 2)
        energy_X = torch.sum(X ** 2)
        self.assertTrue(torch.allclose(energy_x, energy_X, rtol=1e-4))

    def test_dct_compress_bounds(self):
        x = torch.randn(128, device=self.device)
        
        # r = 0.0 -> all zeros
        rec_zero, coeffs_zero = compress_dct(x, 0.0)
        self.assertEqual(coeffs_zero.numel(), 0)
        self.assertTrue(torch.equal(rec_zero, torch.zeros_like(x)))
        
        # r = 1.0 -> full recovery
        rec_one, coeffs_one = compress_dct(x, 1.0)
        self.assertTrue(torch.allclose(rec_one, x, atol=1e-5))

        # 0 < r < 1 -> partial energy retained
        rec_part, coeffs_part = compress_dct(x, 0.25)
        self.assertEqual(coeffs_part.numel(), 32)
        # For white noise, expected cos_sim for keeping fraction r is sqrt(r) = sqrt(0.25) = 0.5
        cos_sim = torch.dot(x, rec_part) / (torch.norm(x) * torch.norm(rec_part))
        self.assertGreater(cos_sim.item(), 0.4)

        # For smooth/correlated signal (typical of deep learning tensors), energy is heavily low-frequency
        smooth_x = torch.cumsum(torch.randn(128, device=self.device), dim=-1)
        rec_smooth, _ = compress_dct(smooth_x, 0.25)
        smooth_cos_sim = torch.dot(smooth_x, rec_smooth) / (torch.norm(smooth_x) * torch.norm(rec_smooth))
        self.assertGreater(smooth_cos_sim.item(), 0.8)

    def test_fwht_invertibility(self):
        # Non-power-of-two length
        x = torch.randn(4, 120, device=self.device)
        X = fwht(x, normalize=True)
        x_recon = ifwht(X, orig_len=120, normalize=True)
        self.assertTrue(torch.allclose(x, x_recon, atol=1e-5), f"Max diff: {torch.max(torch.abs(x - x_recon)).item()}")

    def test_hadamard_compress_bounds(self):
        x = torch.randn(256, device=self.device)
        
        # r = 0.0
        rec_zero, coeffs_zero = compress_hadamard(x, 0.0)
        self.assertTrue(torch.equal(rec_zero, torch.zeros_like(x)))
        
        # r = 1.0
        rec_one, coeffs_one = compress_hadamard(x, 1.0)
        self.assertTrue(torch.allclose(rec_one, x, atol=1e-5))

        # r = 0.25 top-k
        rec_part, coeffs_part = compress_hadamard(x, 0.25, mode="topk")
        self.assertEqual(coeffs_part.numel(), 64)
        cos_sim = torch.dot(x, rec_part) / (torch.norm(x) * torch.norm(rec_part))
        self.assertGreater(cos_sim.item(), 0.7)

    def test_unified_apply_transform(self):
        x = torch.randn(128, device=self.device)
        rec_none, _ = apply_transform(x, "none", 0.0)
        self.assertTrue(torch.equal(rec_none, torch.zeros_like(x)))
        
        rec_dct, _ = apply_transform(x, "dct", 0.2)
        self.assertEqual(rec_dct.shape, x.shape)
        
        rec_wht, _ = apply_transform(x, "hadamard", 0.2)
        self.assertEqual(rec_wht.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
