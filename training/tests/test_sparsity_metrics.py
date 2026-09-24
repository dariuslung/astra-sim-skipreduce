"""
Unit tests for sparsity metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import torch
import numpy as np
from training.core.sparsity.metrics import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou,
    compute_gini_index
)


class TestSparsityMetrics(unittest.TestCase):
    def test_hoyer_sparsity_bounds(self):
        # Dense uniform vector: [1, 1, 1, 1, ..., 1] -> Hoyer = 0.0
        d = 1000
        uniform = torch.ones(d)
        h_uniform = compute_hoyer_sparsity(uniform)
        self.assertAlmostEqual(h_uniform, 0.0, places=3)

        # Maximally sparse one-hot vector: [1, 0, 0, ..., 0] -> Hoyer = 1.0
        one_hot = torch.zeros(d)
        one_hot[0] = 5.0
        h_one_hot = compute_hoyer_sparsity(one_hot)
        self.assertAlmostEqual(h_one_hot, 1.0, places=3)

        # All zeros edge case
        all_zeros = torch.zeros(d)
        self.assertEqual(compute_hoyer_sparsity(all_zeros), 0.0)

    def test_energy_concentration(self):
        d = 100
        # One-hot vector -> 1 element is 100% of the energy
        one_hot = torch.zeros(d)
        one_hot[0] = 10.0
        energy, mask = compute_energy_concentration(one_hot, top_fraction=0.10)
        self.assertAlmostEqual(energy, 100.0, places=3)
        self.assertTrue(mask[0].item())
        self.assertEqual(mask.sum().item(), 10)

        # Uniform vector -> top 10% elements carry exactly 10% of energy
        uniform = torch.ones(d)
        energy_unif, _ = compute_energy_concentration(uniform, top_fraction=0.10)
        self.assertAlmostEqual(energy_unif, 10.0, places=3)

    def test_relative_threshold_sparsity(self):
        torch.manual_seed(42)
        g = torch.randn(10000)
        thresh_sp = compute_relative_threshold_sparsity(g, factor=0.05)
        self.assertGreaterEqual(thresh_sp, 3.0)
        self.assertLessEqual(thresh_sp, 5.0)

    def test_mask_iou(self):
        m1 = torch.tensor([True, True, False, False])
        m2 = torch.tensor([True, True, False, False])
        m3 = torch.tensor([False, False, True, True])
        m4 = torch.tensor([True, False, False, False])

        # Identical
        self.assertAlmostEqual(compute_mask_iou(m1, m2), 1.0, places=5)
        # Disjoint
        self.assertAlmostEqual(compute_mask_iou(m1, m3), 0.0, places=5)
        # Partial
        self.assertAlmostEqual(compute_mask_iou(m1, m4), 0.5, places=5)

    def test_gini_index(self):
        d = 1000
        # Dense uniform vector: Gini = 0.0
        uniform = torch.ones(d)
        self.assertAlmostEqual(compute_gini_index(uniform), 0.0, places=3)

        # One-hot vector: Gini close to 1.0 (1 - 1/d = 0.999)
        one_hot = torch.zeros(d)
        one_hot[0] = 100.0
        self.assertAlmostEqual(compute_gini_index(one_hot), 0.999, places=3)

        # All zeros edge case
        zeros = torch.zeros(d)
        self.assertEqual(compute_gini_index(zeros), 0.0)

    def test_gradient_sparsity_tracker(self):
        from training.core.sparsity import GradientSparsityTracker
        from training.models import get_cifar_resnet50

        model = get_cifar_resnet50(num_classes=10)
        tracker = GradientSparsityTracker(model, sample_per_epoch=2)

        self.assertTrue(tracker.should_sample(0))
        self.assertTrue(tracker.should_sample(1))
        self.assertFalse(tracker.should_sample(2))

        # Synthetic forward & backward
        x = torch.randn(4, 3, 32, 32)
        target = torch.randint(0, 10, (4,))
        criterion = torch.nn.CrossEntropyLoss()

        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        tracker.record_step(0)
        epoch_stats = tracker.finish_epoch(1)

        self.assertIn("global", epoch_stats)
        self.assertIn("by_layer_type", epoch_stats)
        self.assertIn("by_stage", epoch_stats)
        self.assertGreater(epoch_stats["global"]["energy10"], 0.0)
        self.assertLessEqual(epoch_stats["global"]["energy10"], 100.0)
        self.assertGreater(epoch_stats["global"]["hoyer"], 0.0)
        self.assertLessEqual(epoch_stats["global"]["hoyer"], 1.0)
        self.assertIn("conv3x3_spatial", epoch_stats["by_layer_type"])
        self.assertIn("stage1", epoch_stats["by_stage"])


if __name__ == "__main__":
    unittest.main()


