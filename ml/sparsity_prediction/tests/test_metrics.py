"""
Unit tests for sparsity metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import numpy as np
from ml.sparsity_prediction.metrics import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou
)


def test_hoyer_sparsity_bounds():
    # Dense uniform vector: [1, 1, 1, 1, ..., 1] -> Hoyer = 0.0
    d = 1000
    uniform = torch.ones(d)
    h_uniform = compute_hoyer_sparsity(uniform)
    assert abs(h_uniform - 0.0) < 1e-4, f"Expected ~0.0, got {h_uniform}"

    # Maximally sparse one-hot vector: [1, 0, 0, ..., 0] -> Hoyer = 1.0
    one_hot = torch.zeros(d)
    one_hot[0] = 5.0
    h_one_hot = compute_hoyer_sparsity(one_hot)
    assert abs(h_one_hot - 1.0) < 1e-4, f"Expected ~1.0, got {h_one_hot}"

    # All zeros edge case
    all_zeros = torch.zeros(d)
    assert compute_hoyer_sparsity(all_zeros) == 0.0


def test_energy_concentration():
    d = 100
    # One-hot vector -> 1 element is 100% of the energy
    one_hot = torch.zeros(d)
    one_hot[0] = 10.0
    energy, mask = compute_energy_concentration(one_hot, top_fraction=0.10)
    assert abs(energy - 100.0) < 1e-4
    assert mask[0].item() is True
    assert mask.sum().item() == 10  # top 10% of 100 is 10

    # Uniform vector -> top 10% elements carry exactly 10% of energy
    uniform = torch.ones(d)
    energy_unif, _ = compute_energy_concentration(uniform, top_fraction=0.10)
    assert abs(energy_unif - 10.0) < 1e-4


def test_relative_threshold_sparsity():
    # Gaussian distribution: coordinates near 0 should be captured
    torch.manual_seed(42)
    g = torch.randn(10000)
    thresh_sp = compute_relative_threshold_sparsity(g, factor=0.05)
    # For standard normal, P(|X| < 0.05) ~= 2 * 0.01994 = ~3.99%
    assert 3.0 <= thresh_sp <= 5.0


def test_mask_iou():
    m1 = torch.tensor([True, True, False, False])
    m2 = torch.tensor([True, True, False, False])
    m3 = torch.tensor([False, False, True, True])
    m4 = torch.tensor([True, False, False, False])

    # Identical
    assert abs(compute_mask_iou(m1, m2) - 1.0) < 1e-5
    # Disjoint
    assert abs(compute_mask_iou(m1, m3) - 0.0) < 1e-5
    # Partial: m1 has {0,1}, m4 has {0}. Intersection=1, Union=2 -> 0.5
    assert abs(compute_mask_iou(m1, m4) - 0.5) < 1e-5


if __name__ == "__main__":
    test_hoyer_sparsity_bounds()
    test_energy_concentration()
    test_relative_threshold_sparsity()
    test_mask_iou()
    print("All metric tests passed successfully!")
