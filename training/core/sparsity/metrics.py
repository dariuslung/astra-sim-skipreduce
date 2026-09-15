"""
Mathematical metrics for gradient sparsity and predictability profiling.
"""

import torch
from typing import Tuple


def compute_hoyer_sparsity(g: torch.Tensor) -> float:
    """
    Computes the scale-invariant Hoyer sparsity index:
        Hoyer(g) = (sqrt(d) - ||g||_1 / ||g||_2) / (sqrt(d) - 1)
    Bounded in [0, 1]:
        0.0 = completely dense (all coordinates equal)
        1.0 = maximally sparse (all energy in 1 coordinate)
    """
    d = g.numel()
    if d <= 1:
        return 0.0
    
    g_flat = g.detach().flatten().float()
    l2_norm = torch.linalg.norm(g_flat, 2)
    if l2_norm == 0.0 or torch.isnan(l2_norm):
        return 0.0
    
    l1_norm = torch.linalg.norm(g_flat, 1)
    sqrt_d = d ** 0.5
    hoyer = (sqrt_d - (l1_norm / l2_norm)) / (sqrt_d - 1.0)
    return float(torch.clamp(hoyer, 0.0, 1.0).item())


def compute_energy_concentration(g: torch.Tensor, top_fraction: float = 0.10) -> Tuple[float, torch.Tensor]:
    """
    Computes the percentage of total L2^2 energy captured by the top `top_fraction` coordinates.
    Returns:
        energy_pct (float): in [0.0, 100.0]
        top_mask (torch.Tensor): boolean mask of the top coordinates
    """
    d = g.numel()
    if d == 0:
        return 0.0, torch.zeros_like(g, dtype=torch.bool)
    
    g_flat = g.detach().flatten().float()
    sq = g_flat.pow(2)
    total_energy = sq.sum()
    if total_energy == 0.0 or torch.isnan(total_energy):
        return 0.0, torch.zeros(d, dtype=torch.bool, device=g.device)
    
    k = max(1, int(round(top_fraction * d)))
    topk_vals, topk_indices = torch.topk(sq, k=k, largest=True, sorted=False)
    topk_energy = topk_vals.sum()
    energy_pct = float((topk_energy / total_energy * 100.0).clamp(0.0, 100.0).item())
    
    top_mask = torch.zeros(d, dtype=torch.bool, device=g.device)
    top_mask[topk_indices] = True
    return energy_pct, top_mask


def compute_relative_threshold_sparsity(g: torch.Tensor, factor: float = 0.05) -> float:
    """
    Computes the percentage of coordinates whose magnitude is less than
    `factor` * standard_deviation(g).
    Returns:
        sparsity_pct (float): in [0.0, 100.0]
    """
    d = g.numel()
    if d <= 1:
        return 0.0
    
    g_flat = g.detach().flatten().float()
    sigma = torch.std(g_flat)
    if sigma == 0.0 or torch.isnan(sigma):
        return 0.0
    
    thresh = factor * sigma
    small_count = (g_flat.abs() < thresh).sum()
    sparsity_pct = float((small_count.float() / d * 100.0).clamp(0.0, 100.0).item())
    return sparsity_pct


def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Computes the Jaccard similarity (Intersection over Union) of two boolean masks:
        IoU = |M1 & M2| / |M1 | M2|
    """
    m1 = mask1.detach().flatten().bool()
    m2 = mask2.detach().flatten().bool()
    intersection = (m1 & m2).sum().float()
    union = (m1 | m2).sum().float()
    if union == 0.0:
        return 1.0 if intersection == 0.0 else 0.0
    return float((intersection / union).item())
