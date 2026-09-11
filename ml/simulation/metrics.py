"""
Gradient fidelity evaluation metrics.
"""

import torch


def cosine_similarity(
    g_sim: torch.Tensor, 
    g_true: torch.Tensor, 
    eps: float = 1e-8
) -> float:
    """
    Computes cosine similarity between simulated gradient and ground-truth gradient:
    cos(g_sim, g_true) = (g_sim . g_true) / (||g_sim|| * ||g_true||)
    """
    flat_sim = g_sim.flatten()
    flat_true = g_true.flatten()
    
    dot_prod = torch.dot(flat_sim, flat_true)
    norm_sim = torch.norm(flat_sim)
    norm_true = torch.norm(flat_true)
    
    if norm_sim < eps or norm_true < eps:
        return 1.0 if norm_sim < eps and norm_true < eps else 0.0
        
    sim = dot_prod / (norm_sim * norm_true)
    return float(torch.clamp(sim, -1.0, 1.0).item())


def relative_l2_error(
    g_sim: torch.Tensor, 
    g_true: torch.Tensor, 
    eps: float = 1e-8
) -> float:
    """
    Computes relative L2 error:
    ||g_sim - g_true||_2 / ||g_true||_2
    """
    flat_sim = g_sim.flatten()
    flat_true = g_true.flatten()
    
    diff_norm = torch.norm(flat_sim - flat_true)
    true_norm = torch.norm(flat_true)
    
    if true_norm < eps:
        return 0.0 if diff_norm < eps else float(diff_norm.item())
        
    return float((diff_norm / true_norm).item())
