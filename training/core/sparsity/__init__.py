"""
Mathematical metrics for gradient sparsity and predictability profiling.
"""

from .metrics import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou,
    compute_gini_index,
)
from .tracker import GradientSparsityTracker, auto_detect_layer_metadata_fn

__all__ = [
    "compute_hoyer_sparsity",
    "compute_energy_concentration",
    "compute_relative_threshold_sparsity",
    "compute_mask_iou",
    "compute_gini_index",
    "GradientSparsityTracker",
    "auto_detect_layer_metadata_fn",
]


