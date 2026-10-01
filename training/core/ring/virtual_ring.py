"""
Single-GPU Virtual Multi-Rank Ring Simulator for SkipReduce with Domain Transforms.
Accurately replicates the MSCCL ring schedule used in ASTRA-sim without requiring multi-GPU hardware.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch

from training.core.transforms import apply_transform
from training.core.ring.error_feedback import ErrorFeedbackBuffer
from training.core.ring.ring_metrics import cosine_similarity, relative_l2_error



def simulate_skipreduce_ring(
    grads: List[torch.Tensor],
    num_ranks: int,
    s: int,
    transform_type: str = "none",
    retention_ratio: float = 0.0,
    param_id: Optional[int] = None,
    ef_buffer: Optional[ErrorFeedbackBuffer] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Simulates the exact ring reduction schedule of SkipReduce + domain transforms
    across N virtual ranks for a single parameter tensor.

    Args:
        grads: List of N gradient tensors (one per virtual rank).
        num_ranks: Number of virtual ranks (N).
        s: Number of reduction steps skipped in the reduce-scatter phase.
        transform_type: 'none', 'dct', or 'hadamard'.
        retention_ratio: Float in [0.0, 1.0], fraction of coefficients to retain for skipped ranks.
        param_id: Optional ID of the parameter (needed if Error Feedback is enabled).
        ef_buffer: Optional ErrorFeedbackBuffer instance for residual compensation.

    Returns:
        reduced_grad: The final reduced gradient tensor (same shape as input).
        stats: Dictionary containing cosine similarity with true average, relative L2 error,
               and payload transmission ratio.
    """
    assert len(grads) == num_ranks, f"Expected {num_ranks} rank gradients, got {len(grads)}"
    assert 0 <= s < num_ranks, f"s ({s}) must be in [0, {num_ranks - 1}]"

    orig_shape = grads[0].shape
    device = grads[0].device
    dtype = grads[0].dtype

    # Ground-truth full AllReduce for evaluation metrics
    g_true = torch.stack(grads, dim=0).mean(dim=0)

    # Fast path: if s == 0, SkipReduce is identical to standard Ring AllReduce
    if s == 0:
        return g_true, {
            "cos_sim": 1.0,
            "rel_l2_error": 0.0,
            "payload_ratio": 1.0
        }

    flat_grads = [g.flatten() for g in grads]
    total_len = flat_grads[0].numel()

    # Split each rank's flattened gradient into N chunks
    chunk_size = total_len // num_ranks
    remainder = total_len % num_ranks

    rank_chunks = []
    for r in range(num_ranks):
        chunks = []
        for c in range(num_ranks):
            start_idx = c * chunk_size
            end_idx = (c + 1) * chunk_size
            chunks.append(flat_grads[r][start_idx:end_idx].clone())
        rank_chunks.append(chunks)

    reduced_chunks = []
    
    # Ring Schedule matching skipreduce.py:
    # For chunk i: starts at rank i.
    # Reduce-Scatter phase runs for (N - 1 - s) steps.
    # Total active ranks accumulated = N - s.
    # Ranks skipped = s.
    for c in range(num_ranks):
        # 1. Non-skipped ranks (fully reduced)
        active_ranks = [(c + step) % num_ranks for step in range(num_ranks - s)]
        chunk_sum = sum(rank_chunks[r][c] for r in active_ranks)

        # 2. Skipped ranks: apply domain transform if configured
        skipped_ranks = [(c + step) % num_ranks for step in range(num_ranks - s, num_ranks)]
        for r in skipped_ranks:
            slice_data = rank_chunks[r][c]

            # Apply Error Feedback if enabled
            if ef_buffer is not None and param_id is not None:
                res_key = param_id * num_ranks + c
                residual = ef_buffer.get_residual(res_key, r, slice_data.shape, device, dtype)
                slice_data = slice_data + residual

            if transform_type != "none" and retention_ratio > 0.0:
                approx_slice, _ = apply_transform(slice_data, transform_type, retention_ratio)
                
                # Update Error Feedback residual
                if ef_buffer is not None and param_id is not None:
                    new_residual = slice_data - approx_slice
                    ef_buffer.update_residual(res_key, r, new_residual)
                    
                chunk_sum = chunk_sum + approx_slice
            else:
                # Pure SkipReduce: skipped ranks contribute 0
                if ef_buffer is not None and param_id is not None:
                    ef_buffer.update_residual(res_key, r, slice_data)

        # Average across all N ranks
        reduced_chunks.append(chunk_sum / float(num_ranks))

    # Assemble chunks
    assembled = torch.cat(reduced_chunks, dim=0)

    # Handle any remainder elements at the end of the tensor (fully reduced)
    if remainder > 0:
        remainder_sum = sum(flat_grads[r][total_len - remainder:] for r in range(num_ranks))
        assembled = torch.cat([assembled, remainder_sum / float(num_ranks)], dim=0)

    reduced_grad = assembled.reshape(orig_shape)

    # Calculate theoretical payload ratio vs. full AllReduce
    # Full: 2 * (N - 1) / N
    # SkipReduce: (2 * (N - 1) - s * (1 - r)) / N
    # Ratio = (2 * (N - 1) - s * (1 - r)) / (2 * (N - 1))
    full_steps = 2 * (num_ranks - 1)
    saved_steps = s * (1.0 - (retention_ratio if transform_type != "none" else 0.0))
    payload_ratio = (full_steps - saved_steps) / float(full_steps)

    stats = {
        "cos_sim": cosine_similarity(reduced_grad, g_true),
        "rel_l2_error": relative_l2_error(reduced_grad, g_true),
        "payload_ratio": payload_ratio
    }

    return reduced_grad, stats


def simulate_skipreduce_ring_adaptive(
    grads: List[torch.Tensor],
    num_ranks: int,
    layer_config: Dict[str, Any],
    layer_type: str = "conv3x3_spatial",
    param_id: Optional[int] = None,
    ef_buffer: Optional[ErrorFeedbackBuffer] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Simulates ring reduction with layer-specific skipping and CS retention budgets.

    Args:
        grads: List of N gradient tensors (one per virtual rank).
        num_ranks: Number of virtual ranks (N).
        layer_config: Mapping of layer_type string to config dict:
                      {'s': int, 'transform': str, 'retention': float}
                      If layer_type not found, falls back to default in layer_config
                      or standard full Ring AllReduce (s=0, transform='none', retention=1.0).
        layer_type: Structural layer type string (e.g. 'conv3x3_spatial', 'conv1x1_expand').
        param_id: Optional parameter ID for residual tracking.
        ef_buffer: Optional ErrorFeedbackBuffer instance.

    Returns:
        reduced_grad: The final reduced gradient tensor.
        stats: Dictionary containing cosine similarity, relative L2 error, payload ratio, and layer_type.
    """
    default_cfg = layer_config.get("default", {"s": 0, "transform": "none", "retention": 1.0})
    cfg = layer_config.get(layer_type, default_cfg)
    s = cfg.get("s", 0)
    transform_type = cfg.get("transform", "none")
    retention_ratio = cfg.get("retention", 0.0)

    reduced_grad, stats = simulate_skipreduce_ring(
        grads=grads,
        num_ranks=num_ranks,
        s=s,
        transform_type=transform_type,
        retention_ratio=retention_ratio,
        param_id=param_id,
        ef_buffer=ef_buffer,
    )
    stats["layer_type"] = layer_type
    stats["s"] = s
    stats["retention_ratio"] = retention_ratio
    stats["transform"] = transform_type
    return reduced_grad, stats

