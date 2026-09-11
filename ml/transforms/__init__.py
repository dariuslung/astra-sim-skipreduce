import torch
from ml.transforms.dct import dct_1d, idct_1d, compress_dct
from ml.transforms.hadamard import fwht, ifwht, compress_hadamard


def apply_transform(
    tensor: torch.Tensor, 
    transform_type: str = "none", 
    retention_ratio: float = 0.0
):
    """
    Unified interface for applying domain transforms to gradient partitions.
    
    Args:
        tensor: 1D PyTorch tensor to compress.
        transform_type: 'none', 'dct', or 'hadamard'.
        retention_ratio: Float in [0.0, 1.0].
        
    Returns:
        reconstructed: Reconstructed approximate tensor in the spatial domain.
        retained: The retained transform coefficients.
    """
    if transform_type == "none" or retention_ratio <= 0.0:
        return torch.zeros_like(tensor), torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    elif transform_type == "dct":
        return compress_dct(tensor, retention_ratio)
    elif transform_type == "hadamard":
        return compress_hadamard(tensor, retention_ratio, mode="topk")
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
