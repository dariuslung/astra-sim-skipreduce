"""
1D Discrete Cosine Transform (DCT-II) and Inverse DCT (DCT-III) implemented via torch.fft.
Supports arbitrary batch shapes and orthonormal energy conservation.
"""

from typing import Tuple
import torch


def dct_1d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Computes the 1D Discrete Cosine Transform (DCT-II) along the last dimension.
    """
    N = x.shape[-1]
    # Mirrored extension: [x, flip(x)]
    x_seq = torch.cat([x, x.flip(dims=[-1])], dim=-1)
    X = torch.fft.fft(x_seq, dim=-1)[..., :N]
    
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    weights = torch.exp(-1j * torch.pi * k / (2.0 * N))
    res = (X * weights).real
    
    if norm == "ortho":
        res[..., 0] = res[..., 0] * (0.5 * (1.0 / N) ** 0.5)
        res[..., 1:] = res[..., 1:] * (0.5 * (2.0 / N) ** 0.5)
    return res


def idct_1d(X: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Computes the Inverse 1D Discrete Cosine Transform (DCT-III) along the last dimension.
    """
    N = X.shape[-1]
    X_mod = X.clone()
    if norm == "ortho":
        X_mod[..., 0] = X_mod[..., 0] * (2.0 * (N ** 0.5))
        X_mod[..., 1:] = X_mod[..., 1:] * ((2.0 * N) ** 0.5)
        
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    weights = torch.exp(1j * torch.pi * k / (2.0 * N))
    weights[..., 0] *= 0.5
    
    X_pad = torch.zeros(X.shape[:-1] + (2 * N,), dtype=torch.complex64, device=X.device)
    X_pad[..., :N] = X_mod * weights
    res = torch.fft.ifft(X_pad, dim=-1)[..., :N].real * 2
    return res


def compress_dct(
    x: torch.Tensor, 
    retention_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms tensor to DCT domain, retains lowest retention_ratio fraction of
    frequencies, and reconstructs back to the original domain.

    Args:
        x: Input tensor.
        retention_ratio: Float in (0.0, 1.0]. Fraction of low-frequency coefficients to retain.

    Returns:
        reconstructed: Approximated tensor in spatial/gradient domain with same shape as x.
        retained_coeffs: The 1D slice of retained low-frequency coefficients.
    """
    if retention_ratio <= 0.0:
        return torch.zeros_like(x), torch.empty(0, device=x.device, dtype=x.dtype)
    if retention_ratio >= 1.0:
        return x.clone(), dct_1d(x)

    N = x.shape[-1]
    k_keep = max(1, int(round(N * retention_ratio)))
    
    # Forward DCT
    X = dct_1d(x, norm="ortho")
    
    # Retained low-frequency coefficients
    retained_coeffs = X[..., :k_keep]
    
    # Filtered spectrum (zero-pad high frequencies)
    X_filtered = torch.zeros_like(X)
    X_filtered[..., :k_keep] = retained_coeffs
    
    # Inverse DCT
    reconstructed = idct_1d(X_filtered, norm="ortho")
    return reconstructed, retained_coeffs
