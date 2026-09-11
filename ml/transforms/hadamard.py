"""
Fast Walsh-Hadamard Transform (FWHT) and Inverse FWHT implemented with vectorized butterfly stages.
Supports top-k or prefix coefficient retention.
"""

from typing import Tuple
import torch


def fwht(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Computes Fast Walsh-Hadamard Transform (FWHT) along the last dimension.
    Pads input to the next power of 2 if necessary.
    """
    orig_shape = x.shape
    orig_len = orig_shape[-1]
    
    # Compute next power of 2
    pow2 = 1 << (orig_len - 1).bit_length()
    if pow2 != orig_len:
        x = torch.nn.functional.pad(x, (0, pow2 - orig_len))
        
    flat_x = x.reshape(-1, pow2).clone()
    N = pow2
    h = 1
    while h < N:
        flat_x = flat_x.view(-1, N // (2 * h), 2, h)
        a = flat_x[:, :, 0, :].clone()
        b = flat_x[:, :, 1, :].clone()
        flat_x = torch.stack([a + b, a - b], dim=2).view(-1, N)
        h *= 2
        
    if normalize:
        flat_x = flat_x / (pow2 ** 0.5)
        
    return flat_x.reshape(orig_shape[:-1] + (pow2,))


def ifwht(X: torch.Tensor, orig_len: int, normalize: bool = True) -> torch.Tensor:
    """
    Computes Inverse FWHT along the last dimension.
    Since normalized Hadamard matrix is symmetric and orthogonal, FWHT is its own inverse.
    """
    recon = fwht(X, normalize=normalize)
    return recon[..., :orig_len]


def compress_hadamard(
    x: torch.Tensor, 
    retention_ratio: float, 
    mode: str = "topk"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms tensor to Hadamard domain, retains a fraction of coefficients, 
    and reconstructs back.

    Args:
        x: Input tensor.
        retention_ratio: Fraction in (0.0, 1.0] of coefficients to retain.
        mode: 'topk' (largest magnitude coefficients) or 'prefix' (first k coefficients).

    Returns:
        reconstructed: Approximated tensor with shape identical to x.
        retained_coeffs: Tensor of retained coefficients.
    """
    if retention_ratio <= 0.0:
        return torch.zeros_like(x), torch.empty(0, device=x.device, dtype=x.dtype)
    if retention_ratio >= 1.0:
        return x.clone(), fwht(x)

    orig_len = x.shape[-1]
    X = fwht(x, normalize=True)
    total_len = X.shape[-1]
    k_keep = max(1, int(round(orig_len * retention_ratio)))

    X_filtered = torch.zeros_like(X)
    if mode == "topk":
        # Retain top-k largest magnitude coefficients
        _, idx = torch.topk(torch.abs(X), k_keep, dim=-1)
        retained_coeffs = torch.gather(X, -1, idx)
        X_filtered.scatter_(-1, idx, retained_coeffs)
    else:
        # Prefix retention
        retained_coeffs = X[..., :k_keep]
        X_filtered[..., :k_keep] = retained_coeffs

    reconstructed = ifwht(X_filtered, orig_len=orig_len, normalize=True)
    return reconstructed, retained_coeffs
