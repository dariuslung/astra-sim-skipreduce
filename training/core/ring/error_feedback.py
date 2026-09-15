"""
Error Feedback (EF) compensation buffer for virtual multi-rank simulation.
Tracks discarded compression residuals per virtual rank across training iterations.
"""

from typing import Dict, List, Optional
import torch


class ErrorFeedbackBuffer:
    def __init__(self, num_ranks: int):
        self.num_ranks = num_ranks
        # Map: param_id -> List[torch.Tensor] of length num_ranks
        self.buffers: Dict[int, List[Optional[torch.Tensor]]] = {}

    def get_residual(self, param_id: int, rank: int, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Retrieves or initializes the error residual buffer for a specific parameter and rank."""
        if param_id not in self.buffers:
            self.buffers[param_id] = [None] * self.num_ranks
            
        if self.buffers[param_id][rank] is None:
            self.buffers[param_id][rank] = torch.zeros(shape, device=device, dtype=dtype)
            
        return self.buffers[param_id][rank]

    def update_residual(self, param_id: int, rank: int, residual: torch.Tensor):
        """Updates the error residual for a specific parameter and rank."""
        self.buffers[param_id][rank] = residual.detach().clone()

    def reset(self):
        """Clears all stored residuals."""
        self.buffers.clear()
