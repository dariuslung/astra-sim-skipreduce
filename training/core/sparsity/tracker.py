"""
Standardized Gradient Sparsity Tracker for training experiments.

Provides an automated, zero-disruption profiler hook for tracking gradient
sparsity dynamics (energy concentration E10, Hoyer sparsity, relative threshold
sparsity, layer-type and stage breakdowns) across epochs.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np
import torch
import torch.nn as nn

from training.core.sparsity.metrics import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou,
)
from training.models import (
    get_resnet50_layer_metadata,
    get_vit_layer_metadata,
    get_gpt_layer_metadata,
)


def auto_detect_layer_metadata_fn(model: nn.Module) -> Callable[[str], Dict[str, Any]]:
    """
    Infers the appropriate layer metadata function based on parameter names.
    Supports ResNet-50, ViT, GPT, or falls back to a generic parameter parser.
    """
    param_names = [name for name, _ in model.named_parameters()]
    has_resnet = any("conv1" in n or "layer1" in n or "layer2" in n for n in param_names)
    has_vit = any("patch_embed" in n or "blocks." in n for n in param_names)
    has_gpt = any("transformer.h" in n or "wte" in n for n in param_names)

    if has_resnet:
        return get_resnet50_layer_metadata
    elif has_vit:
        return get_vit_layer_metadata
    elif has_gpt:
        return get_gpt_layer_metadata
    else:
        def generic_metadata(name: str) -> Dict[str, Any]:
            parts = name.split(".")
            stage = parts[0] if len(parts) > 1 else "root"
            is_weight = name.endswith(".weight")
            if "conv" in name.lower():
                ltype = "conv"
            elif "linear" in name.lower() or "fc" in name.lower() or "head" in name.lower():
                ltype = "linear"
            elif "norm" in name.lower() or "bn" in name.lower():
                ltype = "norm"
            elif "embed" in name.lower():
                ltype = "embedding"
            else:
                ltype = "other"
            return {
                "layer_type": ltype,
                "stage": stage,
                "depth_index": 0,
                "is_weight": is_weight,
            }
        return generic_metadata


class GradientSparsityTracker:
    """
    Tracks and records mathematical gradient sparsity metrics during model training.

    Usage:
        tracker = GradientSparsityTracker(model, sample_per_epoch=5)

        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                if tracker.should_sample(batch_idx):
                    tracker.record_step(batch_idx)

                optimizer.step()

            epoch_sparsity = tracker.finish_epoch(epoch)
            history['epochs_data'].append({..., 'gradient_sparsity': epoch_sparsity})
    """

    def __init__(
        self,
        model: nn.Module,
        metadata_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        sample_per_epoch: int = 5,
        total_batches: Optional[int] = None,
        sample_mode: str = "onset",
        skip_freq: int = 1,
        top_fraction: float = 0.10,
        rel_thresh_factor: float = 0.05,
    ):
        self.model = model
        self.metadata_fn = metadata_fn or auto_detect_layer_metadata_fn(model)
        self.sample_per_epoch = sample_per_epoch
        self.total_batches = total_batches
        self.sample_mode = sample_mode
        self.skip_freq = skip_freq
        self.top_fraction = top_fraction
        self.rel_thresh_factor = rel_thresh_factor

        self.sample_indices = self._compute_sample_indices()

        # Temporary accumulation buffer for current epoch
        self._epoch_samples_by_type = defaultdict(lambda: {"hoyer": [], "energy10": [], "rel_thresh": []})
        self._epoch_samples_by_stage = defaultdict(lambda: {"hoyer": [], "energy10": []})
        self._epoch_samples_global = {"hoyer": [], "energy10": [], "rel_thresh": []}

        # Multi-epoch history
        self.history: List[Dict[str, Any]] = []

    def _compute_sample_indices(self) -> Set[int]:
        """Calculates which batch indices to sample within each epoch."""
        if self.sample_mode == "onset":
            # First N batches of epoch (or even batches if skipping)
            if self.skip_freq > 1:
                return set([i for i in range(self.sample_per_epoch * 2) if i % self.skip_freq == 0][:self.sample_per_epoch])
            return set(range(self.sample_per_epoch))
        elif self.sample_mode == "uniform" and self.total_batches is not None:
            step = max(1, self.total_batches // self.sample_per_epoch)
            indices = [i * step for i in range(self.sample_per_epoch)]
            if self.skip_freq > 1:
                indices = [i if i % self.skip_freq == 0 else i + 1 for i in indices]
            return set(indices)
        else:
            # Default onset
            return set(range(self.sample_per_epoch))

    def should_sample(self, batch_idx: int) -> bool:
        """Determines if gradients should be sampled on this batch."""
        return batch_idx in self.sample_indices

    def record_step(self, batch_idx: int):
        """Computes mathematical sparsity metrics across all 2D+ parameter gradients."""
        batch_type_stats = defaultdict(lambda: {"hoyer": [], "energy10": [], "rel_thresh": []})
        batch_stage_stats = defaultdict(lambda: {"hoyer": [], "energy10": []})
        step_hoyer_all = []
        step_energy_all = []
        step_thresh_all = []

        for name, p in self.model.named_parameters():
            if p.grad is None or p.dim() < 2 or not p.requires_grad:
                continue

            meta = self.metadata_fn(name)
            lt = meta.get("layer_type", "other")
            stg = meta.get("stage", "unknown")

            h = compute_hoyer_sparsity(p.grad)
            e, _ = compute_energy_concentration(p.grad, top_fraction=self.top_fraction)
            r = compute_relative_threshold_sparsity(p.grad, factor=self.rel_thresh_factor)

            batch_type_stats[lt]["hoyer"].append(h)
            batch_type_stats[lt]["energy10"].append(e)
            batch_type_stats[lt]["rel_thresh"].append(r)

            batch_stage_stats[stg]["hoyer"].append(h)
            batch_stage_stats[stg]["energy10"].append(e)

            step_hoyer_all.append(h)
            step_energy_all.append(e)
            step_thresh_all.append(r)

        if not step_hoyer_all:
            return

        # Store step averages
        self._epoch_samples_global["hoyer"].append(float(np.mean(step_hoyer_all)))
        self._epoch_samples_global["energy10"].append(float(np.mean(step_energy_all)))
        self._epoch_samples_global["rel_thresh"].append(float(np.mean(step_thresh_all)))

        for lt, vals in batch_type_stats.items():
            self._epoch_samples_by_type[lt]["hoyer"].append(float(np.mean(vals["hoyer"])))
            self._epoch_samples_by_type[lt]["energy10"].append(float(np.mean(vals["energy10"])))
            self._epoch_samples_by_type[lt]["rel_thresh"].append(float(np.mean(vals["rel_thresh"])))

        for stg, vals in batch_stage_stats.items():
            self._epoch_samples_by_stage[stg]["hoyer"].append(float(np.mean(vals["hoyer"])))
            self._epoch_samples_by_stage[stg]["energy10"].append(float(np.mean(vals["energy10"])))

    def finish_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        Averages sampled metrics across the epoch, resets the buffer, and appends to history.
        Returns the structured epoch summary.
        """
        global_summary = {
            "hoyer": round(float(np.mean(self._epoch_samples_global["hoyer"])), 4) if self._epoch_samples_global["hoyer"] else 0.0,
            "energy10": round(float(np.mean(self._epoch_samples_global["energy10"])), 2) if self._epoch_samples_global["energy10"] else 0.0,
            "rel_thresh": round(float(np.mean(self._epoch_samples_global["rel_thresh"])), 2) if self._epoch_samples_global["rel_thresh"] else 0.0,
        }

        type_summary = {}
        for lt, vals in self._epoch_samples_by_type.items():
            type_summary[lt] = {
                "hoyer": round(float(np.mean(vals["hoyer"])), 4),
                "energy10": round(float(np.mean(vals["energy10"])), 2),
                "rel_thresh": round(float(np.mean(vals["rel_thresh"])), 2),
            }

        stage_summary = {}
        for stg, vals in self._epoch_samples_by_stage.items():
            stage_summary[stg] = {
                "hoyer": round(float(np.mean(vals["hoyer"])), 4),
                "energy10": round(float(np.mean(vals["energy10"])), 2),
            }

        epoch_record = {
            "epoch": epoch,
            "num_samples": len(self._epoch_samples_global["hoyer"]),
            "global": global_summary,
            "by_layer_type": type_summary,
            "by_stage": stage_summary,
        }

        self.history.append(epoch_record)

        # Reset per-epoch buffers
        self._epoch_samples_by_type.clear()
        self._epoch_samples_by_stage.clear()
        self._epoch_samples_global = {"hoyer": [], "energy10": [], "rel_thresh": []}

        return epoch_record

    def get_summary(self) -> Dict[str, Any]:
        """Generates overall summary metrics across all completed epochs."""
        if not self.history:
            return {}

        first_epoch = self.history[0]
        last_epoch = self.history[-1]

        return {
            "total_epochs_recorded": len(self.history),
            "onset_epoch1_global": first_epoch["global"],
            "converged_final_global": last_epoch["global"],
            "onset_epoch1_by_layer_type": first_epoch["by_layer_type"],
            "converged_final_by_layer_type": last_epoch["by_layer_type"],
        }
