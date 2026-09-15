"""
Model definitions and structural tagging metadata for sparsity profiling.
"""

from .resnet50 import get_cifar_resnet50, get_resnet50_layer_metadata
from .vit import get_cifar_vit_tiny, get_vit_layer_metadata
from .gpt import get_gpt_tiny, get_gpt_layer_metadata

__all__ = [
    "get_cifar_resnet50",
    "get_resnet50_layer_metadata",
    "get_cifar_vit_tiny",
    "get_vit_layer_metadata",
    "get_gpt_tiny",
    "get_gpt_layer_metadata",
]


