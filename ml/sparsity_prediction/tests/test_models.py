"""
Test script for ResNet-50 and ViT model instantiation and metadata tagging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from ml.sparsity_prediction.models.resnet50 import get_cifar_resnet50, get_resnet50_layer_metadata
from ml.sparsity_prediction.models.vit import get_cifar_vit_tiny, get_vit_layer_metadata


def test_resnet50():
    model = get_cifar_resnet50(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), f"Expected (2, 10), got {y.shape}"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-50 total parameters: {total_params:,}")

    # Check backward
    loss = y.sum()
    loss.backward()

    # Check metadata tagging
    tagged_types = set()
    for name, p in model.named_parameters():
        meta = get_resnet50_layer_metadata(name)
        tagged_types.add(meta["layer_type"])
        assert p.grad is not None, f"Gradient missing for {name}"
        assert meta["layer_type"] != "other", f"Untagged layer: {name}"

    assert "conv1x1_reduce" in tagged_types
    assert "conv3x3_spatial" in tagged_types
    assert "conv1x1_expand" in tagged_types
    assert "classifier_head" in tagged_types
    print(f"ResNet-50 layer types tagged: {tagged_types}")


def test_vit():
    model = get_cifar_vit_tiny(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), f"Expected (2, 10), got {y.shape}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViT-Tiny total parameters: {total_params:,}")

    # Check backward
    loss = y.sum()
    loss.backward()

    # Check metadata tagging
    tagged_types = set()
    for name, p in model.named_parameters():
        meta = get_vit_layer_metadata(name)
        tagged_types.add(meta["layer_type"])
        assert p.grad is not None, f"Gradient missing for {name}"
        assert meta["layer_type"] != "other", f"Untagged layer: {name}"

    assert "attn_qkv" in tagged_types
    assert "attn_proj" in tagged_types
    assert "ffn_up" in tagged_types
    assert "ffn_down" in tagged_types
    assert "classifier_head" in tagged_types
    print(f"ViT-Tiny layer types tagged: {tagged_types}")


if __name__ == "__main__":
    test_resnet50()
    test_vit()
    print("All model and tagging tests passed successfully!")
