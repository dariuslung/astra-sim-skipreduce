"""
Unit tests for GPT-Tiny model and structural layer metadata tagging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
from ml.sparsity_prediction.models.gpt import get_gpt_tiny, get_gpt_layer_metadata


def test_gpt():
    vocab_size = 50257
    block_size = 128
    model = get_gpt_tiny(vocab_size=vocab_size, block_size=block_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-Tiny total parameters: {total_params:,}")

    # Forward pass
    idx = torch.randint(0, vocab_size, (2, block_size))
    targets = torch.randint(0, vocab_size, (2, block_size))
    logits, loss = model(idx, targets)

    assert logits.shape == (2, block_size, vocab_size), f"Expected (2, {block_size}, {vocab_size}), got {logits.shape}"
    assert loss is not None and not torch.isnan(loss)
    print(f"Initial loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Metadata tagging check
    tagged_types = set()
    for name, p in model.named_parameters():
        meta = get_gpt_layer_metadata(name)
        tagged_types.add(meta["layer_type"])
        assert p.grad is not None, f"Gradient missing for {name}"
        assert meta["layer_type"] != "other", f"Untagged layer: {name}"

    assert "embedding" in tagged_types
    assert "attn_qkv" in tagged_types
    assert "attn_proj" in tagged_types
    assert "ffn_up" in tagged_types
    assert "ffn_down" in tagged_types
    assert "lm_head" in tagged_types
    print(f"GPT-Tiny layer types tagged: {tagged_types}")


if __name__ == "__main__":
    test_gpt()
    print("GPT-Tiny tests passed successfully!")
