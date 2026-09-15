"""
Unit tests for GPT-Tiny model and structural layer metadata tagging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import torch
from training.models.gpt import get_gpt_tiny, get_gpt_layer_metadata


class TestGPT(unittest.TestCase):
    def test_gpt_forward_backward_tagging(self):
        vocab_size = 50257
        block_size = 128
        model = get_gpt_tiny(vocab_size=vocab_size, block_size=block_size)

        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 40000000)

        # Forward pass
        idx = torch.randint(0, vocab_size, (2, block_size))
        targets = torch.randint(0, vocab_size, (2, block_size))
        logits, loss = model(idx, targets)

        self.assertEqual(logits.shape, (2, block_size, vocab_size))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))


        # Backward pass
        loss.backward()

        # Metadata tagging check
        tagged_types = set()
        for name, p in model.named_parameters():
            meta = get_gpt_layer_metadata(name)
            tagged_types.add(meta["layer_type"])
            self.assertIsNotNone(p.grad, f"Gradient missing for {name}")
            self.assertNotEqual(meta["layer_type"], "other", f"Untagged layer: {name}")

        self.assertIn("embedding", tagged_types)
        self.assertIn("attn_qkv", tagged_types)
        self.assertIn("attn_proj", tagged_types)
        self.assertIn("ffn_up", tagged_types)
        self.assertIn("ffn_down", tagged_types)
        self.assertIn("lm_head", tagged_types)


if __name__ == "__main__":
    unittest.main()

