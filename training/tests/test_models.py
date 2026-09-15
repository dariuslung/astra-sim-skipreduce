"""
Test script for ResNet-50 and ViT model instantiation and metadata tagging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import torch
from training.models.resnet50 import get_cifar_resnet50, get_resnet50_layer_metadata
from training.models.vit import get_cifar_vit_tiny, get_vit_layer_metadata


class TestModels(unittest.TestCase):


    def test_resnet50(self):
        model = get_cifar_resnet50(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

        # Check backward
        loss = y.sum()
        loss.backward()

        # Check metadata tagging
        tagged_types = set()
        for name, p in model.named_parameters():
            meta = get_resnet50_layer_metadata(name)
            tagged_types.add(meta["layer_type"])
            self.assertIsNotNone(p.grad, f"Gradient missing for {name}")
            self.assertNotEqual(meta["layer_type"], "other", f"Untagged layer: {name}")

        self.assertIn("conv1x1_reduce", tagged_types)
        self.assertIn("conv3x3_spatial", tagged_types)
        self.assertIn("conv1x1_expand", tagged_types)
        self.assertIn("classifier_head", tagged_types)

    def test_vit(self):
        model = get_cifar_vit_tiny(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

        # Check backward
        loss = y.sum()
        loss.backward()

        # Check metadata tagging
        tagged_types = set()
        for name, p in model.named_parameters():
            meta = get_vit_layer_metadata(name)
            tagged_types.add(meta["layer_type"])
            self.assertIsNotNone(p.grad, f"Gradient missing for {name}")
            self.assertNotEqual(meta["layer_type"], "other", f"Untagged layer: {name}")

        self.assertIn("attn_qkv", tagged_types)
        self.assertIn("attn_proj", tagged_types)
        self.assertIn("ffn_up", tagged_types)
        self.assertIn("ffn_down", tagged_types)
        self.assertIn("classifier_head", tagged_types)


if __name__ == "__main__":
    unittest.main()

