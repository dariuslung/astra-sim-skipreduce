"""
CIFAR-10 adapted ResNet-50 with structural layer metadata tagging.
"""

import re
import torch.nn as nn
import torchvision


def get_cifar_resnet50(num_classes: int = 10) -> nn.Module:
    """
    Creates a CIFAR-10 adapted ResNet-50:
    - 3x3 initial conv with stride 1, padding 1 (instead of 7x7 stride 2)
    - Identity maxpool (to preserve 32x32 feature map resolution)
    - 10-class linear projection head
    """
    model = torchvision.models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50_layer_metadata(name: str) -> dict:

    """
    Maps parameter name to structural attributes:
    - layer_type: 'conv1x1_reduce', 'conv3x3_spatial', 'conv1x1_expand', 'conv1x1_downsample',
                  'batchnorm', 'classifier_head'
    - stage: 'stem', 'stage1', 'stage2', 'stage3', 'stage4', 'head'
    - depth_index: integer from 0 (stem) to 17 (head)
    - is_weight: True if weight tensor, False if bias or running stat
    """
    is_weight = name.endswith(".weight")
    
    # Stem
    if name.startswith("conv1."):
        return {
            "layer_type": "conv3x3_spatial",
            "stage": "stem",
            "depth_index": 0,
            "is_weight": is_weight,
        }
    if name.startswith("bn1."):
        return {
            "layer_type": "batchnorm",
            "stage": "stem",
            "depth_index": 0,
            "is_weight": is_weight,
        }
    
    # Classifier head
    if name.startswith("fc."):
        return {
            "layer_type": "classifier_head",
            "stage": "head",
            "depth_index": 17,
            "is_weight": is_weight,
        }
    
    # Bottleneck layers: layer{1..4}.{block_idx}.{submodule}
    match = re.match(r"layer([1-4])\.(\d+)\.(.*)", name)
    if match:
        stage_num = int(match.group(1))
        block_idx = int(match.group(2))
        submodule = match.group(3)
        
        # Calculate sequential block depth index:
        # stage 1 has 3 blocks (0, 1, 2) -> indices 1..3
        # stage 2 has 4 blocks (0, 1, 2, 3) -> indices 4..7
        # stage 3 has 6 blocks (0..5) -> indices 8..13
        # stage 4 has 3 blocks (0..2) -> indices 14..16
        stage_offsets = {1: 1, 2: 4, 3: 8, 4: 14}
        depth_index = stage_offsets[stage_num] + block_idx
        stage_name = f"stage{stage_num}"
        
        if "conv1" in submodule:
            layer_type = "conv1x1_reduce"
        elif "conv2" in submodule:
            layer_type = "conv3x3_spatial"
        elif "conv3" in submodule:
            layer_type = "conv1x1_expand"
        elif "downsample.0" in submodule:
            layer_type = "conv1x1_downsample"
        elif "bn" in submodule or "downsample.1" in submodule:
            layer_type = "batchnorm"
        else:
            layer_type = "other"
            
        return {
            "layer_type": layer_type,
            "stage": stage_name,
            "depth_index": depth_index,
            "is_weight": is_weight,
        }
    
    return {
        "layer_type": "other",
        "stage": "unknown",
        "depth_index": 99,
        "is_weight": is_weight,
    }
