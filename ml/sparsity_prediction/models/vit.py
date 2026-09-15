"""
Vision Transformer (ViT-Tiny) adapted for CIFAR-10 with structural layer metadata tagging.
"""

import math
import re
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int = 256, hidden_features: int = 1024, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Classify via [CLS] token
        out = self.head(x[:, 0])
        return out


def get_cifar_vit_tiny(num_classes: int = 10) -> nn.Module:
    """Instantiates CIFAR-10 ViT-Tiny (~5.8M parameters)."""
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    )


def get_vit_layer_metadata(name: str) -> dict:
    """
    Maps ViT parameter names to structural attributes:
    - layer_type: 'attn_qkv', 'attn_proj', 'ffn_up', 'ffn_down', 'layernorm',
                  'classifier_head', 'patch_embed', 'embedding'
    - stage: 'patch_embed', 'block_0'..'block_5', 'head'
    - depth_index: integer 0 (patch embed) to 7 (head)
    - is_weight: True if weight tensor
    """
    is_weight = name.endswith(".weight")

    if "cls_token" in name or "pos_embed" in name:
        return {
            "layer_type": "embedding",
            "stage": "patch_embed",
            "depth_index": 0,
            "is_weight": True,
        }

    if name.startswith("patch_embed."):
        return {
            "layer_type": "patch_embed",
            "stage": "patch_embed",
            "depth_index": 0,
            "is_weight": is_weight,
        }

    if name.startswith("head."):
        return {
            "layer_type": "classifier_head",
            "stage": "head",
            "depth_index": 7,
            "is_weight": is_weight,
        }

    if name.startswith("norm."):
        return {
            "layer_type": "layernorm",
            "stage": "head",
            "depth_index": 7,
            "is_weight": is_weight,
        }

    match = re.match(r"blocks\.(\d+)\.(.*)", name)
    if match:
        block_idx = int(match.group(1))
        submodule = match.group(2)
        depth_index = 1 + block_idx
        stage_name = f"block_{block_idx}"

        if "attn.qkv" in submodule:
            layer_type = "attn_qkv"
        elif "attn.proj" in submodule:
            layer_type = "attn_proj"
        elif "mlp.fc1" in submodule:
            layer_type = "ffn_up"
        elif "mlp.fc2" in submodule:
            layer_type = "ffn_down"
        elif "norm" in submodule:
            layer_type = "layernorm"
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
