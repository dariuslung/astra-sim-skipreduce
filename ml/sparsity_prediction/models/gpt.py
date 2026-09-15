"""
Causal Language Transformer (GPT-Tiny) with structural layer metadata tagging.
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int = 384, n_head: int = 6, block_size: int = 128, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        # Key, query, value projections for all heads in a single linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

        # Causal mask to ensure attention is only directed to previous positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int = 384, mlp_ratio: float = 4.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, int(n_embd * mlp_ratio), bias=True)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(int(n_embd * mlp_ratio), n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int = 384, n_head: int = 6, block_size: int = 128, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 128,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([
                TransformerBlock(n_embd, n_head, block_size, mlp_ratio) for _ in range(n_layer)
            ]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


def get_gpt_tiny(vocab_size: int = 50257, block_size: int = 128) -> nn.Module:
    """Instantiates GPT-Tiny (~29.5M parameters with BPE 50,257)."""
    return GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        mlp_ratio=4.0,
    )


def get_gpt_layer_metadata(name: str) -> dict:
    """
    Maps GPT parameter names to structural attributes:
    - layer_type: 'embedding', 'attn_qkv', 'attn_proj', 'ffn_up', 'ffn_down',
                  'layernorm', 'lm_head'
    - stage: 'embed', 'block_0'..'block_5', 'head'
    - depth_index: integer 0 (embed) to 7 (head)
    - is_weight: True if weight tensor
    """
    is_weight = name.endswith(".weight")

    if name.startswith("transformer.wte.") or name.startswith("transformer.wpe."):
        return {
            "layer_type": "embedding",
            "stage": "embed",
            "depth_index": 0,
            "is_weight": True,
        }

    if name.startswith("lm_head."):
        return {
            "layer_type": "lm_head",
            "stage": "head",
            "depth_index": 7,
            "is_weight": is_weight,
        }

    if name.startswith("transformer.ln_f."):
        return {
            "layer_type": "layernorm",
            "stage": "head",
            "depth_index": 7,
            "is_weight": is_weight,
        }

    match = re.match(r"transformer\.h\.(\d+)\.(.*)", name)
    if match:
        block_idx = int(match.group(1))
        submodule = match.group(2)
        depth_index = 1 + block_idx
        stage_name = f"block_{block_idx}"

        if "attn.c_attn" in submodule:
            layer_type = "attn_qkv"
        elif "attn.c_proj" in submodule:
            layer_type = "attn_proj"
        elif "mlp.c_fc" in submodule:
            layer_type = "ffn_up"
        elif "mlp.c_proj" in submodule:
            layer_type = "ffn_down"
        elif "ln_1" in submodule or "ln_2" in submodule:
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
