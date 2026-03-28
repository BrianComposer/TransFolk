# ------------------------------
# model/transformer_rope.py
# ------------------------------
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_music_model import BaseMusicModel


# -------------------------------------------------
# Rotary Embeddings
# -------------------------------------------------
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        return (
            self.cos[:, :, :seq_len, :].to(x.device),
            self.sin[:, :, :seq_len, :].to(x.device),
        )


# -------------------------------------------------
# Causal Self-Attention with RoPE
# -------------------------------------------------
class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 🔥 RoPE
        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        # attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


# -------------------------------------------------
# Feed Forward
# -------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Transformer Block
# -------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, max_seq_len):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionRoPE(
            d_model, n_heads, dropout, max_seq_len
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# -------------------------------------------------
# RoPE Transformer (Decoder-only)
# Decoder-only GPT
# Sin positional embeddings
# Con Rotary Positional Embeddings (RoPE)
# -------------------------------------------------
class MusicTransformerRoPE(BaseMusicModel):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512,
    ):
        super().__init__(vocab_size=vocab_size)

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # embeddings (sin positional embeddings)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout, max_seq_len
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_embedding.weight

        # 🔥 inicialización
        self.apply(self._init_weights)

    @property
    def arch_type(self):
        return "decoder_only_rope"

    def forward(self, x):
        B, T = x.size()

        if T > self.max_seq_len:
            raise ValueError("Sequence too long")

        x = self.token_embedding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)