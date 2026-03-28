# ------------------------------
# model/transformer_decoder_only_gpt.py
# ------------------------------
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_music_model import BaseMusicModel


# -------------------------------------------------
# Multi-Head Causal Self-Attention
# -------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.split(C, dim=2)

        # reshape → [B, heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # [B, heads, T, head_dim]

        # reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


# -------------------------------------------------
# Feed Forward
# -------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------
# Transformer Block (GPT style)
# -------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# -------------------------------------------------
# GPT-like Decoder Only Model
# -------------------------------------------------
class MusicTransformerGPT(BaseMusicModel):
    """
    Transformer decoder-only estilo GPT:
    - Bloques propios
    - Causal self-attention explícito
    - Preparado para futuras extensiones (KV cache, RoPE)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__(vocab_size=vocab_size)

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # Output head
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying (opcional pero recomendable)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def arch_type(self) -> str:
        return "decoder_only_gpt"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, T = x.size()

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({T}) exceeds max_seq_len ({self.max_seq_len})"
            )

        # token + position embeddings
        token_emb = self.token_embedding(x)  # [B, T, C]
        pos = torch.arange(0, T, device=x.device)
        pos_emb = self.position_embedding(pos)[None, :, :]  # [1, T, C]

        x = token_emb + pos_emb
        x = self.dropout(x)

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)
        return logits