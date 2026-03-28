# ------------------------------
# model/transformer_relative.py
# ------------------------------
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_music_model import BaseMusicModel


# -------------------------------------------------
# Relative Position Bias (tipo T5 simplificado)
# -------------------------------------------------
class RelativePositionBias(nn.Module):
    """
    Aprende un bias en función de la distancia relativa entre tokens.

    En lugar de usar embeddings posicionales absolutos,
    añadimos un sesgo a la atención según:
        distancia = i - j
    """

    def __init__(self, num_heads, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        # tabla de embeddings de distancias relativas
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, T, device):
        """
        Devuelve bias de forma:
            [1, n_heads, T, T]
        """

        # matriz de posiciones
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]  # [T, T]

        # clip a rango permitido
        rel = rel.clamp(-self.max_distance, self.max_distance)

        # shift a índices positivos
        rel = rel + self.max_distance

        # lookup embeddings
        bias = self.relative_bias(rel)  # [T, T, n_heads]

        # reordenar a [1, heads, T, T]
        bias = bias.permute(2, 0, 1).unsqueeze(0)

        return bias


# -------------------------------------------------
# Attention con Relative Bias
# -------------------------------------------------
class CausalSelfAttentionRelative(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_distance=128):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_bias = RelativePositionBias(n_heads, max_distance)

    def forward(self, x):
        """
        x: [B, T, C]
        """

        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # atención estándar
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 🔥 AÑADIMOS SESGO RELATIVO
        att = att + self.rel_bias(T, x.device)

        # máscara causal
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
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionRelative(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# -------------------------------------------------
# Modelo principal
# -------------------------------------------------
class MusicTransformerRelative(BaseMusicModel):
    """
    Transformer decoder-only con atención relativa.

    Diferencia clave:
        NO usa positional embeddings absolutos.
        Usa sesgos en función de distancias relativas.

    Ventaja:
        mejor modelado de ritmo y estructura repetitiva.
    """

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

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @property
    def arch_type(self):
        return "decoder_only_relative"

    def forward(self, x):
        """
        x: [B, T]
        """
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