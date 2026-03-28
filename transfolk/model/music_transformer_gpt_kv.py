# ------------------------------
# model/transformer_gpt_kv.py
# ------------------------------
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_music_model import BaseMusicModel


# -------------------------------------------------
# Causal Self-Attention con KV Cache
# -------------------------------------------------
class CausalSelfAttentionKV(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, past_k=None, past_v=None):
        """
        x: [B, T, C]
        past_k, past_v: cache de pasos anteriores

        Si hay cache:
        - solo calculamos atención del último token
        - concatenamos con claves/valores pasados
        """

        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 🔥 KV CACHE
        if past_k is not None and past_v is not None:
            # concatenar con lo anterior
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # guardar nuevo cache
        new_k, new_v = k, v

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # máscara causal
        T_total = k.size(2)
        mask = torch.tril(torch.ones(T, T_total, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, new_k, new_v


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
# Transformer Block con cache
# -------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionKV(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, past_k=None, past_v=None):
        attn_out, k, v = self.attn(self.ln1(x), past_k, past_v)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, k, v


# -------------------------------------------------
# Modelo GPT con KV Cache
# -------------------------------------------------
class MusicTransformerGPT_KV(BaseMusicModel):
    """
    GPT decoder-only con KV cache para generación eficiente.

    Durante training:
        funciona como un transformer normal

    Durante generate:
        reutiliza claves/valores → O(T) en vez de O(T²)
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

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

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
        return "decoder_only_gpt_kv"

    def forward(self, x, past_kv=None):
        """
        x: [B, T]

        past_kv:
            lista de (k, v) por capa

        returns:
            logits, new_kv
        """

        B, T = x.size()

        pos = torch.arange(0, T, device=x.device)
        x = self.token_embedding(x) + self.pos_embedding(pos)[None, :, :]

        x = self.dropout(x)

        new_kv = []

        for i, block in enumerate(self.blocks):
            past_k, past_v = (None, None)
            if past_kv is not None:
                past_k, past_v = past_kv[i]

            x, k, v = block(x, past_k, past_v)
            new_kv.append((k, v))

        x = self.ln_f(x)
        logits = self.head(x)

        return logits, new_kv

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)