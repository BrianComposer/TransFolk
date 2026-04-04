from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_music_model import BaseMusicModel


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos[:, :, :seq_len, :].to(x.device),
            self.sin[:, :, :seq_len, :].to(x.device),
        )


class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "For RoPE, head_dim must be even"

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            att = att.masked_fill(~key_mask, float("-inf"))

        att = F.softmax(att, dim=-1)

        if attention_mask is not None:
            query_mask = attention_mask[:, None, :, None].bool()
            att = att * query_mask

        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if attention_mask is not None:
            y = y * attention_mask[:, :, None].to(y.dtype)

        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionRoPE(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ff(self.ln2(x))
        if attention_mask is not None:
            x = x * attention_mask[:, :, None].to(x.dtype)
        return x


class MusicTransformerRoPE(BaseMusicModel):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__(vocab_size=vocab_size)

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @property
    def arch_type(self):
        return "decoder_only_rope"

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        del past_kv

        B, T = x.size()
        del B

        if T > self.max_seq_len:
            raise ValueError(f"Sequence too long: {T} > {self.max_seq_len}")

        if attention_mask is None:
            attention_mask = self.build_attention_mask_from_input(x)

        x = self.token_embedding(x)
        x = self.dropout(x)

        if attention_mask is not None:
            x = x * attention_mask[:, :, None].to(x.dtype)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)

        if attention_mask is not None:
            x = x * attention_mask[:, :, None].to(x.dtype)

        logits = self.head(x)
        return self.maybe_return(logits=logits, past_kv=None, return_dict=return_dict)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
