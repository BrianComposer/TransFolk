from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_music_model import BaseMusicModel


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ff(self.ln2(x))
        if attention_mask is not None:
            x = x * attention_mask[:, :, None].to(x.dtype)
        return x


class MusicTransformerGPT(BaseMusicModel):
    """
    Transformer decoder-only estilo GPT con contrato de forward unificado
    y soporte transversal de padding mask.
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

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def arch_type(self) -> str:
        return "decoder_only_gpt"

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        del past_kv

        B, T = x.size()

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({T}) exceeds max_seq_len ({self.max_seq_len})"
            )

        if attention_mask is None:
            attention_mask = self.build_attention_mask_from_input(x)

        token_emb = self.token_embedding(x)
        pos = torch.arange(0, T, device=x.device)
        pos_emb = self.position_embedding(pos)[None, :, :]

        x = token_emb + pos_emb
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
