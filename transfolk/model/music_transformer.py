# ------------------------------
# model/music_transformer.py
# ------------------------------
from __future__ import annotations

import torch
import torch.nn as nn

from .base_music_model import BaseMusicModel


class MusicTransformer(BaseMusicModel):
    """
    Transformer autoregresivo decoder-only basado en TransformerEncoder
    con máscara causal.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__(vocab_size=vocab_size)

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        self.register_buffer("positional_encoding", positional_encoding, persistent=False)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_linear = nn.Linear(d_model, vocab_size)

    @property
    def arch_type(self) -> str:
        return "decoder_only"

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape [batch, seq_len] con ids de tokens.

        Returns:
            logits: Tensor de shape [batch, seq_len, vocab_size]
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)

        mask = self._build_causal_mask(seq_len, x.device)
        x = self.transformer(x, mask=mask)

        logits = self.output_linear(x)
        return logits


#
# # ------------------------------
# # model/music_transformer.py
# # ------------------------------
# import torch
# import torch.nn as nn
#
# class MusicTransformer(nn.Module):
#     def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_len=512):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, d_model)
#         self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
#         self.dropout = nn.Dropout(dropout)
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
#         self.output_linear = nn.Linear(d_model, vocab_size)
#
#     def _generate_positional_encoding(self, max_len, d_model):
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)
#
#     def forward(self, x):
#         seq_len = x.size(1)
#         x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
#         x = self.dropout(x)
#         mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
#         x = self.transformer(x, mask=mask)
#         return self.output_linear(x)
