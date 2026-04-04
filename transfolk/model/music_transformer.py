from __future__ import annotations

import torch
import torch.nn as nn

from .base_music_model import BaseMusicModel


class MusicTransformer(BaseMusicModel):
    """
    Transformer autoregresivo decoder-only basado en TransformerEncoder
    con máscara causal + padding mask.
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

        self.output_linear = nn.Linear(d_model, vocab_size, bias=False)
        self.output_linear.weight = self.token_embedding.weight

        causal_mask = self._build_causal_mask(max_seq_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.apply(self._init_weights)

    @property
    def arch_type(self) -> str:
        return "decoder_only"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

    def _build_causal_mask(self, seq_len: int) -> torch.Tensor:
        # En PyTorch con attn mask booleana:
        # True  = bloquear
        # False = permitir
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        past_kv=None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        del past_kv

        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )

        if attention_mask is None:
            attention_mask = self.build_attention_mask_from_input(x)

        key_padding_mask = self.to_key_padding_mask(attention_mask)

        x = self.token_embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:seq_len, :seq_len].to(x.device)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        logits = self.output_linear(x)
        return self.maybe_return(logits=logits, past_kv=None, return_dict=return_dict)
