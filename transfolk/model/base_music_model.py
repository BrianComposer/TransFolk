from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseMusicModel(nn.Module, ABC):
    """
    Clase base abstracta para todos los modelos generativos de TransFolk.

    Contrato unificado de forward:
        forward(
            x,
            past_kv=None,
            attention_mask=None,
            return_dict=False,
        )

    Convención para attention_mask:
    - shape: [batch, seq_len]
    - dtype: bool
    - True  -> token válido
    - False -> PAD / token que debe quedar enmascarado

    Convención de salida:
    - Por defecto devuelve logits: [batch, seq_len, vocab_size]
    - Si return_dict=True, devuelve un dict con al menos:
        {
            "logits": ...,
            "past_kv": ...,
        }
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    @property
    @abstractmethod
    def arch_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        return self.vocab_size

    @staticmethod
    def build_attention_mask_from_input(
        x: torch.Tensor,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Devuelve una máscara booleana [B, T] con True en tokens válidos.
        """
        return x != pad_token_id

    @staticmethod
    def to_key_padding_mask(
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Convierte la máscara estándar del proyecto a key_padding_mask de PyTorch.

        Entrada estándar del proyecto:
            True  = válido
            False = PAD

        key_padding_mask de PyTorch:
            True  = ignorar / enmascarar
            False = mantener
        """
        if attention_mask is None:
            return None

        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        return ~attention_mask

    @staticmethod
    def maybe_return(
        logits: torch.Tensor,
        past_kv: Any | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        if return_dict:
            return {
                "logits": logits,
                "past_kv": past_kv,
            }
        return logits
