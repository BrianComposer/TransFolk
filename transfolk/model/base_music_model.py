# ------------------------------
# model/base_model.py
# ------------------------------
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseMusicModel(nn.Module, ABC):
    """
    Clase base abstracta para todos los modelos generativos de TransFolk.

    Objetivo:
    - Unificar la interfaz de futuros modelos.
    - Permitir intercambiar arquitecturas con cambios mínimos en el pipeline.
    - Mantener compatibilidad con nn.Module y con PyTorch.

    Contrato mínimo:
    - Todo modelo debe implementar forward(...)
    - Todo modelo debe exponer el tamaño de vocabulario
    - Todo modelo debe declarar su tipo de arquitectura
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    @property
    @abstractmethod
    def arch_type(self) -> str:
        """
        Identificador del tipo de arquitectura.
        Ejemplos futuros:
        - 'decoder_only'
        - 'encoder_decoder'
        - 'hierarchical'
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Debe devolver los logits del modelo.
        En el caso autoregresivo actual, típicamente:
            [batch, seq_len, vocab_size]
        """
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        return self.vocab_size