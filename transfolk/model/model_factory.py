# ------------------------------
# model/model_factory.py
# ------------------------------
from __future__ import annotations

from typing import Type

from transfolk_config.entities.transformer_architecture import TransformerArchitecture

from .music_transformer import MusicTransformer
from .music_transformer_gpt import MusicTransformerGPT
from .music_transformer_rope import MusicTransformerRoPE
from .music_transformer_gpt_kv import MusicTransformerGPT_KV
from .music_transformer_relative import MusicTransformerRelative

class ModelFactory:
    """
    Factory de modelos basada en TransformerArchitecture.

    Construye modelos directamente desde tu objeto de configuración
    (Serializable), evitando diccionarios intermedios.
    """

    _MODEL_REGISTRY: dict[str, Type] = {
        "decoder_only": MusicTransformer,
        "decoder_only_gpt": MusicTransformerGPT,
        "decoder_only_rope": MusicTransformerRoPE,
        "decoder_only_gpt_kv": MusicTransformerGPT_KV,
        "decoder_only_relative": MusicTransformerRelative
    }

    # --------------------------------------------------
    # MÉTODO PRINCIPAL
    # --------------------------------------------------
    @classmethod
    def build(
        cls,
        architecture: TransformerArchitecture,
        vocab_size: int,
    ):
        if architecture.type is None:
            raise ValueError("TransformerArchitecture.type must be defined")

        arch_type = architecture.type

        if arch_type not in cls._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown architecture type '{arch_type}'. "
                f"Available: {list(cls._MODEL_REGISTRY.keys())}"
            )

        model_class = cls._MODEL_REGISTRY[arch_type]

        kwargs = cls._build_kwargs(architecture, vocab_size, arch_type)

        return model_class(**kwargs)

    # --------------------------------------------------
    # CONSTRUCCIÓN DE KWARGS
    # --------------------------------------------------
    @staticmethod
    def _build_kwargs(
        arch: TransformerArchitecture,
        vocab_size: int,
        arch_type: str,
    ):
        """
        Traduce TransformerArchitecture → kwargs del modelo.
        Aquí está el desacoplamiento real.
        """

        # parámetros comunes
        kwargs = {
            "vocab_size": vocab_size,
            "d_model": arch.d_model,
            "dropout": arch.dropout,
            "max_seq_len": arch.max_seq_len,
        }

        # específicos por arquitectura
        if arch_type == "decoder_only":
            kwargs.update({
                "nhead": arch.n_heads,
                "num_layers": arch.n_layers,
                "dim_feedforward": arch.d_ff,
            })

        elif arch_type == "decoder_only_gpt":
            kwargs.update({
                "n_heads": arch.n_heads,
                "n_layers": arch.n_layers,
                "d_ff": arch.d_ff,
            })

        elif arch_type == "decoder_only_rope":
            kwargs.update({
                "n_heads": arch.n_heads,
                "n_layers": arch.n_layers,
                "d_ff": arch.d_ff,
            })

        elif arch_type == "decoder_only_gpt_kv":
            kwargs.update({
                "n_heads": arch.n_heads,
                "n_layers": arch.n_layers,
                "d_ff": arch.d_ff,
            })

        elif arch_type == "decoder_only_relative":
            kwargs.update({
                "n_heads": arch.n_heads,
                "n_layers": arch.n_layers,
                "d_ff": arch.d_ff,
            })

        else:
            raise ValueError(f"Unsupported architecture type: {arch_type}")

        # validación mínima (muy importante)
        for k, v in kwargs.items():
            if v is None:
                raise ValueError(f"Missing required architecture parameter: {k}")

        return kwargs

    # --------------------------------------------------
    # REGISTRO DINÁMICO
    # --------------------------------------------------
    @classmethod
    def register(cls, name: str, model_class: Type):
        if name in cls._MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")

        cls._MODEL_REGISTRY[name] = model_class

    # --------------------------------------------------
    # DEBUG
    # --------------------------------------------------
    @classmethod
    def available_models(cls):
        return list(cls._MODEL_REGISTRY.keys())