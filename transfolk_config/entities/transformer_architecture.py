from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class TransformerArchitecture(Serializable):
    id: int
    name: str

    type: Optional[str] = None
    description: Optional[str] = None
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: Optional[float] = None
    max_seq_len: Optional[int] = None

    attention_type: Optional[str] = None
    activation: Optional[str] = None
    positional_encoding: Optional[str] = None
    layer_norm_eps: Optional[float] = None
    bias: Optional[int] = None
    weight_tying: Optional[int] = None
    embedding_dropout: Optional[float] = None
    residual_dropout: Optional[float] = None
    attention_dropout: Optional[float] = None
    initializer: Optional[str] = None
    rotary_dim: Optional[int] = None
    encoder_layers: Optional[int] = None
    decoder_layers: Optional[int] = None