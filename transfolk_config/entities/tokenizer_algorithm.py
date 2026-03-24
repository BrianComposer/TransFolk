from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class TokenizerAlgorithm(Serializable):
    id: int
    name: str
    description: Optional[str] = None