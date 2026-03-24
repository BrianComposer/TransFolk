from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class Corpus(Serializable):
    id: int
    name: str
    subcorpus: Optional[str] = None
    descripcion: Optional[str] = None