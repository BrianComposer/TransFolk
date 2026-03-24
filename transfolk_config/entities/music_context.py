from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class MusicContext(Serializable):
    id: int
    name: str
    tonality: Optional[str] = None
    time_signature: Optional[str] = None