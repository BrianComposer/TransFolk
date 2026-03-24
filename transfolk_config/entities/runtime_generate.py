from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class RuntimeGenerate(Serializable):
    id: int
    name: str

    temperature: Optional[float] = None
    max_len: Optional[int] = None
    num_productions: Optional[int] = None

    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    greedy: Optional[int] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    mixed_precision: Optional[int] = None
    num_workers: Optional[int] = None
    deterministic: Optional[int] = None