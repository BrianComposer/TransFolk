from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class RuntimeTrain(Serializable):
    id: int
    name: str

    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    gradient_clip: Optional[float] = None
    scheduler: Optional[str] = None
    warmup_steps: Optional[int] = None
    accumulation_steps: Optional[int] = None
    early_stopping: Optional[int] = None
    patience: Optional[int] = None
    save_every: Optional[int] = None