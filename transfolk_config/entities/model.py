from dataclasses import dataclass
from typing import Optional
from .experiment import Experiment
from .transformer_architecture import TransformerArchitecture
from .runtime_train import RuntimeTrain
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class Model(Serializable):
    id: int
    name: str
    architecture: TransformerArchitecture
    experiment: Experiment
    runtime_train: RuntimeTrain
    description: Optional[str] = None
    train_start_time: Optional[str] = None
    train_end_time: Optional[str] = None
    train_total_time: Optional[float] = None
    train_date: Optional[str] = None
    vocab_file: Optional[str] = None

    @classmethod
    def from_dict(cls, data):
        data = {k: v for k, v in data.items() if k != "__class__"}

        return cls(
            id=data["id"],
            name=data["name"],
            architecture=Serializable._deserialize(data["architecture"]),
            experiment=Serializable._deserialize(data["experiment"]),
            runtime_train=Serializable._deserialize(data["runtime_train"]),
            description=data.get("description"),
            train_start_time=data.get("train_start_time"),
            train_end_time=data.get("train_end_time"),
            train_total_time=data.get("train_total_time"),
            train_date=data.get("train_date"),
            vocab_file=data.get("vocab_file"),
        )