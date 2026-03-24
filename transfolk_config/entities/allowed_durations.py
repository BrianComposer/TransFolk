from typing import Optional, List
import json

from dataclasses import dataclass
from typing import Optional
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class AllowedDurations(Serializable):
    id: int
    name: str
    durations: List[float]
    description: Optional[str] = None

    @staticmethod
    def from_db(row):
        return AllowedDurations(
            id=row["id"],
            name=row["name"],
            durations=json.loads(row["durations"]),
            description=row["description"]
        )

    def __iter__(self):
        return iter(self.durations)

    def __len__(self):
        return len(self.durations)

    @property
    def values(self):
        return self.durations