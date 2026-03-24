from dataclasses import is_dataclass
import json
from pathlib import Path

CLASS_REGISTRY = {}

def register(cls):
    CLASS_REGISTRY[cls.__name__] = cls
    return cls


class Serializable:

    def to_dict(self):
        def serialize(obj):
            if is_dataclass(obj):
                data = {k: serialize(v) for k, v in obj.__dict__.items()}
                data["__class__"] = obj.__class__.__name__
                return data
            elif isinstance(obj, list):
                return [serialize(x) for x in obj]
            else:
                return obj

        return serialize(self)

    def save_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._deserialize(data)

    @classmethod
    def _deserialize(cls, data):
        if isinstance(data, dict) and "__class__" in data:
            real_cls = CLASS_REGISTRY.get(data["__class__"], cls)
            return real_cls.from_dict(data)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict):
        # fallback genérico (para clases simples)
        data = {k: v for k, v in data.items() if k != "__class__"}
        return cls(**data)