from dataclasses import dataclass
from typing import Optional
from .corpus import Corpus
from .tokenizer_algorithm import TokenizerAlgorithm
from .music_context import MusicContext
from .allowed_durations import AllowedDurations
from transfolk_config.entities.serializable import Serializable, register

@register
@dataclass
class Experiment(Serializable):
    id: int
    name: Optional[str]
    corpus: Corpus
    tokenizer: TokenizerAlgorithm
    music_context: MusicContext
    allowed_durations: AllowedDurations
    descripcion: Optional[str] = None

    @classmethod
    def from_dict(cls, data):
        data = {k: v for k, v in data.items() if k != "__class__"}

        return cls(
            id=data["id"],
            name=data.get("name"),
            corpus=Serializable._deserialize(data["corpus"]),
            tokenizer=Serializable._deserialize(data["tokenizer"]),
            music_context=Serializable._deserialize(data["music_context"]),
            allowed_durations=Serializable._deserialize(data["allowed_durations"]),
            descripcion=data.get("descripcion"),
        )
