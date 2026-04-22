from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_tokenized(self) -> Path:
        return self.root / "data_tokenized"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def models_training(self) -> Path:
        return self.models / "training"

    @property
    def models_classifier(self) -> Path:
        return self.root / "models_classifier/sty_curves"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"