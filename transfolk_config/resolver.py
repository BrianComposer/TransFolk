from pathlib import Path
from datetime import datetime

from transfolk_config import RuntimeGenerate
from transfolk_config import Corpus
from transfolk_config import Experiment
from transfolk_config import TransformerArchitecture
from transfolk_config import Model
from transfolk_config.paths import ProjectPaths


class PathResolver:

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    # --------------------
    # Helpers
    # --------------------

    @staticmethod
    def _safe(value: str) -> str:
        if not value:
            return "x"
        return (
            value.replace("/", "_")
                 .replace(" ", "")
        )

    @staticmethod
    def temp_slug(temperature: float) -> str:
        return f"{temperature:.1f}"

    # --------------------
    # DATA
    # --------------------

    def data_raw(self, corpus: Corpus, subcorpus=None) -> Path:
        if subcorpus is None:
            return self.paths.data / corpus.name
        else:
            return self.paths.data / f"{corpus.name}" / f"{subcorpus}"


    def data_clean(self, corpus: Corpus, subcorpus=None) -> Path:
        if subcorpus is None:
            return self.paths.data / f"{corpus.name}_clean"
        else:
            return self.paths.data / f"{corpus.name}" / f"{subcorpus}_clean"

    def data_mid(self, corpus: Corpus, subcorpus=None) -> Path:
        if subcorpus is None:
            return self.paths.data / f"{corpus.name}_mid"
        else:
            return self.paths.data / f"{corpus.name}" / f"{subcorpus}_mid"


    def data_token(self, corpus: Corpus, subcorpus=None) -> Path:
        if subcorpus is None:
            return self.paths.data_tokenized / f"{corpus.name}"
        else:
            return self.paths.data_tokenized / f"{corpus.name}" / f"{subcorpus}"

    def tokenize_dir(self, exp: Experiment) -> Path:
        return (
            self.paths.data_tokenized
            / exp.corpus.name
            / exp.tokenizer.name
        )


    # --------------------
    # TRAINING
    # --------------------

    def train_dir(self, arch: TransformerArchitecture, exp: Experiment) -> Path:
        return (
            self.paths.models
            / "training"
            / arch.name
            / exp.corpus.name
            / exp.tokenizer.name
        )

    # --------------------
    # MODELS RELEASED
    # --------------------

    def models_released_dir(self) -> Path:
        return self.paths.models / "released"

    # --------------------
    # PRODUCTIONS
    # --------------------

    def production_dir_all(self) -> Path:
        return self.paths.root / "experiments" / "productions"

    def production_dir(self, model: Model, runtime: RuntimeGenerate) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        return (
            self.paths.root
            / "experiments"
            / "productions"
            / model.architecture.name
            / model.experiment.corpus.name
            / model.experiment.tokenizer.name
            / ton
            / ts
            / self.temp_slug(runtime.temperature)
        )



    # --------------------
    # FILES
    # --------------------

    def sequences_file(self, exp: Experiment) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)
        filename = f"sequences_{exp.corpus.name}_{exp.tokenizer.name}_{ton}_{ts}.json"
        return self.data_token(exp.corpus)/ exp.tokenizer.name / filename

    def vocab_file(self, exp: Experiment) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)
        filename = f"vocab_{exp.corpus.name}_{exp.tokenizer.name}_{ton}_{ts}.json"
        return self.data_token(exp.corpus)/ exp.tokenizer.name / filename

    def loss_log_file(self, model: Model) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = f"training_loss_{model.experiment.corpus.name}_{model.experiment.tokenizer.name}_{ton}_{ts}.json"
        return self.train_dir(model.architecture, model.experiment) / filename

    def token_errors_file(self, exp: Experiment) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)
        filename = f"errors_{exp.corpus.name}_{exp.tokenizer.name}_{ton}_{ts}.json"
        return self.tokenize_dir(exp) / filename

    def model_file(self, model: Model) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}"
        )

        return self.train_dir(model.architecture, model.experiment) / f"{filename}.pt"

    def model_file_epoch(self, model: Model, epoch:int) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}_"
            f"epoch{epoch}"
        )

        return self.train_dir(model.architecture, model.experiment) / f"{filename}.pt"

    def model_cfg_file(self, model: Model) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}"
        )

        return self.train_dir(model.architecture, model.experiment) / f"{filename}.json"

    def model_epoch_cfg_file(self, model: Model, epoch:int) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}_"
            f"epoch{epoch}"
        )

        return self.train_dir(model.architecture, model.experiment) / f"{filename}.json"


    def model_snapshot_file(self, exp: Experiment, model_id: str) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)

        filename = (
            f"{model_id}_"
            f"{exp.corpus.name}_"
            f"{exp.tokenizer.name}_"
            f"{ton}_"
            f"{ts}_snapshot.json"
        )

        return self.train_dir(exp) / filename

    def generated_new_file(self, model: Model, runtime: RuntimeGenerate) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        filename = f"{model.architecture.name}_{model.experiment.tokenizer.name}_{ts}_{ton}_{timestamp}.musicxml"

        return self.production_dir(model, runtime) / filename

    # --------------------
    # CLASSIFIER
    # --------------------

    def classifier_dir(self, corpus: Corpus) -> Path:
        return self.paths.models_classifier / corpus.name

    # --------------------
    # CHARTS
    # --------------------

    def charts_dir(self) -> Path:
        return self.paths.experiments / "charts"

    def classifier_experiments_dir(self) -> Path:
        return self.paths.experiments / "classifier"