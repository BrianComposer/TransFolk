from transfolk.pipeline import TransFolkPipeline
from transfolk.core.model_registry import MODEL_REGISTRY

PIPELINE_CACHE = {}


def get_pipeline(model_id: str) -> TransFolkPipeline:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_id}")

    if model_id not in PIPELINE_CACHE:
        cfg = MODEL_REGISTRY[model_id]
        model_cfg = cfg["model_config"]

        if model_cfg.architecture is None:
            raise RuntimeError(f"Model {model_id} has no architecture")

        if model_cfg.experiment is None:
            raise RuntimeError(f"Model {model_id} has no experiment")

        if model_cfg.experiment.allowed_durations is None:
            raise RuntimeError(f"Model {model_id} has no allowed_durations")

        if cfg["vocab_file"] is None:
            raise RuntimeError(f"Model {model_id} has no vocab")

        PIPELINE_CACHE[model_id] = TransFolkPipeline(
            model_file=cfg["model_file"],
            vocab_file=cfg["vocab_file"],
            model_config=model_cfg,
            device="cpu"
        )

    return PIPELINE_CACHE[model_id]