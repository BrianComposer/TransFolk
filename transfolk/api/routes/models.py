from fastapi import APIRouter
from transfolk.core.model_registry import MODEL_REGISTRY

router = APIRouter()


def serialize_dataclass(obj):
    if obj is None:
        return None

    data = {}
    for k, v in vars(obj).items():
        if k.startswith("_"):
            continue
        if v is not None:
            data[k] = v

    return data


@router.get("/models")
def list_models():
    models = []

    for model_id, cfg in MODEL_REGISTRY.items():
        model = cfg["model_config"]
        exp = model.experiment
        mc = exp.music_context
        tokenizer = exp.tokenizer
        corpus = exp.corpus

        models.append({
            "id": model_id,
            "name": model.name,
            "description": model.description,

            "algorithm": getattr(tokenizer, "name", None),
            "time_signature": getattr(mc, "time_signature", None),
            "mode": getattr(mc, "tonality", None),

            "architecture": serialize_dataclass(model.architecture),

            "experiment": {
                "id": getattr(exp, "id", None),
                "name": getattr(exp, "name", None),
                "description": getattr(exp, "descripcion", None),
                "corpus": getattr(corpus, "name", None),
                "tokenizer": getattr(tokenizer, "name", None),
                "music_context": serialize_dataclass(mc),
                "allowed_durations": serialize_dataclass(exp.allowed_durations),
            },

            "training": {
                "date": model.train_date,
                "start": model.train_start_time,
                "end": model.train_end_time,
                "total_time": model.train_total_time,
            },

            "artifacts": {
                "has_weights": cfg["model_file"] is not None,
                "has_vocab": cfg["vocab_file"] is not None,
                "config_available": cfg["json_path"] is not None,
            }
        })

    return {
        "status": "ok",
        "n_models": len(models),
        "models": models
    }