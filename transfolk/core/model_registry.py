from pathlib import Path
from transfolk_config.entities.model import Model

BASE_MODEL_DIR = Path("models/released")


def build_model_registry(base_dir: Path = BASE_MODEL_DIR):
    registry = {}

    if not base_dir.exists():
        print("Base dir does not exist:", base_dir)
        return registry

    json_files = list(base_dir.rglob("*.json"))

    for json_path in json_files:
        print(f"\n=== {json_path.name}")

        if "vocab" in json_path.name.lower():
            #print("skip vocab file")
            continue

        try:
            model: Model = Model.load_json(json_path)
        except Exception as e:
            print("ERROR load_json:", e)
            continue

        print("model:", getattr(model, "name", None))

        if model.experiment is None:
            print("FAIL: experiment None")
            continue

        if model.architecture is None:
            print("FAIL: architecture None")
            continue

        exp = model.experiment

        if exp.tokenizer is None:
            print("FAIL: tokenizer None")
            continue

        if exp.music_context is None:
            print("FAIL: music_context None")
            continue

        if exp.allowed_durations is None:
            print("FAIL: allowed_durations None")
            continue

        model_stem = json_path.stem
        model_file = json_path.parent / f"{model_stem}.pt"

        if not model_file.exists():
            print("FAIL: model file missing", model_file)
            continue

        if not model.vocab_file:
            print("FAIL: vocab_file empty")
            continue

        vocab_path = Path(model.vocab_file)
        if not vocab_path.is_absolute():
            vocab_path = json_path.parent / vocab_path

        if not vocab_path.exists():
            print("FAIL: vocab not found", vocab_path)
            continue

        registry[model.name] = {
            "model_file": model_file,
            "vocab_file": vocab_path,
            "model_config": model,
            "json_path": json_path,
        }

        print("OK -> registered")

    return registry

MODEL_REGISTRY = build_model_registry()