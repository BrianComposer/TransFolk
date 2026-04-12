from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_classifier.corpus_membership_classifier import *


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    # === CONFIGURACIÓN ===

    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    registry = ConfigRegistry()
    registry.load_all()
    corpus = registry.find_by_name("todos")
    data_dir_raw = resolver.data_raw(corpus)
    data_dir_clean = resolver.data_clean(corpus)
    new_dir = ""       # carpeta de nuevas obras
    model_dir = resolver.paths.models_classifier # carpeta para guardar o cargar el modelo

    # Parámetros del modelo
    pca_components = 10
    nu = 0.05
    percentile_threshold = 85 #97.5
    mode = "train"

    # === CONTROL ===
    if mode == "train":
        train_model(data_dir_clean, model_dir,
                    pca_components=pca_components,
                    nu=nu,
                    percentile_threshold=percentile_threshold)
    elif mode == "evaluate":
        evaluate_model(new_dir, model_dir)
    else:
        print("Valor de 'mode' inválido. Use 'train' o 'evaluate'.")
