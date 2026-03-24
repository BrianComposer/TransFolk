# from transfolk_config import DEFAULT_DOMAIN, DEFAULT_TRANSFORMER_MODEL, Experiment,TransformerModel,RuntimeConfig,ProjectPaths,PathResolver, Settings
# from transfolk_classifier.corpus_membership_classifier import *
#
#
# # =========================================================
# # MAIN
# # =========================================================
#
# if __name__ == "__main__":
#     # === CONFIGURACIÓN ===
#
#     settings = Settings()
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#
#     exp = Experiment(corpus="todos")
#     runtime = RuntimeConfig(mode="train") # "train" / "evaluate"
#
#     data_dir_raw = resolver.data_raw(exp)
#     data_dir_clean = resolver.data_clean(exp)
#
#     corpus_dir = resolver.data_clean(exp)
#     new_dir = resolver.production_dir_all(exp, runtime)        # carpeta de nuevas obras
#     model_dir = resolver.paths.models_classifier          # carpeta para guardar o cargar el modelo
#
#     # Parámetros del modelo
#     pca_components = 10
#     nu = 0.05
#     percentile_threshold = 70 #97.5
#
#     # === CONTROL ===
#     if runtime.mode == "train":
#         train_model(corpus_dir, model_dir,
#                     pca_components=pca_components,
#                     nu=nu,
#                     percentile_threshold=percentile_threshold)
#     elif runtime.mode == "evaluate":
#         evaluate_model(new_dir, model_dir)
#     else:
#         print("Valor de 'mode' inválido. Use 'train' o 'evaluate'.")
