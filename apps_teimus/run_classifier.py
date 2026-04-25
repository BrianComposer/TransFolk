import sys
import pandas as pd

from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_classifier.classifier import (
    ALGORITHM_NAME,
    load_model_and_evaluate,
    train_and_save_model,
)
from transfolk_classifier.evaluate_models import evaluate_all_models
from transfolk_classifier.feature_importance import compute_feature_importance
from transfolk_classifier.build_corpus_split import build_split_metadata
from transfolk_classifier.classifier_curves import generate_plots_for_algorithms
from transfolk_classifier.ablation_study import run_ablation_study, aggregate_ablation_results
from transfolk_classifier.interpretability import (
    run_musical_interpretability,
    aggregate_musical_interpretability,
)
from transfolk_classifier.final_results import aggregate_final_results
from transfolk_features.extract_features import extract_features_musicxml


def _parse_algorithm_ids(argv):
    if len(argv) <= 2 or str(argv[2]).lower() == "all":
        return list(ALGORITHM_NAME.keys())

    raw = str(argv[2]).strip()
    ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    invalid = [x for x in ids if x not in ALGORITHM_NAME]
    if invalid:
        raise ValueError(f"algorithm_id inválido: {invalid}. Debe estar entre 1 y 8.")
    return ids


def _parse_modes(argv):
    default_modes = ["train", "eval", "compare", "features", "curves", "ablation", "interpretability", "final"]
    if len(argv) <= 3:
        return default_modes

    valid = set(default_modes)
    modes = [m.strip().lower() for m in str(argv[3]).split(",") if m.strip()]
    invalid = [m for m in modes if m not in valid]
    if invalid:
        raise ValueError(f"MODE inválido: {invalid}. Modos válidos: {sorted(valid)}")
    return modes


if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
    algorithm_ids = _parse_algorithm_ids(sys.argv)
    modes = _parse_modes(sys.argv)

    settings = Settings(ruta_base)
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name("ataa")
    corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
    corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)

    CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
    CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))

    seeds = [17, 1, 22, 309, 56, 44]

    for seed in seeds:
        output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
        output_path.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = str(output_path)

        df_split = build_split_metadata(
            str(CORPUS_P_DIR),
            str(CORPUS_R_DIR),
            test_size=0.20,
            random_state=seed,
        )

        output_file = output_path / "corpus_split.csv"
        df_split.to_csv(output_file, index=False, encoding="utf-8")

        # Antes estaba hardcodeado como: for MODE in ["ablation"]
        # Ahora se respeta el argumento CLI leído en _parse_modes().
        for MODE in ["curves"]:
            if MODE == "compare":
                results = evaluate_all_models(
                    results_dir=OUTPUT_DIR,
                    out_csv="model_comparison.csv",
                )
                print("\nResultados finales:")
                print(results)
                continue

            if MODE == "curves":
                selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
                generate_plots_for_algorithms(
                    results_dir=OUTPUT_DIR,
                    algorithm_names=selected_algorithm_names,
                    style="tableau-colorblind10",
                    show=False,
                )
                continue

            # El agregado final se ejecuta una sola vez al terminar todas las seeds.
            if MODE == "final":
                continue

            for algorithm_id in algorithm_ids:
                algorithm_name = ALGORITHM_NAME[algorithm_id]
                model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
                df_split = pd.read_csv(output_file)

                if MODE == "train":
                    train_and_save_model(
                        df_split=df_split,
                        algorithm_id=algorithm_id,
                        out_dir=str(model_dir),
                        random_state=seed,
                    )

                elif MODE == "eval":
                    load_model_and_evaluate(
                        model_dir=str(model_dir),
                        df_split=df_split,
                        out_csv_path=str(output_path / f"eval_predictions_{algorithm_name}.csv"),
                    )

                elif MODE == "features":
                    compute_feature_importance(
                        model_dir=str(model_dir),
                        df_split=df_split,
                        extract_features_musicxml=extract_features_musicxml,
                        method="permutation",
                        scoring="balanced_accuracy",
                        n_repeats=30,
                        random_state=seed,
                        out_csv_path=str(output_path / f"feature_importance_{algorithm_name}.csv"),
                    )

                elif MODE == "ablation":
                    run_ablation_study(
                        df_split=df_split,
                        algorithm_id=algorithm_id,
                        output_dir=OUTPUT_DIR,
                        extract_features_musicxml=extract_features_musicxml,
                        random_state=seed,
                        style="tableau-colorblind10",
                        force_recompute_features=False,
                        main_metric="balanced_accuracy",
                    )

                elif MODE == "interpretability":
                    run_musical_interpretability(
                        df_split=df_split,
                        model_dir=str(model_dir),
                        output_dir=OUTPUT_DIR,
                        extract_features_musicxml=extract_features_musicxml,
                        algorithm_name=algorithm_name,
                        random_state=seed,
                        style="tableau-colorblind10",
                        scoring="balanced_accuracy",
                        n_repeats=30,
                        top_n=20,
                        force_recompute_features=False,
                    )

    base_results_dir = str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}")
    selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]

    if "ablation" in modes:
        aggregate_ablation_results(
            base_results_dir=base_results_dir,
            seeds=seeds,
            algorithm_names=selected_algorithm_names,
            style="tableau-colorblind10",
            main_metric="balanced_accuracy",
        )

    if "interpretability" in modes:
        aggregate_musical_interpretability(
            base_results_dir=base_results_dir,
            seeds=seeds,
            algorithm_names=selected_algorithm_names,
            style="tableau-colorblind10",
            top_n=20,
        )

    if "final" in modes:
        aggregate_final_results(
            base_results_dir=base_results_dir,
            seeds=seeds,
            algorithm_names=selected_algorithm_names,
            style="tableau-colorblind10",
            main_metric="balanced_accuracy",
            top_n=20,
        )








# import sys
# import pandas as pd
#
# from transfolk_config import *
# from apps.db.config_registry import ConfigRegistry
# from transfolk_classifier.classifier import (
#     ALGORITHM_NAME,
#     load_model_and_evaluate,
#     train_and_save_model,
# )
# from transfolk_classifier.evaluate_models import evaluate_all_models
# from transfolk_classifier.feature_importance import compute_feature_importance
# from transfolk_classifier.build_corpus_split import build_split_metadata
# from transfolk_classifier.classifier_curves import generate_plots_for_algorithms
# from transfolk_classifier.ablation_study import run_ablation_study, aggregate_ablation_results
# from transfolk_classifier.interpretability import (
#     run_musical_interpretability,
#     aggregate_musical_interpretability,
# )
# from transfolk_features.extract_features import extract_features_musicxml
#
#
# def _parse_algorithm_ids(argv):
#     if len(argv) <= 2 or str(argv[2]).lower() == "all":
#         return list(ALGORITHM_NAME.keys())
#
#     raw = str(argv[2]).strip()
#     ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
#     invalid = [x for x in ids if x not in ALGORITHM_NAME]
#     if invalid:
#         raise ValueError(f"algorithm_id inválido: {invalid}. Debe estar entre 1 y 8.")
#     return ids
#
#
# def _parse_modes(argv):
#     default_modes = [ "train", "eval", "compare", "features", "curves", "ablation", "interpretability"]
#     if len(argv) <= 3:
#         return default_modes
#
#     valid = set(default_modes)
#     modes = [m.strip().lower() for m in str(argv[3]).split(",") if m.strip()]
#     invalid = [m for m in modes if m not in valid]
#     if invalid:
#         raise ValueError(f"MODE inválido: {invalid}. Modos válidos: {sorted(valid)}")
#     return modes
#
#
# if __name__ == "__main__":
#     ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
#     algorithm_ids = _parse_algorithm_ids(sys.argv)
#     modes = _parse_modes(sys.argv)
#
#     settings = Settings(ruta_base)
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#
#     CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
#     CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))
#
#     seeds = [17, 1, 22, 309, 56, 44]
#
#     for seed in seeds:
#         output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
#         output_path.mkdir(parents=True, exist_ok=True)
#         OUTPUT_DIR = str(output_path)
#
#         df_split = build_split_metadata(
#             str(CORPUS_P_DIR),
#             str(CORPUS_R_DIR),
#             test_size=0.20,
#             random_state=seed,
#         )
#
#         output_file = output_path / "corpus_split.csv"
#         df_split.to_csv(output_file, index=False, encoding="utf-8")
#
#         # Antes estaba hardcodeado como: for MODE in ["ablation"]
#         # Ahora se respeta el argumento CLI leído en _parse_modes().
#         for MODE in modes:
#             if MODE == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv",
#                 )
#                 print("\nResultados finales:")
#                 print(results)
#                 continue
#
#             if MODE == "curves":
#                 selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
#                 generate_plots_for_algorithms(
#                     results_dir=OUTPUT_DIR,
#                     algorithm_names=selected_algorithm_names,
#                     style="tableau-colorblind10",
#                     show=False,
#                 )
#                 continue
#
#             for algorithm_id in algorithm_ids:
#                 algorithm_name = ALGORITHM_NAME[algorithm_id]
#                 model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
#                 df_split = pd.read_csv(output_file)
#
#                 if MODE == "train":
#                     train_and_save_model(
#                         df_split=df_split,
#                         algorithm_id=algorithm_id,
#                         out_dir=str(model_dir),
#                         random_state=seed,
#                     )
#
#                 elif MODE == "eval":
#                     load_model_and_evaluate(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         out_csv_path=str(output_path / f"eval_predictions_{algorithm_name}.csv"),
#                     )
#
#                 elif MODE == "features":
#                     compute_feature_importance(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         extract_features_musicxml=extract_features_musicxml,
#                         method="permutation",
#                         scoring="balanced_accuracy",
#                         n_repeats=30,
#                         random_state=seed,
#                         out_csv_path=str(output_path / f"feature_importance_{algorithm_name}.csv"),
#                     )
#
#                 elif MODE == "ablation":
#                     run_ablation_study(
#                         df_split=df_split,
#                         algorithm_id=algorithm_id,
#                         output_dir=OUTPUT_DIR,
#                         extract_features_musicxml=extract_features_musicxml,
#                         random_state=seed,
#                         style="tableau-colorblind10",
#                         force_recompute_features=False,
#                         main_metric="balanced_accuracy",
#                     )
#
#                 elif MODE == "interpretability":
#                     run_musical_interpretability(
#                         df_split=df_split,
#                         model_dir=str(model_dir),
#                         output_dir=OUTPUT_DIR,
#                         extract_features_musicxml=extract_features_musicxml,
#                         algorithm_name=algorithm_name,
#                         random_state=seed,
#                         style="tableau-colorblind10",
#                         scoring="balanced_accuracy",
#                         n_repeats=30,
#                         top_n=20,
#                         force_recompute_features=False,
#                     )
#
#     base_results_dir = str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}")
#     selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
#
#     if "ablation" in modes:
#         aggregate_ablation_results(
#             base_results_dir=base_results_dir,
#             seeds=seeds,
#             algorithm_names=selected_algorithm_names,
#             style="tableau-colorblind10",
#             main_metric="balanced_accuracy",
#         )
#
#     if "interpretability" in modes:
#         aggregate_musical_interpretability(
#             base_results_dir=base_results_dir,
#             seeds=seeds,
#             algorithm_names=selected_algorithm_names,
#             style="tableau-colorblind10",
#             top_n=20,
#         )
#
#


# import sys
# import pandas as pd
# from transfolk_config import *
# from apps.db.config_registry import ConfigRegistry
# from transfolk_classifier.classifier import train_and_save_model, load_model_and_evaluate, ALGORITHM_NAME
# from transfolk_classifier.evaluate_models import evaluate_all_models
# from transfolk_features.extract_features import extract_features_musicxml
# from transfolk_classifier.feature_importance import compute_feature_importance
# from transfolk_classifier.build_corpus_split import build_split_metadata
# from transfolk_classifier.classifier_curves import generate_plots_for_algorithms
#
#
# def _parse_algorithm_ids(argv):
#     """
#     Optional CLI usage:
#         python -m apps_teimus.run_classifier <root> 3
#         python -m apps_teimus.run_classifier <root> 1,3,5
#         python -m apps_teimus.run_classifier <root> all
#
#     If omitted, all classifiers are processed.
#     """
#     if len(argv) <= 2 or str(argv[2]).lower() == "all":
#         return list(ALGORITHM_NAME.keys())
#
#     raw = str(argv[2]).strip()
#     ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
#     invalid = [x for x in ids if x not in ALGORITHM_NAME]
#     if invalid:
#         raise ValueError(f"algorithm_id inválido: {invalid}. Debe estar entre 1 y 8.")
#     return ids
#
#
# if __name__ == "__main__":
#     ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
#     algorithm_ids = _parse_algorithm_ids(sys.argv)
#
#     settings = Settings(ruta_base)
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#
#     CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
#     CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))
#
#     # Puedes cambiar esta lista de semillas sin tocar el módulo de gráficas.
#     seeds = [17, 1, 22, 309, 56, 44]
#
#     # Modos del pipeline. "curves" se ejecuta después de "eval" porque usa eval_predictions_<modelo>.csv.
#     modes = ["train", "eval", "compare", "features", "curves"]
#
#     for seed in seeds:
#         output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
#         output_path.mkdir(parents=True, exist_ok=True)
#         OUTPUT_DIR = str(output_path)
#
#         # Etiquetamos todas las obras y dividimos el corpus TRAIN/EVAL.
#         df_split = build_split_metadata(
#             str(CORPUS_P_DIR),
#             str(CORPUS_R_DIR),
#             test_size=0.20,
#             random_state=seed,
#         )
#
#         output_file = output_path / "corpus_split.csv"
#         df_split.to_csv(output_file, index=False, encoding="utf-8")
#
#         for MODE in modes:
#             if MODE.lower() == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv",
#                 )
#                 print("\nResultados finales:")
#                 print(results)
#                 continue
#
#             if MODE.lower() == "curves":
#                 selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
#                 generate_plots_for_algorithms(
#                     results_dir=OUTPUT_DIR,
#                     algorithm_names=selected_algorithm_names,
#                     style="tableau-colorblind10",  # cambia a "petroff10" si prefieres esa paleta
#                     show=False,
#                 )
#                 continue
#
#             for algorithm_id in algorithm_ids:
#                 algorithm_name = ALGORITHM_NAME[algorithm_id]
#
#                 model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
#                 df_split = pd.read_csv(output_path / "corpus_split.csv")
#
#                 if MODE.lower() == "train":
#                     train_and_save_model(
#                         df_split=df_split,
#                         algorithm_id=algorithm_id,
#                         out_dir=str(model_dir),
#                     )
#
#                 elif MODE.lower() == "eval":
#                     load_model_and_evaluate(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         out_csv_path=str(output_path / f"eval_predictions_{algorithm_name}.csv"),
#                     )
#
#                 elif MODE.lower() == "features":
#                     compute_feature_importance(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         extract_features_musicxml=extract_features_musicxml,
#                         method="permutation",
#                         scoring="balanced_accuracy",
#                         n_repeats=30,
#                         out_csv_path=str(output_path / f"feature_importance_{algorithm_name}.csv"),
#                     )
#
#


# import sys
# import pandas as pd
# from transfolk_config import *
# from apps.db.config_registry import ConfigRegistry
# from transfolk_classifier.classifier import train_and_save_model, load_model_and_evaluate, ALGORITHM_NAME
# from transfolk_classifier.evaluate_models import evaluate_all_models
# from transfolk_features.extract_features import extract_features_musicxml
# from transfolk_classifier.feature_importance  import compute_feature_importance
# from transfolk_classifier.build_corpus_split import build_split_metadata
#
# if __name__ == "__main__":
#     ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
#
#     settings = Settings(ruta_base)
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#     corpusE = Corpus(name=corpus.name, subcorpus="eval", id=None)
#
#     CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
#     CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))
#
#
#     for seed in [17, 1, 22, 309, 56, 44]:
#
#         # OUTPUT_DIR = str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}")
#         output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
#         output_path.mkdir(parents=True, exist_ok=True)
#
#         OUTPUT_DIR = str(output_path)
#         # Etiquetamos todas las obras y dividimos el CORPUS TRAIN/EVAL
#         df_split = build_split_metadata(
#             str(CORPUS_P_DIR),
#             str(CORPUS_R_DIR),
#             test_size=0.20,
#             random_state=seed
#         )
#
#         output_file = f"{OUTPUT_DIR}\corpus_split.csv"
#         df_split.to_csv(output_file, index=False, encoding="utf-8")
#
#
#         for MODE in ["train", "eval", "compare", "features"]:
#             # Algoritmo
#
#             if MODE.lower() == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv"
#                 )
#                 print("\nResultados finales:")
#                 print(results)
#             else:
#
#                 for algorithm_id in range(1, 9):
#                     algorithm_name = ALGORITHM_NAME[algorithm_id]
#                     # {
#                     #     1: "logistic_regression",
#                     #     2: "svm_linear_calibrated",
#                     #     3: "gradient_boosting",
#                     #     4: "random_forest",
#                     #     5: "hist_gradient_boosting",
#                     #     6: "knn",
#                     #     7: "naive_bayes",
#                     #     8: "decision_tree"
#                     # }
#
#                     # --- Rutas ---
#                     df_prof_dir = CORPUS_P_DIR
#                     df_reli_dir = CORPUS_R_DIR
#                     # Carpeta donde guardar/cargar el modelo
#                     model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
#                     # Cargamos el dataframe con corpus de training y eval
#                     df_split = pd.read_csv(f"{OUTPUT_DIR}\corpus_split.csv")
#
#                     if MODE.lower() == "train":
#                         train_and_save_model(
#                             df_split=df_split,
#                             algorithm_id=algorithm_id,
#                             out_dir=str(model_dir),
#                         )
#                     elif MODE.lower() == "eval":
#                         load_model_and_evaluate(
#                             model_dir=str(model_dir),
#                             df_split=df_split,
#                             out_csv_path = rf"{OUTPUT_DIR}\eval_predictions_{algorithm_name}.csv",
#                         )
#                     elif MODE.lower()=="features":
#                         df_imp = compute_feature_importance(
#                             model_dir=str(model_dir),
#                             df_split=df_split,
#                             extract_features_musicxml=extract_features_musicxml,
#                             method="permutation",
#                             scoring="balanced_accuracy",
#                             n_repeats=30,
#                             out_csv_path = rf"{OUTPUT_DIR}\feature_importance{algorithm_name}.csv"
#                         )



# ablation
# import sys
# import pandas as pd
#
# from transfolk_config import *
# from apps.db.config_registry import ConfigRegistry
# from transfolk_classifier.classifier import (
#     ALGORITHM_NAME,
#     load_model_and_evaluate,
#     train_and_save_model,
# )
# from transfolk_classifier.evaluate_models import evaluate_all_models
# from transfolk_classifier.feature_importance import compute_feature_importance
# from transfolk_classifier.build_corpus_split import build_split_metadata
# from transfolk_classifier.classifier_curves import generate_plots_for_algorithms
# from transfolk_classifier.ablation_study import run_ablation_study, aggregate_ablation_results
# from transfolk_features.extract_features import extract_features_musicxml
#
#
# def _parse_algorithm_ids(argv):
#     """
#     Optional CLI usage:
#         python -m apps_teimus.run_classifier <root> 3
#         python -m apps_teimus.run_classifier <root> 1,3,5
#         python -m apps_teimus.run_classifier <root> all
#
#     If omitted, all classifiers are processed.
#     """
#     if len(argv) <= 2 or str(argv[2]).lower() == "all":
#         return list(ALGORITHM_NAME.keys())
#
#     raw = str(argv[2]).strip()
#     ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
#     invalid = [x for x in ids if x not in ALGORITHM_NAME]
#     if invalid:
#         raise ValueError(f"algorithm_id inválido: {invalid}. Debe estar entre 1 y 8.")
#     return ids
#
#
# def _parse_modes(argv):
#     """
#     Optional CLI usage:
#         python -m apps_teimus.run_classifier <root> 3 train,eval,curves,ablation
#         python -m apps_teimus.run_classifier <root> all ablation
#
#     If omitted, the complete paper pipeline is executed.
#     """
#     default_modes = ["train", "eval", "compare", "features", "curves", "ablation"]
#     if len(argv) <= 3:
#         return default_modes
#
#     valid = set(default_modes)
#     modes = [m.strip().lower() for m in str(argv[3]).split(",") if m.strip()]
#     invalid = [m for m in modes if m not in valid]
#     if invalid:
#         raise ValueError(f"MODE inválido: {invalid}. Modos válidos: {sorted(valid)}")
#     return modes
#
#
# if __name__ == "__main__":
#     ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
#     algorithm_ids = _parse_algorithm_ids(sys.argv)
#     modes = _parse_modes(sys.argv)
#
#     settings = Settings(ruta_base)
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#
#     CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
#     CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))
#
#     # Seis splits independientes para reportar medias y desviaciones estándar.
#     seeds = [17, 1, 22, 309, 56, 44]
#
#     for seed in seeds:
#         output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
#         output_path.mkdir(parents=True, exist_ok=True)
#         OUTPUT_DIR = str(output_path)
#
#         df_split = build_split_metadata(
#             str(CORPUS_P_DIR),
#             str(CORPUS_R_DIR),
#             test_size=0.20,
#             random_state=seed,
#         )
#
#         output_file = output_path / "corpus_split.csv"
#         df_split.to_csv(output_file, index=False, encoding="utf-8")
#
#         for MODE in ["ablation"]:
#             if MODE == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv",
#                 )
#                 print("\nResultados finales:")
#                 print(results)
#                 continue
#
#             if MODE == "curves":
#                 selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
#                 generate_plots_for_algorithms(
#                     results_dir=OUTPUT_DIR,
#                     algorithm_names=selected_algorithm_names,
#                     style="tableau-colorblind10",
#                     show=False,
#                 )
#                 continue
#
#             for algorithm_id in algorithm_ids:
#                 algorithm_name = ALGORITHM_NAME[algorithm_id]
#                 model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
#                 df_split = pd.read_csv(output_file)
#
#                 if MODE == "train":
#                     train_and_save_model(
#                         df_split=df_split,
#                         algorithm_id=algorithm_id,
#                         out_dir=str(model_dir),
#                         random_state=seed,
#                     )
#
#                 elif MODE == "eval":
#                     load_model_and_evaluate(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         out_csv_path=str(output_path / f"eval_predictions_{algorithm_name}.csv"),
#                     )
#
#                 elif MODE == "features":
#                     compute_feature_importance(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         extract_features_musicxml=extract_features_musicxml,
#                         method="permutation",
#                         scoring="balanced_accuracy",
#                         n_repeats=30,
#                         random_state=seed,
#                         out_csv_path=str(output_path / f"feature_importance_{algorithm_name}.csv"),
#                     )
#
#                 elif MODE == "ablation":
#                     run_ablation_study(
#                         df_split=df_split,
#                         algorithm_id=algorithm_id,
#                         output_dir=OUTPUT_DIR,
#                         extract_features_musicxml=extract_features_musicxml,
#                         random_state=seed,
#                         style="tableau-colorblind10",
#                         force_recompute_features=False,
#                         main_metric="balanced_accuracy",
#                     )
#
#     if "ablation" in modes:
#         aggregate_ablation_results(
#             base_results_dir=str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}"),
#             seeds=seeds,
#             algorithm_names=[ALGORITHM_NAME[i] for i in algorithm_ids],
#             style="tableau-colorblind10",
#             main_metric="balanced_accuracy",
#         )


#graficas
# import sys
# import pandas as pd
# from transfolk_config import *
# from apps.db.config_registry import ConfigRegistry
# from transfolk_classifier.classifier import train_and_save_model, load_model_and_evaluate, ALGORITHM_NAME
# from transfolk_classifier.evaluate_models import evaluate_all_models
# from transfolk_features.extract_features import extract_features_musicxml
# from transfolk_classifier.feature_importance import compute_feature_importance
# from transfolk_classifier.build_corpus_split import build_split_metadata
# from transfolk_classifier.classifier_curves import generate_plots_for_algorithms
#
#
# def _parse_algorithm_ids(argv):
#     """
#     Optional CLI usage:
#         python -m apps_teimus.run_classifier <root> 3
#         python -m apps_teimus.run_classifier <root> 1,3,5
#         python -m apps_teimus.run_classifier <root> all
#
#     If omitted, all classifiers are processed.
#     """
#     if len(argv) <= 2 or str(argv[2]).lower() == "all":
#         return list(ALGORITHM_NAME.keys())
#
#     raw = str(argv[2]).strip()
#     ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
#     invalid = [x for x in ids if x not in ALGORITHM_NAME]
#     if invalid:
#         raise ValueError(f"algorithm_id inválido: {invalid}. Debe estar entre 1 y 8.")
#     return ids
#
#
# if __name__ == "__main__":
#     ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
#     algorithm_ids = _parse_algorithm_ids(sys.argv)
#
#     settings = Settings(ruta_base)
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#
#     CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
#     CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))
#
#     # Puedes cambiar esta lista de semillas sin tocar el módulo de gráficas.
#     seeds = [17, 1, 22, 309, 56, 44]
#
#     # Modos del pipeline. "curves" se ejecuta después de "eval" porque usa eval_predictions_<modelo>.csv.
#     modes = ["train", "eval", "compare", "features", "curves"]
#
#     for seed in seeds:
#         output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
#         output_path.mkdir(parents=True, exist_ok=True)
#         OUTPUT_DIR = str(output_path)
#
#         # Etiquetamos todas las obras y dividimos el corpus TRAIN/EVAL.
#         df_split = build_split_metadata(
#             str(CORPUS_P_DIR),
#             str(CORPUS_R_DIR),
#             test_size=0.20,
#             random_state=seed,
#         )
#
#         output_file = output_path / "corpus_split.csv"
#         df_split.to_csv(output_file, index=False, encoding="utf-8")
#
#         for MODE in modes:
#             if MODE.lower() == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv",
#                 )
#                 print("\nResultados finales:")
#                 print(results)
#                 continue
#
#             if MODE.lower() == "curves":
#                 selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
#                 generate_plots_for_algorithms(
#                     results_dir=OUTPUT_DIR,
#                     algorithm_names=selected_algorithm_names,
#                     style="tableau-colorblind10",  # cambia a "petroff10" si prefieres esa paleta
#                     show=False,
#                 )
#                 continue
#
#             for algorithm_id in algorithm_ids:
#                 algorithm_name = ALGORITHM_NAME[algorithm_id]
#
#                 model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
#                 df_split = pd.read_csv(output_path / "corpus_split.csv")
#
#                 if MODE.lower() == "train":
#                     train_and_save_model(
#                         df_split=df_split,
#                         algorithm_id=algorithm_id,
#                         out_dir=str(model_dir),
#                     )
#
#                 elif MODE.lower() == "eval":
#                     load_model_and_evaluate(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         out_csv_path=str(output_path / f"eval_predictions_{algorithm_name}.csv"),
#                     )
#
#                 elif MODE.lower() == "features":
#                     compute_feature_importance(
#                         model_dir=str(model_dir),
#                         df_split=df_split,
#                         extract_features_musicxml=extract_features_musicxml,
#                         method="permutation",
#                         scoring="balanced_accuracy",
#                         n_repeats=30,
#                         out_csv_path=str(output_path / f"feature_importance_{algorithm_name}.csv"),
#                     )
#
#


# import sys
# import pandas as pd
# from transfolk_config import *
# from apps.db.config_registry import ConfigRegistry
# from transfolk_classifier.classifier import train_and_save_model, load_model_and_evaluate, ALGORITHM_NAME
# from transfolk_classifier.evaluate_models import evaluate_all_models
# from transfolk_features.extract_features import extract_features_musicxml
# from transfolk_classifier.feature_importance  import compute_feature_importance
# from transfolk_classifier.build_corpus_split import build_split_metadata
#
# if __name__ == "__main__":
#     ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
#
#     settings = Settings(ruta_base)
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#     corpusE = Corpus(name=corpus.name, subcorpus="eval", id=None)
#
#     CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
#     CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))
#
#
#     for seed in [17, 1, 22, 309, 56, 44]:
#
#         # OUTPUT_DIR = str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}")
#         output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
#         output_path.mkdir(parents=True, exist_ok=True)
#
#         OUTPUT_DIR = str(output_path)
#         # Etiquetamos todas las obras y dividimos el CORPUS TRAIN/EVAL
#         df_split = build_split_metadata(
#             str(CORPUS_P_DIR),
#             str(CORPUS_R_DIR),
#             test_size=0.20,
#             random_state=seed
#         )
#
#         output_file = f"{OUTPUT_DIR}\corpus_split.csv"
#         df_split.to_csv(output_file, index=False, encoding="utf-8")
#
#
#         for MODE in ["train", "eval", "compare", "features"]:
#             # Algoritmo
#
#             if MODE.lower() == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv"
#                 )
#                 print("\nResultados finales:")
#                 print(results)
#             else:
#
#                 for algorithm_id in range(1, 9):
#                     algorithm_name = ALGORITHM_NAME[algorithm_id]
#                     # {
#                     #     1: "logistic_regression",
#                     #     2: "svm_linear_calibrated",
#                     #     3: "gradient_boosting",
#                     #     4: "random_forest",
#                     #     5: "hist_gradient_boosting",
#                     #     6: "knn",
#                     #     7: "naive_bayes",
#                     #     8: "decision_tree"
#                     # }
#
#                     # --- Rutas ---
#                     df_prof_dir = CORPUS_P_DIR
#                     df_reli_dir = CORPUS_R_DIR
#                     # Carpeta donde guardar/cargar el modelo
#                     model_dir = resolver.classifier_dir(corpus) / "teimus" / f"{seed}" / f"{algorithm_name}"
#                     # Cargamos el dataframe con corpus de training y eval
#                     df_split = pd.read_csv(f"{OUTPUT_DIR}\corpus_split.csv")
#
#                     if MODE.lower() == "train":
#                         train_and_save_model(
#                             df_split=df_split,
#                             algorithm_id=algorithm_id,
#                             out_dir=str(model_dir),
#                         )
#                     elif MODE.lower() == "eval":
#                         load_model_and_evaluate(
#                             model_dir=str(model_dir),
#                             df_split=df_split,
#                             out_csv_path = rf"{OUTPUT_DIR}\eval_predictions_{algorithm_name}.csv",
#                         )
#                     elif MODE.lower()=="features":
#                         df_imp = compute_feature_importance(
#                             model_dir=str(model_dir),
#                             df_split=df_split,
#                             extract_features_musicxml=extract_features_musicxml,
#                             method="permutation",
#                             scoring="balanced_accuracy",
#                             n_repeats=30,
#                             out_csv_path = rf"{OUTPUT_DIR}\feature_importance{algorithm_name}.csv"
#                  