# -*- coding: utf-8 -*-
import sys
from pathlib import Path
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
from transfolk_classifier.feature_distributions import aggregate_feature_distributions
from transfolk_features.extract_features import extract_features_musicxml
from transfolk_classifier.sequential_classifier import get_or_create_momet_corpus_tokenization


PER_SEED_ORDER = [
    "train",
    "eval",
    "compare",
    "features",
    "curves",
    "ablation",
    "interpretability",
]

FINAL_ORDER = [
    "ablation",
    "interpretability",
    "distributions",
    "final",
]


def _parse_algorithm_ids(argv):
    if len(argv) <= 2 or str(argv[2]).lower() == "all":
        return list(ALGORITHM_NAME.keys())

    raw = str(argv[2]).strip()
    ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    invalid = [x for x in ids if x not in ALGORITHM_NAME]
    if invalid:
        raise ValueError(f"algorithm_id inválido: {invalid}. Debe estar entre {sorted(ALGORITHM_NAME)}.")
    return ids


def _parse_modes(argv):
    default_modes = [
        "train",
        "eval",
        "compare",
        "features",
        "curves",
        "ablation",
        "interpretability",
        "distributions",
        "final",
    ]
    if len(argv) <= 3:
        return default_modes

    valid = set(default_modes)
    modes = [m.strip().lower() for m in str(argv[3]).split(",") if m.strip()]
    invalid = [m for m in modes if m not in valid]
    if invalid:
        raise ValueError(f"MODE inválido: {invalid}. Modos válidos: {sorted(valid)}")
    return modes




def _parse_bool_cli_options(argv):
    """Lee opciones key=value posteriores a los modos, por ejemplo forceTokenization=True."""
    opts = {"forceTokenization": False}
    for raw in argv[4:]:
        if "=" not in str(raw):
            continue
        key, value = str(raw).split("=", 1)
        key = key.strip()
        value = value.strip().lower()
        if key == "forceTokenization":
            opts[key] = value in {"1", "true", "yes", "y", "si", "sí"}
        else:
            print(f"[WARN] Opción CLI no reconocida y omitida: {key}")
    return opts

#ejemplo de uso por consola
# python -m apps_teimus.run_classifier . all distributions
# python -m apps_teimus.run_classifier . 10 train,eval,compare,curves,final forceTokenization=True


if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
    algorithm_ids = _parse_algorithm_ids(sys.argv)
    modes = _parse_modes(sys.argv)
    cli_options = _parse_bool_cli_options(sys.argv)
    force_tokenization = bool(cli_options.get("forceTokenization", False))

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

    # seeds = [1, 17, 22, 44, 56, 309]
    # seeds = [2, 5, 28, 74, 150, 221, 407, 553, 703]
    seeds = [1, 2, 5, 17, 22, 28, 44, 56, 74, 150, 221, 309,407, 553, 703]
    #seeds = [1, 2, 5, 17, 22]

    selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]

    selected_tabular_algorithm_names = [name for name in selected_algorithm_names if name != "cnn1d_momet"]

    # Tokenización global de corpus para modelos secuenciales. Se ejecuta una sola vez
    # antes de recorrer las seeds. Si ya existe un JSON momet_<corpus>_<fecha>.json,
    # se reutiliza; solo se regenera con forceTokenization=True.
    tokenization_json_path = None
    if "cnn1d_momet" in selected_algorithm_names:
        tokenization_dir = Path(CORPUS_P_DIR).parent / "tokenization"
        tokenization_json_path = get_or_create_momet_corpus_tokenization(
            prof_dir=CORPUS_P_DIR,
            reli_dir=CORPUS_R_DIR,
            corpus_name=corpus.name,
            tokenization_dir=tokenization_dir,
            force_tokenization=force_tokenization,
        )

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

        for MODE in PER_SEED_ORDER:
            if MODE not in modes:
                continue

            if MODE == "compare":
                results = evaluate_all_models(
                    results_dir=OUTPUT_DIR,
                    out_csv="model_comparison.csv",
                )
                print("\nResultados finales de la seed:")
                print(results)
                continue

            if MODE == "curves":
                generate_plots_for_algorithms(
                    results_dir=OUTPUT_DIR,
                    algorithm_names=selected_algorithm_names,
                    style="tableau-colorblind10",
                    show=False,
                    positive_classes=("profano", "religioso"),
                )
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
                        tokenization_json_path=str(tokenization_json_path) if algorithm_name == "cnn1d_momet" else None,
                    )

                elif MODE == "eval":
                    load_model_and_evaluate(
                        model_dir=str(model_dir),
                        df_split=df_split,
                        out_csv_path=str(output_path / f"eval_predictions_{algorithm_name}.csv"),
                        tokenization_json_path=str(tokenization_json_path) if algorithm_name == "cnn1d_momet" else None,
                    )

                elif MODE == "features":
                    if algorithm_name == "cnn1d_momet":
                        print("[INFO] Se omite feature_importance para cnn1d_momet: el modelo usa tokens secuenciales, no columnas tabulares.")
                        continue
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
                    if algorithm_name == "cnn1d_momet":
                        print("[INFO] Se omite ablation tabular para cnn1d_momet: no usa grupos de features musicales.")
                        continue
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
                    if algorithm_name == "cnn1d_momet":
                        print("[INFO] Se omite interpretability tabular para cnn1d_momet: no hay features numericas alineadas con el modelo.")
                        continue
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

    for MODE in FINAL_ORDER:
        if MODE not in modes:
            continue

        if MODE == "ablation":
            if not selected_tabular_algorithm_names:
                print("[INFO] Se omite agregado ablation: solo hay modelos secuenciales seleccionados.")
                continue
            aggregate_ablation_results(
                base_results_dir=base_results_dir,
                seeds=seeds,
                algorithm_names=selected_tabular_algorithm_names,
                style="tableau-colorblind10",
                main_metric="balanced_accuracy",
            )

        elif MODE == "interpretability":
            if not selected_tabular_algorithm_names:
                print("[INFO] Se omite agregado interpretability: solo hay modelos secuenciales seleccionados.")
                continue
            aggregate_musical_interpretability(
                base_results_dir=base_results_dir,
                seeds=seeds,
                algorithm_names=selected_tabular_algorithm_names,
                style="tableau-colorblind10",
                top_n=20,
            )

        elif MODE == "distributions":
            aggregate_feature_distributions(
                base_results_dir=base_results_dir,
                seeds=seeds,
                extract_features_musicxml=extract_features_musicxml,
                style="tableau-colorblind10",
                top_n=20,
                n_bins=40,
                max_features=None,
                force_recompute_features=False,
            )

        elif MODE == "final":
            aggregate_final_results(
                base_results_dir=base_results_dir,
                seeds=seeds,
                algorithm_names=selected_algorithm_names,
                style="tableau-colorblind10",
                main_metric="balanced_accuracy",
                top_n=20,
            )







#antes del cnn1d_momet. Funciona OK
# # -*- coding: utf-8 -*-
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
# from transfolk_classifier.final_results import aggregate_final_results
# from transfolk_classifier.feature_distributions import aggregate_feature_distributions
# from transfolk_features.extract_features import extract_features_musicxml
#
#
# PER_SEED_ORDER = [
#     "train",
#     "eval",
#     "compare",
#     "features",
#     "curves",
#     "ablation",
#     "interpretability",
# ]
#
# FINAL_ORDER = [
#     "ablation",
#     "interpretability",
#     "distributions",
#     "final",
# ]
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
#     default_modes = [
#         "train",
#         "eval",
#         "compare",
#         "features",
#         "curves",
#         "ablation",
#         "interpretability",
#         "distributions",
#         "final",
#     ]
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
# #ejemplo de uso por consola
# # python -m apps_teimus.run_classifier . all distributions
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
#     # seeds = [1, 17, 22, 44, 56, 309]
#     # seeds = [2, 5, 28, 74, 150, 221, 407, 553, 703]
#     seeds = [1, 17, 22, 44, 56, 309, 2, 5, 28, 74, 150, 221, 407, 553, 703]
#
#     selected_algorithm_names = [ALGORITHM_NAME[i] for i in algorithm_ids]
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
#         for MODE in PER_SEED_ORDER:
#             if MODE not in modes:
#                 continue
#
#             if MODE == "compare":
#                 results = evaluate_all_models(
#                     results_dir=OUTPUT_DIR,
#                     out_csv="model_comparison.csv",
#                 )
#                 print("\nResultados finales de la seed:")
#                 print(results)
#                 continue
#
#             if MODE == "curves":
#                 generate_plots_for_algorithms(
#                     results_dir=OUTPUT_DIR,
#                     algorithm_names=selected_algorithm_names,
#                     style="tableau-colorblind10",
#                     show=False,
#                     positive_classes=("profano", "religioso"),
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
#
#     for MODE in FINAL_ORDER:
#         if MODE not in modes:
#             continue
#
#         if MODE == "ablation":
#             aggregate_ablation_results(
#                 base_results_dir=base_results_dir,
#                 seeds=seeds,
#                 algorithm_names=selected_algorithm_names,
#                 style="tableau-colorblind10",
#                 main_metric="balanced_accuracy",
#             )
#
#         elif MODE == "interpretability":
#             aggregate_musical_interpretability(
#                 base_results_dir=base_results_dir,
#                 seeds=seeds,
#                 algorithm_names=selected_algorithm_names,
#                 style="tableau-colorblind10",
#                 top_n=20,
#             )
#
#         elif MODE == "distributions":
#             aggregate_feature_distributions(
#                 base_results_dir=base_results_dir,
#                 seeds=seeds,
#                 extract_features_musicxml=extract_features_musicxml,
#                 style="tableau-colorblind10",
#                 top_n=20,
#                 n_bins=40,
#                 max_features=None,
#                 force_recompute_features=False,
#             )
#
#         elif MODE == "final":
#             aggregate_final_results(
#                 base_results_dir=base_results_dir,
#                 seeds=seeds,
#                 algorithm_names=selected_algorithm_names,
#                 style="tableau-colorblind10",
#                 main_metric="balanced_accuracy",
#                 top_n=20,
#             )




