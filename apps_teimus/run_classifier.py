import sys
import pandas as pd
from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_classifier.classifier import train_and_save_model, load_model_and_evaluate, ALGORITHM_NAME
from transfolk_classifier.evaluate_models import evaluate_all_models
from transfolk_features.extract_features import extract_features_musicxml
from transfolk_classifier.feature_importance  import compute_feature_importance
from transfolk_classifier.build_corpus_split import build_split_metadata

if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None

    settings = Settings(ruta_base)
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name("ataa")
    corpusP = Corpus(name=corpus.name, subcorpus="profano", id=None)
    corpusR = Corpus(name=corpus.name, subcorpus="religioso", id=None)
    corpusE = Corpus(name=corpus.name, subcorpus="eval", id=None)

    CORPUS_P_DIR = str(resolver.data_clean(corpusP, corpusP.subcorpus))
    CORPUS_R_DIR = str(resolver.data_clean(corpusR, corpusR.subcorpus))


    for seed in [44, 1, 22, 309, 56, 105]:

        # OUTPUT_DIR = str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}")
        output_path = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" / f"{seed}"
        output_path.mkdir(parents=True, exist_ok=True)

        OUTPUT_DIR = str(output_path)
        # Etiquetamos todas las obras y dividimos el CORPUS TRAIN/EVAL
        df_split = build_split_metadata(
            str(CORPUS_P_DIR),
            str(CORPUS_R_DIR),
            test_size=0.20,
            random_state=seed
        )

        output_file = f"{OUTPUT_DIR}\corpus_split.csv"
        df_split.to_csv(output_file, index=False, encoding="utf-8")


        for MODE in [ "train" , "eval" , "compare" , "features"]:
            # Algoritmo

            if MODE.lower() == "compare":
                results = evaluate_all_models(
                    results_dir=OUTPUT_DIR,
                    out_csv="model_comparison.csv"
                )
                print("\nResultados finales:")
                print(results)
            else:

                for algorithm_id in range(1, 9):
                    algorithm_name = ALGORITHM_NAME[algorithm_id]
                    # {
                    #     1: "logistic_regression",
                    #     2: "svm_linear_calibrated",
                    #     3: "gradient_boosting",
                    #     4: "random_forest",
                    #     5: "hist_gradient_boosting",
                    #     6: "knn",
                    #     7: "naive_bayes",
                    #     8: "decision_tree"
                    # }

                    # --- Rutas ---
                    df_prof_dir = CORPUS_P_DIR
                    df_reli_dir = CORPUS_R_DIR
                    # Carpeta donde guardar/cargar el modelo

                    model_dir = rf"{resolver.classifier_dir(corpus)}\teimus\{algorithm_name}"
                    # Cargamos el dataframe con corpus de training y eval
                    df_split = pd.read_csv(f"{OUTPUT_DIR}\corpus_split.csv")

                    if MODE.lower() == "train":
                        train_and_save_model(
                            df_split=df_split,
                            algorithm_id=algorithm_id,
                            out_dir=model_dir,
                        )
                    elif MODE.lower() == "eval":
                        load_model_and_evaluate(
                            model_dir=model_dir,
                            df_split=df_split,
                            out_csv_path=f"{OUTPUT_DIR}\eval_predictions_{algorithm_name}.csv",
                        )
                    elif MODE.lower()=="features":
                        df_imp = compute_feature_importance(
                            model_dir=model_dir,
                            df_split=df_split,
                            extract_features_musicxml=extract_features_musicxml,
                            method="permutation",
                            scoring="balanced_accuracy",
                            n_repeats=30,
                            out_csv_path=f"{OUTPUT_DIR}\feature_importance{algorithm_name}.csv"
                        )