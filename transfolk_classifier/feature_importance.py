# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV


def compute_feature_importance(
    model_dir: str,
    df_split: pd.DataFrame,
    extract_features_musicxml,
    method: str = "permutation",
    scoring: str = "balanced_accuracy",
    n_repeats: int = 30,
    random_state: int = 42,
    out_csv_path: str = None,
) -> pd.DataFrame:
    """
    Calcula la importancia de variables de un modelo ya entrenado.

    Parámetros
    ----------
    model_dir : str
        Carpeta del modelo. Debe contener:
            - model.joblib
            - features.json
            - train_summary.json (opcional)

    df_split : pd.DataFrame
        DataFrame con al menos estas columnas:
            - __path
            - label_id
            - split

    extract_features_musicxml : callable
        Función de extracción de rasgos que recibe una ruta y devuelve dict.

    method : str
        "permutation" (recomendado) o "native".

    scoring : str
        Métrica usada por permutation importance.

    n_repeats : int
        Número de repeticiones en permutation importance.

    random_state : int
        Semilla aleatoria.

    out_csv_path : str or None
        Si no es None, guarda el resultado en CSV.

    Devuelve
    --------
    pd.DataFrame
        Ranking de features con columnas:
            - feature
            - importance
            - importance_std
            - algorithm_id
            - algorithm_name
            - method

        Si method="native" y el modelo es lineal, añade:
            - signed_weight
    """

    def _build_eval_feature_df(
        df_split_local: pd.DataFrame,
        feature_cols_local: list
    ) -> pd.DataFrame:
        df_meta = df_split_local[df_split_local["split"].isin(["test", "eval"])].copy()

        rows = []

        print("[INFO] Extrayendo features eval para importancia de variables...")

        for _, r in df_meta.iterrows():
            p = r["__path"]
            y_true = int(r["label_id"])

            try:
                feats = extract_features_musicxml(p)
                if not feats:
                    continue

                row = {c: feats.get(c, np.nan) for c in feature_cols_local}
                row["__path"] = p
                row["__file"] = os.path.basename(p)
                row["__y_true"] = y_true
                rows.append(row)

            except Exception as e:
                print("[WARN] Error eval features:", os.path.basename(p), "->", repr(e))

        if not rows:
            raise RuntimeError("No se extrajeron features del conjunto de evaluación.")

        df_eval_local = pd.DataFrame(rows)

        for c in feature_cols_local:
            df_eval_local[c] = pd.to_numeric(df_eval_local[c], errors="coerce")

        return df_eval_local

    model_path = os.path.join(model_dir, "model.joblib")
    feat_path = os.path.join(model_dir, "features.json")
    summ_path = os.path.join(model_dir, "train_summary.json")

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        raise FileNotFoundError("No encuentro model.joblib o features.json en: %s" % model_dir)

    model = joblib.load(model_path)

    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    algorithm_id = None
    algorithm_name = os.path.basename(model_dir)

    if os.path.exists(summ_path):
        with open(summ_path, "r", encoding="utf-8") as f:
            summ = json.load(f)
            algorithm_id = summ.get("algorithm_id", None)
            algorithm_name = summ.get("algorithm_name", algorithm_name)

    df_eval = _build_eval_feature_df(df_split_local=df_split, feature_cols_local=feature_cols)

    X_eval = df_eval[feature_cols]
    y_eval = df_eval["__y_true"].astype(int).values

    clf = model.named_steps["clf"]

    if method.lower() == "permutation":
        result = permutation_importance(
            estimator=model,
            X=X_eval,
            y=y_eval,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        df_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        })

    elif method.lower() == "native":
        if hasattr(clf, "feature_importances_"):
            df_imp = pd.DataFrame({
                "feature": feature_cols,
                "importance": clf.feature_importances_,
                "importance_std": np.nan,
            })

        elif hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_)
            if coef.ndim == 2:
                coef = coef[0]

            df_imp = pd.DataFrame({
                "feature": feature_cols,
                "importance": np.abs(coef),
                "importance_std": np.nan,
                "signed_weight": coef,
            })

        elif isinstance(clf, CalibratedClassifierCV) and hasattr(clf, "estimator"):
            base = clf.estimator

            if hasattr(base, "coef_"):
                coef = np.asarray(base.coef_)
                if coef.ndim == 2:
                    coef = coef[0]

                df_imp = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": np.abs(coef),
                    "importance_std": np.nan,
                    "signed_weight": coef,
                })
            else:
                raise ValueError(
                    "El clasificador calibrado no expone coef_. Usa method='permutation'."
                )

        else:
            raise ValueError(
                "Este modelo no soporta importancia nativa. Usa method='permutation'."
            )

    else:
        raise ValueError("method debe ser 'permutation' o 'native'.")

    df_imp["algorithm_id"] = algorithm_id
    df_imp["algorithm_name"] = algorithm_name
    df_imp["method"] = method.lower()

    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)

    if out_csv_path is not None:
        df_imp.to_csv(out_csv_path, index=False, encoding="utf-8")
        print("[OK] Importancia de variables guardada en:", out_csv_path)

    print("\n==============================")
    print("IMPORTANCIA DE VARIABLES")
    print("==============================")
    print("Modelo:", algorithm_name)
    print("Método:", method.lower())
    #print(df_imp.head(20))
    print(df_imp.head(20).to_csv(sep="\t", index=False))

    return df_imp