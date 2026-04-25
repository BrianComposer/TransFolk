import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


CLASS_CONFIGS = {
    "profano": {
        "display_name": "Profano",
        "positive_label_id": 1,
        "score_column": "score_profano",
    },
    "religioso": {
        "display_name": "Religioso",
        "positive_label_id": 0,
        "score_column": "score_religioso",
    },
}


def _add_religioso_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "score_profano" in df.columns and "score_religioso" not in df.columns:
        df["score_religioso"] = 1.0 - pd.to_numeric(df["score_profano"], errors="coerce")
    return df


def _metrics_for_positive_class(
    df: pd.DataFrame,
    algorithm_id,
    algorithm_name,
    class_key: str,
) -> dict:
    cfg = CLASS_CONFIGS[class_key]

    y_true_raw = df["y_true"].astype(int).values
    y_pred_raw = df["y_pred"].astype(int).values

    y_true = (y_true_raw == int(cfg["positive_label_id"])).astype(int)
    y_pred = (y_pred_raw == int(cfg["positive_label_id"])).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, df[cfg["score_column"]].values)
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "algorithm_id": algorithm_id,
        "algorithm_name": algorithm_name,
        "positive_class": cfg["display_name"],
        "positive_class_suffix": class_key,
        "positive_label_id": cfg["positive_label_id"],
        "score_column": cfg["score_column"],
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_all_models(results_dir: str, out_csv: str = "model_comparison.csv") -> pd.DataFrame:
    """
    Lee los CSV de evaluación dentro de results_dir y calcula métricas comparativas.

    Se generan métricas con dos convenciones de clase positiva:
        - *_profano.csv: Profano = positivo.
        - *_religioso.csv: Religioso = positivo.

    El CSV principal contiene ambas filas por modelo para que el informe final pueda
    agregar las dos perspectivas sin reentrenar ni recalcular predicciones.
    """

    rows = []

    csv_files = [
        f for f in os.listdir(results_dir)
        if f.lower().endswith(".csv") and f.startswith("eval_predictions_")
    ]

    if not csv_files:
        raise RuntimeError("No se encontraron CSV eval_predictions_<modelo>.csv en el directorio.")

    for fname in csv_files:
        path = os.path.join(results_dir, fname)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print("[WARN] No se pudo leer:", fname, "->", e)
            continue

        required_cols = {"y_true", "y_pred", "score_profano"}
        if not required_cols.issubset(df.columns):
            print("[WARN] CSV inválido:", fname)
            continue

        df = _add_religioso_score(df)

        algorithm_id = df["algorithm_id"].iloc[0] if "algorithm_id" in df.columns else None
        algorithm_name = df["algorithm_name"].iloc[0] if "algorithm_name" in df.columns else fname.replace("eval_predictions_", "").replace(".csv", "")

        for class_key in ("profano", "religioso"):
            rows.append(_metrics_for_positive_class(df, algorithm_id, algorithm_name, class_key))

    df_results = pd.DataFrame(rows)

    if df_results.empty:
        raise RuntimeError("No se encontraron resultados válidos.")

    df_results = df_results.sort_values(["positive_class_suffix", "balanced_accuracy"], ascending=[True, False])

    out_path = os.path.join(results_dir, out_csv)
    df_results.to_csv(out_path, index=False, encoding="utf-8")

    stem, ext = os.path.splitext(out_csv)
    ext = ext or ".csv"
    for class_key in ("profano", "religioso"):
        df_class = df_results[df_results["positive_class_suffix"] == class_key].copy()
        df_class.to_csv(os.path.join(results_dir, f"{stem}_{class_key}{ext}"), index=False, encoding="utf-8")

    print("\n==============================")
    print("COMPARACIÓN DE MODELOS")
    print("==============================")
    print(df_results)
    print("\n[OK] Tabla guardada en:", out_path)

    return df_results





# import os
# import pandas as pd
# import numpy as np
#
# from sklearn.metrics import (
#     accuracy_score,
#     balanced_accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix
# )
#
#
# def evaluate_all_models(results_dir: str, out_csv: str = "model_comparison.csv") -> pd.DataFrame:
#     """
#     Lee todos los CSV de resultados de evaluación dentro de results_dir
#     y calcula métricas comparativas entre modelos.
#     """
#
#     rows = []
#
#     csv_files = [
#         f for f in os.listdir(results_dir)
#         if f.lower().endswith(".csv")
#     ]
#
#     if not csv_files:
#         raise RuntimeError("No se encontraron CSV en el directorio.")
#
#     for fname in csv_files:
#
#         path = os.path.join(results_dir, fname)
#
#         try:
#             df = pd.read_csv(path)
#         except Exception as e:
#             print("[WARN] No se pudo leer:", fname, "->", e)
#             continue
#
#         required_cols = {"y_true", "y_pred"}
#         if not required_cols.issubset(df.columns):
#             print("[WARN] CSV inválido:", fname)
#             continue
#
#         # Información del modelo desde el propio CSV
#         algorithm_id = df["algorithm_id"].iloc[0] if "algorithm_id" in df.columns else None
#         algorithm_name = df["algorithm_name"].iloc[0] if "algorithm_name" in df.columns else fname
#
#         y_true = df["y_true"].astype(int).values
#         y_pred = df["y_pred"].astype(int).values
#
#         acc = accuracy_score(y_true, y_pred)
#         bacc = balanced_accuracy_score(y_true, y_pred)
#
#         precision = precision_score(y_true, y_pred)
#         recall = recall_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred)
#
#         precision_macro = precision_score(y_true, y_pred, average="macro")
#         recall_macro = recall_score(y_true, y_pred, average="macro")
#         f1_macro = f1_score(y_true, y_pred, average="macro")
#
#         try:
#             auc = roc_auc_score(y_true, df["score_profano"].values)
#         except Exception:
#             auc = np.nan
#
#         cm = confusion_matrix(y_true, y_pred)
#
#         tn, fp, fn, tp = cm.ravel()
#
#         rows.append({
#             "algorithm_id": algorithm_id,
#             "algorithm_name": algorithm_name,
#
#             "accuracy": acc,
#             "balanced_accuracy": bacc,
#
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#
#             "precision_macro": precision_macro,
#             "recall_macro": recall_macro,
#             "f1_macro": f1_macro,
#
#             "roc_auc": auc,
#
#             "tp": tp,
#             "tn": tn,
#             "fp": fp,
#             "fn": fn,
#         })
#
#     df_results = pd.DataFrame(rows)
#
#     if df_results.empty:
#         raise RuntimeError("No se encontraron resultados válidos.")
#
#     df_results = df_results.sort_values("balanced_accuracy", ascending=False)
#
#     out_path = os.path.join(results_dir, out_csv)
#     df_results.to_csv(out_path, index=False)
#
#     print("\n==============================")
#     print("COMPARACIÓN DE MODELOS")
#     print("==============================")
#     print(df_results)
#
#     print("\n[OK] Tabla guardada en:", out_path)
#
#     return df_results