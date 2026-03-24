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
    confusion_matrix
)


def evaluate_all_models(results_dir: str, out_csv: str = "model_comparison.csv") -> pd.DataFrame:
    """
    Lee todos los CSV de resultados de evaluación dentro de results_dir
    y calcula métricas comparativas entre modelos.
    """

    rows = []

    csv_files = [
        f for f in os.listdir(results_dir)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise RuntimeError("No se encontraron CSV en el directorio.")

    for fname in csv_files:

        path = os.path.join(results_dir, fname)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print("[WARN] No se pudo leer:", fname, "->", e)
            continue

        required_cols = {"y_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            print("[WARN] CSV inválido:", fname)
            continue

        # Información del modelo desde el propio CSV
        algorithm_id = df["algorithm_id"].iloc[0] if "algorithm_id" in df.columns else None
        algorithm_name = df["algorithm_name"].iloc[0] if "algorithm_name" in df.columns else fname

        y_true = df["y_true"].astype(int).values
        y_pred = df["y_pred"].astype(int).values

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        precision_macro = precision_score(y_true, y_pred, average="macro")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        try:
            auc = roc_auc_score(y_true, df["score_profano"].values)
        except Exception:
            auc = np.nan

        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "algorithm_id": algorithm_id,
            "algorithm_name": algorithm_name,

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
        })

    df_results = pd.DataFrame(rows)

    if df_results.empty:
        raise RuntimeError("No se encontraron resultados válidos.")

    df_results = df_results.sort_values("balanced_accuracy", ascending=False)

    out_path = os.path.join(results_dir, out_csv)
    df_results.to_csv(out_path, index=False)

    print("\n==============================")
    print("COMPARACIÓN DE MODELOS")
    print("==============================")
    print(df_results)

    print("\n[OK] Tabla guardada en:", out_path)

    return df_results