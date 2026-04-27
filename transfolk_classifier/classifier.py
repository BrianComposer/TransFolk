# -*- coding: utf-8 -*-
"""
Train/Eval separados: Profano vs Religioso (Python 3.7)

- Entrenar:
    train_and_save_model(df_prof_dir, df_reli_dir, algorithm_id, out_dir)
- Evaluar:
    load_model_and_evaluate(model_dir=out_dir, df_eval_dir, out_csv_path)

IMPORTANTE:
    Debes conectar extract_features_musicxml(path) de tu proyecto.
"""

import os
import glob
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from transfolk_classifier.mlp_classifier import build_mlp_classifier

from transfolk_features.extract_features import extract_features_musicxml


MUSIC_EXTS = (".musicxml", ".xml", ".mxl")
ALGORITHM_NAME = {
    1: "logistic_regression",
    2: "svm_linear_calibrated",
    3: "gradient_boosting",
    4: "random_forest",
    5: "hist_gradient_boosting",
    6: "knn",
    7: "naive_bayes",
    8: "decision_tree",
    9: "mlp_features"
}

def list_music_files(folder: str) -> List[str]:
    paths = []
    for ext in ("*.musicxml", "*.xml", "*.mxl"):
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(set(paths))


def build_feature_df_from_folder(folder: str) -> pd.DataFrame:
    files = list_music_files(folder)
    rows = []

    for p in files:
        try:
            feats = extract_features_musicxml(p)
            if not isinstance(feats, dict) or len(feats) == 0:
                continue
            row = dict(feats)
            row["__path"] = p
            row["__file"] = os.path.basename(p)
            rows.append(row)
        except Exception as e:
            print("[WARN] Error features:", os.path.basename(p), "->", repr(e))
            continue

    if not rows:
        raise RuntimeError("No se extrajeron features de: %s" % folder)

    return pd.DataFrame(rows)




def make_model(algorithm_id: int, random_state: int = 42) -> Pipeline:

    if algorithm_id == 1:
        # Logistic Regression
        clf = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            random_state=random_state
        )

    elif algorithm_id == 2:
        # Linear SVM calibrado para probabilidades
        base = LinearSVC(C=1.0, max_iter=10000, dual=False, random_state=random_state)
        clf = CalibratedClassifierCV(
            estimator=base,   # <-- cambio importante en sklearn 1.4
            method="sigmoid",
            cv=3
        )

    elif algorithm_id == 3:
        # Gradient Boosting clásico
        clf = GradientBoostingClassifier(
            random_state=random_state
        )

    elif algorithm_id == 4:
        # Random Forest
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=random_state
        )

    elif algorithm_id == 5:
        # HistGradientBoosting (más moderno y rápido)
        clf = HistGradientBoostingClassifier(
            random_state=random_state
        )

    elif algorithm_id == 6:
        # kNN
        clf = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        )

    elif algorithm_id == 7:
        # Naive Bayes gaussiano
        clf = GaussianNB()

    elif algorithm_id == 8:
        # Decision Tree
        clf = DecisionTreeClassifier(
            max_depth=None,
            random_state=random_state
        )

    elif algorithm_id == 9:
        # MLP sobre el vector tabular de features musicales ya extraídas.
        # La imputación y el escalado se mantienen en el Pipeline común.
        clf = build_mlp_classifier(random_state=random_state)

    else:
        raise ValueError(
            f"algorithm_id debe estar en {sorted(ALGORITHM_NAME)}."
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

def _predict_with_scores(model: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba(X)
        score = proba[:, 1]
        pred = (score >= 0.5).astype(int)
        return pred, score

    # fallback: decision_function normalizado
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X)
        dec = np.asarray(dec, dtype=float)
        lo, hi = float(np.min(dec)), float(np.max(dec))
        if np.isclose(lo, hi):
            score = np.ones_like(dec) * 0.5
        else:
            score = (dec - lo) / (hi - lo)
        pred = (score >= 0.5).astype(int)
        return pred, score

    pred = model.predict(X).astype(int)
    return pred, pred.astype(float)


# ============================================================
# 3) TRAIN -> guarda modelo + columnas
# ============================================================
def train_and_save_model(
    df_split: pd.DataFrame,
    algorithm_id: int,
    out_dir: str,
    label_prof: str = "Profano",
    label_reli: str = "Religioso",
    random_state: int = 42,
) -> str:

    os.makedirs(out_dir, exist_ok=True)

    df_meta = df_split[df_split["split"] == "train"].copy()

    rows = []

    print("[INFO] Extrayendo features train...")

    for _, r in df_meta.iterrows():
        p = r["__path"]
        y = int(r["label_id"])

        try:
            feats = extract_features_musicxml(p)
            if not feats:
                continue

            row = dict(feats)
            row["__path"] = p
            row["__file"] = os.path.basename(p)
            row["__y"] = y

            rows.append(row)

        except Exception as e:
            print("[WARN] Error extrayendo features:", os.path.basename(p), "->", repr(e))

    if not rows:
        raise RuntimeError("No se pudieron extraer features del training set.")

    df_train = pd.DataFrame(rows)

    meta_cols = ["__path", "__file", "__y"]
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    for c in feature_cols:
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce")

    X = df_train[feature_cols]
    y = df_train["__y"].astype(int).values

    model = make_model(algorithm_id, random_state=random_state)
    model.fit(X, y)

    model_path = os.path.join(out_dir, "model.joblib")
    feat_path = os.path.join(out_dir, "features.json")
    summ_path = os.path.join(out_dir, "train_summary.json")

    joblib.dump(model, model_path)

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    summary = {
        "algorithm_id": algorithm_id,
        "algorithm_name": ALGORITHM_NAME[algorithm_id],
        "label_prof": label_prof,
        "label_reli": label_reli,
        "n_train": int(len(df_train)),
        "n_prof": int((y == 1).sum()),
        "n_reli": int((y == 0).sum()),
        "n_features": int(len(feature_cols)),
    }

    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] Modelo guardado en:", model_path)

    return out_dir



# ============================================================
# 4) EVAL -> carga modelo + columnas y evalúa carpeta
# ============================================================
def load_model_and_evaluate(
    model_dir: str,
    df_split: pd.DataFrame,
    out_csv_path: str = "eval_predictions.csv",
) -> pd.DataFrame:

    model_path = os.path.join(model_dir, "model.joblib")
    feat_path = os.path.join(model_dir, "features.json")
    summ_path = os.path.join(model_dir, "train_summary.json")

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        raise FileNotFoundError("No encuentro model.joblib o features.json en: %s" % model_dir)

    model = joblib.load(model_path)

    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    label_prof = "Profano"
    label_reli = "Religioso"

    # nuevos metadatos del modelo
    algorithm_id = None
    algorithm_name = os.path.basename(model_dir)

    if os.path.exists(summ_path):
        with open(summ_path, "r", encoding="utf-8") as f:
            summ = json.load(f)
            label_prof = summ.get("label_prof", label_prof)
            label_reli = summ.get("label_reli", label_reli)
            algorithm_id = summ.get("algorithm_id", None)
            algorithm_name = summ.get("algorithm_name", algorithm_name)

    df_meta = df_split[df_split["split"].isin(["test", "eval"])].copy()

    rows = []

    print("[INFO] Extrayendo features eval...")

    for _, r in df_meta.iterrows():

        p = r["__path"]
        y_true = int(r["label_id"])

        try:
            feats = extract_features_musicxml(p)

            if not feats:
                continue

            row = {c: feats.get(c, np.nan) for c in feature_cols}

            row["__path"] = p
            row["__file"] = os.path.basename(p)
            row["__y_true"] = y_true

            rows.append(row)

        except Exception as e:
            print("[WARN] Error eval features:", os.path.basename(p), "->", repr(e))

    if not rows:
        raise RuntimeError("No se extrajeron features del conjunto de evaluación.")

    df_eval = pd.DataFrame(rows)

    for c in feature_cols:
        df_eval[c] = pd.to_numeric(df_eval[c], errors="coerce")

    X_eval = df_eval[feature_cols]

    y_pred, score_prof = _predict_with_scores(model, X_eval)

    df_out = pd.DataFrame({
        "file": df_eval["__file"].values,
        "path": df_eval["__path"].values,
        "y_true": df_eval["__y_true"].values,
        "y_pred": y_pred,
        "score_profano": score_prof,
        "score_religioso": 1.0 - np.asarray(score_prof, dtype=float),
        "label_pred": [label_prof if yp == 1 else label_reli for yp in y_pred],

        # nuevas columnas de metadatos del modelo
        "algorithm_id": algorithm_id,
        "algorithm_name": algorithm_name,
    })

    df_out.to_csv(out_csv_path, index=False, encoding="utf-8")

    print("[OK] CSV guardado:", out_csv_path)

    y_true = df_out["y_true"].astype(int).values
    y_pred_int = df_out["y_pred"].astype(int).values

    acc = accuracy_score(y_true, y_pred_int)
    bacc = balanced_accuracy_score(y_true, y_pred_int)
    f1w = f1_score(y_true, y_pred_int, average="weighted")

    try:
        auc = roc_auc_score(y_true, df_out["score_profano"].values)
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred_int)

    print("\n==============================")
    print("RESUMEN EVALUACIÓN")
    print("==============================")

    print("Modelo:", algorithm_name)
    print("Accuracy:          %.4f" % acc)
    print("Balanced accuracy: %.4f" % bacc)
    print("F1 weighted:       %.4f" % f1w)

    if np.isfinite(auc):
        print("ROC-AUC:           %.4f" % auc)

    print("\nMatriz de confusión")
    print(cm)

    print("\nClassification report")
    print(classification_report(y_true, y_pred_int, target_names=[label_reli, label_prof]))

    return df_out


#antes del MLP funciona OK
# # -*- coding: utf-8 -*-
# """
# Train/Eval separados: Profano vs Religioso (Python 3.7)
#
# - Entrenar:
#     train_and_save_model(df_prof_dir, df_reli_dir, algorithm_id, out_dir)
# - Evaluar:
#     load_model_and_evaluate(model_dir=out_dir, df_eval_dir, out_csv_path)
#
# IMPORTANTE:
#     Debes conectar extract_features_musicxml(path) de tu proyecto.
# """
#
# import os
# import glob
# import json
# from typing import List, Tuple
#
# import numpy as np
# import pandas as pd
# import joblib
#
# from sklearn.metrics import (
#     accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
#     confusion_matrix, classification_report
# )
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
#
# from transfolk_features.extract_features import extract_features_musicxml
#
#
# MUSIC_EXTS = (".musicxml", ".xml", ".mxl")
# ALGORITHM_NAME = {
#     1: "logistic_regression",
#     2: "svm_linear_calibrated",
#     3: "gradient_boosting",
#     4: "random_forest",
#     5: "hist_gradient_boosting",
#     6: "knn",
#     7: "naive_bayes",
#     8: "decision_tree"
# }
#
# def list_music_files(folder: str) -> List[str]:
#     paths = []
#     for ext in ("*.musicxml", "*.xml", "*.mxl"):
#         paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
#     return sorted(set(paths))
#
#
# def build_feature_df_from_folder(folder: str) -> pd.DataFrame:
#     files = list_music_files(folder)
#     rows = []
#
#     for p in files:
#         try:
#             feats = extract_features_musicxml(p)
#             if not isinstance(feats, dict) or len(feats) == 0:
#                 continue
#             row = dict(feats)
#             row["__path"] = p
#             row["__file"] = os.path.basename(p)
#             rows.append(row)
#         except Exception as e:
#             print("[WARN] Error features:", os.path.basename(p), "->", repr(e))
#             continue
#
#     if not rows:
#         raise RuntimeError("No se extrajeron features de: %s" % folder)
#
#     return pd.DataFrame(rows)
#
#
#
#
# def make_model(algorithm_id: int, random_state: int = 42) -> Pipeline:
#
#     if algorithm_id == 1:
#         # Logistic Regression
#         clf = LogisticRegression(
#             solver="liblinear",
#             penalty="l2",
#             C=1.0,
#             max_iter=2000,
#             random_state=random_state
#         )
#
#     elif algorithm_id == 2:
#         # Linear SVM calibrado para probabilidades
#         base = LinearSVC(C=1.0, max_iter=10000, dual=False, random_state=random_state)
#         clf = CalibratedClassifierCV(
#             estimator=base,   # <-- cambio importante en sklearn 1.4
#             method="sigmoid",
#             cv=3
#         )
#
#     elif algorithm_id == 3:
#         # Gradient Boosting clásico
#         clf = GradientBoostingClassifier(
#             random_state=random_state
#         )
#
#     elif algorithm_id == 4:
#         # Random Forest
#         clf = RandomForestClassifier(
#             n_estimators=300,
#             max_depth=None,
#             min_samples_split=2,
#             n_jobs=-1,
#             random_state=random_state
#         )
#
#     elif algorithm_id == 5:
#         # HistGradientBoosting (más moderno y rápido)
#         clf = HistGradientBoostingClassifier(
#             random_state=random_state
#         )
#
#     elif algorithm_id == 6:
#         # kNN
#         clf = KNeighborsClassifier(
#             n_neighbors=7,
#             weights="distance"
#         )
#
#     elif algorithm_id == 7:
#         # Naive Bayes gaussiano
#         clf = GaussianNB()
#
#     elif algorithm_id == 8:
#         # Decision Tree
#         clf = DecisionTreeClassifier(
#             max_depth=None,
#             random_state=random_state
#         )
#
#     else:
#         raise ValueError(
#             "algorithm_id debe estar entre 1 y 8."
#         )
#
#     return Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#             ("clf", clf),
#         ]
#     )
#
# def _predict_with_scores(model: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
#     clf = model.named_steps["clf"]
#     if hasattr(clf, "predict_proba"):
#         proba = model.predict_proba(X)
#         score = proba[:, 1]
#         pred = (score >= 0.5).astype(int)
#         return pred, score
#
#     # fallback: decision_function normalizado
#     if hasattr(model, "decision_function"):
#         dec = model.decision_function(X)
#         dec = np.asarray(dec, dtype=float)
#         lo, hi = float(np.min(dec)), float(np.max(dec))
#         if np.isclose(lo, hi):
#             score = np.ones_like(dec) * 0.5
#         else:
#             score = (dec - lo) / (hi - lo)
#         pred = (score >= 0.5).astype(int)
#         return pred, score
#
#     pred = model.predict(X).astype(int)
#     return pred, pred.astype(float)
#
#
# # ============================================================
# # 3) TRAIN -> guarda modelo + columnas
# # ============================================================
# def train_and_save_model(
#     df_split: pd.DataFrame,
#     algorithm_id: int,
#     out_dir: str,
#     label_prof: str = "Profano",
#     label_reli: str = "Religioso",
#     random_state: int = 42,
# ) -> str:
#
#     os.makedirs(out_dir, exist_ok=True)
#
#     df_meta = df_split[df_split["split"] == "train"].copy()
#
#     rows = []
#
#     print("[INFO] Extrayendo features train...")
#
#     for _, r in df_meta.iterrows():
#         p = r["__path"]
#         y = int(r["label_id"])
#
#         try:
#             feats = extract_features_musicxml(p)
#             if not feats:
#                 continue
#
#             row = dict(feats)
#             row["__path"] = p
#             row["__file"] = os.path.basename(p)
#             row["__y"] = y
#
#             rows.append(row)
#
#         except Exception as e:
#             print("[WARN] Error extrayendo features:", os.path.basename(p), "->", repr(e))
#
#     if not rows:
#         raise RuntimeError("No se pudieron extraer features del training set.")
#
#     df_train = pd.DataFrame(rows)
#
#     meta_cols = ["__path", "__file", "__y"]
#     feature_cols = [c for c in df_train.columns if c not in meta_cols]
#
#     for c in feature_cols:
#         df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
#
#     X = df_train[feature_cols]
#     y = df_train["__y"].astype(int).values
#
#     model = make_model(algorithm_id, random_state=random_state)
#     model.fit(X, y)
#
#     model_path = os.path.join(out_dir, "model.joblib")
#     feat_path = os.path.join(out_dir, "features.json")
#     summ_path = os.path.join(out_dir, "train_summary.json")
#
#     joblib.dump(model, model_path)
#
#     with open(feat_path, "w", encoding="utf-8") as f:
#         json.dump(feature_cols, f, ensure_ascii=False, indent=2)
#
#     summary = {
#         "algorithm_id": algorithm_id,
#         "algorithm_name": ALGORITHM_NAME[algorithm_id],
#         "label_prof": label_prof,
#         "label_reli": label_reli,
#         "n_train": int(len(df_train)),
#         "n_prof": int((y == 1).sum()),
#         "n_reli": int((y == 0).sum()),
#         "n_features": int(len(feature_cols)),
#     }
#
#     with open(summ_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)
#
#     print("[OK] Modelo guardado en:", model_path)
#
#     return out_dir
#
#
#
# # ============================================================
# # 4) EVAL -> carga modelo + columnas y evalúa carpeta
# # ============================================================
# def load_model_and_evaluate(
#     model_dir: str,
#     df_split: pd.DataFrame,
#     out_csv_path: str = "eval_predictions.csv",
# ) -> pd.DataFrame:
#
#     model_path = os.path.join(model_dir, "model.joblib")
#     feat_path = os.path.join(model_dir, "features.json")
#     summ_path = os.path.join(model_dir, "train_summary.json")
#
#     if not os.path.exists(model_path) or not os.path.exists(feat_path):
#         raise FileNotFoundError("No encuentro model.joblib o features.json en: %s" % model_dir)
#
#     model = joblib.load(model_path)
#
#     with open(feat_path, "r", encoding="utf-8") as f:
#         feature_cols = json.load(f)
#
#     label_prof = "Profano"
#     label_reli = "Religioso"
#
#     # nuevos metadatos del modelo
#     algorithm_id = None
#     algorithm_name = os.path.basename(model_dir)
#
#     if os.path.exists(summ_path):
#         with open(summ_path, "r", encoding="utf-8") as f:
#             summ = json.load(f)
#             label_prof = summ.get("label_prof", label_prof)
#             label_reli = summ.get("label_reli", label_reli)
#             algorithm_id = summ.get("algorithm_id", None)
#             algorithm_name = summ.get("algorithm_name", algorithm_name)
#
#     df_meta = df_split[df_split["split"].isin(["test", "eval"])].copy()
#
#     rows = []
#
#     print("[INFO] Extrayendo features eval...")
#
#     for _, r in df_meta.iterrows():
#
#         p = r["__path"]
#         y_true = int(r["label_id"])
#
#         try:
#             feats = extract_features_musicxml(p)
#
#             if not feats:
#                 continue
#
#             row = {c: feats.get(c, np.nan) for c in feature_cols}
#
#             row["__path"] = p
#             row["__file"] = os.path.basename(p)
#             row["__y_true"] = y_true
#
#             rows.append(row)
#
#         except Exception as e:
#             print("[WARN] Error eval features:", os.path.basename(p), "->", repr(e))
#
#     if not rows:
#         raise RuntimeError("No se extrajeron features del conjunto de evaluación.")
#
#     df_eval = pd.DataFrame(rows)
#
#     for c in feature_cols:
#         df_eval[c] = pd.to_numeric(df_eval[c], errors="coerce")
#
#     X_eval = df_eval[feature_cols]
#
#     y_pred, score_prof = _predict_with_scores(model, X_eval)
#
#     df_out = pd.DataFrame({
#         "file": df_eval["__file"].values,
#         "path": df_eval["__path"].values,
#         "y_true": df_eval["__y_true"].values,
#         "y_pred": y_pred,
#         "score_profano": score_prof,
#         "score_religioso": 1.0 - np.asarray(score_prof, dtype=float),
#         "label_pred": [label_prof if yp == 1 else label_reli for yp in y_pred],
#
#         # nuevas columnas de metadatos del modelo
#         "algorithm_id": algorithm_id,
#         "algorithm_name": algorithm_name,
#     })
#
#     df_out.to_csv(out_csv_path, index=False, encoding="utf-8")
#
#     print("[OK] CSV guardado:", out_csv_path)
#
#     y_true = df_out["y_true"].astype(int).values
#     y_pred_int = df_out["y_pred"].astype(int).values
#
#     acc = accuracy_score(y_true, y_pred_int)
#     bacc = balanced_accuracy_score(y_true, y_pred_int)
#     f1w = f1_score(y_true, y_pred_int, average="weighted")
#
#     try:
#         auc = roc_auc_score(y_true, df_out["score_profano"].values)
#     except Exception:
#         auc = np.nan
#
#     cm = confusion_matrix(y_true, y_pred_int)
#
#     print("\n==============================")
#     print("RESUMEN EVALUACIÓN")
#     print("==============================")
#
#     print("Modelo:", algorithm_name)
#     print("Accuracy:          %.4f" % acc)
#     print("Balanced accuracy: %.4f" % bacc)
#     print("F1 weighted:       %.4f" % f1w)
#
#     if np.isfinite(auc):
#         print("ROC-AUC:           %.4f" % auc)
#
#     print("\nMatriz de confusión")
#     print(cm)
#
#     print("\nClassification report")
#     print(classification_report(y_true, y_pred_int, target_names=[label_reli, label_prof]))
#
#     return df_out
#
