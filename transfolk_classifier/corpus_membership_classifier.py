# -*- coding: utf-8 -*-
"""
corpus_membership_classifier.py
Clasificador no supervisado de pertenencia estilística para corpus MusicXML.

Modo 'train': entrena el modelo a partir del corpus base y guarda los artefactos.
Modo 'evaluate': carga el modelo entrenado y evalúa nuevas obras, mostrando porcentajes de pertenencia.
"""
import math
import os, glob, warnings, joblib
import numpy as np
import pandas as pd
from music21 import converter, meter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# Extracción de características seguras
# =========================================================

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def safe_numeric_array(values):
    """Convierte una lista en array float limpio (sin None, NaN, strings)."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.array([0.0])
    return arr


def entropy(arr, bins):
    """Entropía de Shannon segura."""
    arr = safe_numeric_array(arr)
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def extract_features(path):
    try:
        score = converter.parse(path)
    except Exception:
        return None

    flat = score.flat.notesAndRests
    n_notes, n_rests, durations, pitches = 0, 0, [], []

    for el in flat:
        try:
            dur = float(getattr(el, "quarterLength", 0.0))
        except Exception:
            dur = 0.0
        if el.isNote:
            n_notes += 1
            durations.append(dur)
            pitches.append(float(el.pitch.midi))
        elif el.isRest:
            n_rests += 1
            durations.append(dur)

    durations = safe_numeric_array(durations)
    pitches = safe_numeric_array(pitches)

    total_dur = float(np.sum(durations))
    mean_dur = float(np.mean(durations))
    std_dur = float(np.std(durations))
    mean_pitch = float(np.mean(pitches)) if pitches.size else 60.0
    std_pitch = float(np.std(pitches)) if pitches.size else 0.0

    pitch_entropy = entropy([p % 12 for p in pitches], bins=12)
    dur_entropy = entropy(durations, bins=8)

    try:
        ts = score.recurse().getElementsByClass(meter.TimeSignature)
        beats = float(ts[0].beatCount) if ts else 4.0
    except Exception:
        beats = 4.0

    notes_per_bar = safe_div(n_notes, safe_div(total_dur, beats))

    return dict(
        n_notes=n_notes,
        n_rests=n_rests,
        total_dur=total_dur,
        mean_dur=mean_dur,
        std_dur=std_dur,
        mean_pitch=mean_pitch,
        std_pitch=std_pitch,
        pitch_entropy=pitch_entropy,
        dur_entropy=dur_entropy,
        notes_per_bar=notes_per_bar,
    )


def load_features_from_dir(path):
    # paths = sorted(glob.glob(os.path.join(path, "**", "*.musicxml"), recursive=True))
    # if not paths:
    #     paths = sorted(glob.glob(os.path.join(path, "**", "*.xml"), recursive=True))
    patterns = ["*.musicxml", "*.xml", "*.mxl"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))

    paths = sorted(paths)

    feats, valid = [], []
    for i in range(len(paths)):
        p=paths[i]
        print(f"\rProcesando {i}/{len(paths)}")
        f = extract_features(p)
        if f:
            feats.append(f)
            valid.append(p)
    if not feats:
        raise RuntimeError(f"No se extrajeron características válidas de {path}")
    return pd.DataFrame(feats, index=valid)


# =========================================================
# ENTRENAMIENTO
# =========================================================

def train_model(corpus_dir, model_dir, pca_components=10, nu=0.05, percentile_threshold=97.5):
    print(f"Entrenando modelo con corpus: {corpus_dir}")
    df_corpus = load_features_from_dir(corpus_dir)
    print(f"  {len(df_corpus)} obras cargadas.\n")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_corpus = scaler.fit_transform(imputer.fit_transform(df_corpus))

    if pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        X_corpus = pca.fit_transform(X_corpus)
    else:
        pca = None

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    model.fit(X_corpus)

    scores_corpus = -model.decision_function(X_corpus)
    threshold = np.percentile(scores_corpus, percentile_threshold)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(imputer, os.path.join(model_dir, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    if pca:
        joblib.dump(pca, os.path.join(model_dir, "pca.joblib"))
    joblib.dump(threshold, os.path.join(model_dir, "threshold.joblib"))
    joblib.dump(list(df_corpus.columns), os.path.join(model_dir, "features.joblib"))

    print("Modelo guardado correctamente.\n")


# =========================================================
# EVALUACIÓN
# =========================================================
# =========================================================
# EVALUACIÓN (con desviación típica)
# =========================================================

def evaluate_model(new_dir, model_dir,show= True):
    if show:
        print(f"Evaluando nuevas obras en: {new_dir}")
    df_new = load_features_from_dir(new_dir)
    if show:
        print(f"  {len(df_new)} obras nuevas cargadas.\n")


    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    threshold = joblib.load(os.path.join(model_dir, "threshold.joblib"))
    features = joblib.load(os.path.join(model_dir, "features.joblib"))
    pca_path = os.path.join(model_dir, "pca.joblib")
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None

    for col in features:
        if col not in df_new.columns:
            df_new[col] = np.nan
    df_new = df_new[features]

    X_new = scaler.transform(imputer.transform(df_new))
    if pca:
        X_new = pca.transform(X_new)

    scores_new = -model.decision_function(X_new)
    belongs = scores_new <= threshold
    perc_per_work = 100 * (1 - (scores_new - scores_new.min()) /
                           (threshold - scores_new.min() + 1e-9))
    perc_per_work = np.clip(perc_per_work, 0, 100)

    global_membership = 100 * belongs.mean()
    # std_membership = float(np.std(perc_per_work))

    if show:
        print("Resultados individuales:\n")
    std_membership=0
    N = 0
    for i, obra in enumerate(df_new.index):
        if show:
            print(f"{os.path.basename(obra):40s} -> {perc_per_work[i]:6.2f}% pertenencia {'✓' if belongs[i] else '✗'}")
        std_membership=std_membership + math.pow(perc_per_work[i]-global_membership,2)
        N+=1
    std_membership=math.pow(std_membership,0.5)/N

    if show:

        print("\n--------------------------------------------")
        print(f"Porcentaje global de pertenencia del nuevo corpus: {global_membership:.2f}%")
        print(f"Desviación típica del membership global:           {std_membership:.2f}%")
        print("--------------------------------------------\n")

    return global_membership, std_membership



# def evaluate_model(new_dir, model_dir):
#     print(f"Evaluando nuevas obras en: {new_dir}")
#     df_new = load_features_from_dir(new_dir)
#     print(f"  {len(df_new)} obras nuevas cargadas.\n")
#
#     model = joblib.load(os.path.join(model_dir, "model.joblib"))
#     imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
#     scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
#     threshold = joblib.load(os.path.join(model_dir, "threshold.joblib"))
#     features = joblib.load(os.path.join(model_dir, "features.joblib"))
#     pca_path = os.path.join(model_dir, "pca.joblib")
#     pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
#
#     # Asegurar columnas
#     for col in features:
#         if col not in df_new.columns:
#             df_new[col] = np.nan
#     df_new = df_new[features]
#
#     X_new = scaler.transform(imputer.transform(df_new))
#     if pca:
#         X_new = pca.transform(X_new)
#
#     scores_new = -model.decision_function(X_new)
#     belongs = scores_new <= threshold
#     perc_per_work = 100 * (1 - (scores_new - scores_new.min()) /
#                            (threshold - scores_new.min() + 1e-9))
#     perc_per_work = np.clip(perc_per_work, 0, 100)
#     global_membership = 100 * belongs.mean()
#
#     print("Resultados individuales:\n")
#     for i, obra in enumerate(df_new.index):
#         print(f"{os.path.basename(obra):40s} -> {perc_per_work[i]:6.2f}% pertenencia {'✓' if belongs[i] else '✗'}")
#
#     print("\n--------------------------------------------")
#     print(f"Porcentaje global de pertenencia del nuevo corpus: {global_membership:.2f}%")
#     print("--------------------------------------------\n")
#     return global_membership
