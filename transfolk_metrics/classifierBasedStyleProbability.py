import os
import numpy as np
import joblib
from pathlib import Path
from music21 import converter



# =========================================================
# UTILIDADES NECESARIAS
# =========================================================

def safe_numeric_array(values):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.array([0.0])
    return arr

def entropy(arr, bins):
    arr = safe_numeric_array(arr)
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


# =========================================================
# CARGA DE ARTEFACTOS
# =========================================================

def load_classifier(model_dir):
    model     = joblib.load(os.path.join(model_dir, "model.joblib"))
    imputer   = joblib.load(os.path.join(model_dir, "imputer.joblib"))
    scaler    = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    threshold = joblib.load(os.path.join(model_dir, "threshold.joblib"))
    features  = joblib.load(os.path.join(model_dir, "features.joblib"))
    pca_path  = os.path.join(model_dir, "pca.joblib")
    pca       = joblib.load(pca_path) if os.path.exists(pca_path) else None
    return model, imputer, scaler, pca, threshold, features


# =========================================================
# CARACTERÍSTICAS POR ARCHIVO
# =========================================================

def extract_features_single(path):
    score = converter.parse(path)
    flat = score.flat.notesAndRests

    n_notes, n_rests = 0, 0
    durations = []
    pitches = []

    for el in flat:
        dur = float(getattr(el, "quarterLength", 0.0))
        if el.isNote:
            n_notes += 1
            durations.append(dur)
            pitches.append(float(el.pitch.midi))
        elif el.isRest:
            n_rests += 1
            durations.append(dur)

    dur = safe_numeric_array(durations)
    pit = safe_numeric_array(pitches if pitches else [60.0])

    total_dur = float(np.sum(dur))

    ts = score.recurse().getElementsByClass("TimeSignature")
    beats = float(ts[0].beatCount) if ts else 4.0

    return dict(
        n_notes=n_notes,
        n_rests=n_rests,
        total_dur=total_dur,
        mean_dur=float(np.mean(dur)),
        std_dur=float(np.std(dur)),
        mean_pitch=float(np.mean(pit)),
        std_pitch=float(np.std(pit)),
        pitch_entropy=entropy([p % 12 for p in pit], bins=12),
        dur_entropy=entropy(dur, bins=8),
        notes_per_bar=(n_notes / (total_dur / beats)) if total_dur > 0 else 0.0,
    )


# =========================================================
# MÉTRICA S_folk
# =========================================================

def style_probability_folk(xml_folder, model_dir):
    model, imputer, scaler, pca, threshold, features = load_classifier(model_dir)

    paths = list(Path(xml_folder).glob("*.xml")) + list(Path(xml_folder).glob("*.musicxml"))
    if not paths:
        return 0.0, []

    feat_list = []
    for p in paths:
        f = extract_features_single(str(p))
        feat_list.append(f)

    X = []
    for f in feat_list:
        row = [f[col] if col in f else np.nan for col in features]
        X.append(row)
    X = np.array(X, dtype=float)

    X = imputer.transform(X)
    X = scaler.transform(X)
    if pca:
        X = pca.transform(X)

    scores = -model.decision_function(X)

    perc = 100 * (1 - (scores - scores.min()) / (threshold - scores.min() + 1e-9))
    perc = np.clip(perc, 0, 100)

    S_folk = float(np.mean(perc))

    return S_folk, perc.tolist()
