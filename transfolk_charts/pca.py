import os, glob
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
from music21 import converter, meter


############################################################################
# PCA  del espacio de características del corpus + obras generadas.
############################################################################


# ---------------------------
# 1) Extracción de rasgos
# ---------------------------

def _safe_entropy_from_bins(values, bins=16):
    if len(values) == 0:
        return 0.0
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def extract_features_from_musicxml(path):
    """Rasgos globales simples y robustos (pitch/ritmo) para monodía o textura ligera."""
    try:
        score = converter.parse(path)
    except Exception:
        return None

    try:
        flat = score.flat.notesAndRests
    except Exception:
        flat = score.flatten().notesAndRests

    n_notes, n_rests = 0, 0
    durations = []
    pitches = []

    for el in flat:
        try:
            dur = float(getattr(el, "quarterLength", 0.0))
        except Exception:
            dur = 0.0

        if getattr(el, "isNote", False):
            n_notes += 1
            durations.append(dur)
            try:
                pitches.append(float(el.pitch.midi))
            except Exception:
                pass
        elif getattr(el, "isRest", False):
            n_rests += 1
            durations.append(dur)

    durs = np.asarray(durations, dtype=float)
    durs = durs[np.isfinite(durs)]
    if durs.size == 0:
        return None

    pm = np.asarray(pitches, dtype=float)
    pm = pm[np.isfinite(pm)]

    total_qL  = float(durs.sum())
    mean_dur  = float(np.mean(durs))
    std_dur   = float(np.std(durs))
    mean_pitch = float(np.mean(pm)) if pm.size else 60.0
    std_pitch  = float(np.std(pm)) if pm.size else 0.0
    range_pitch = float(pm.max() - pm.min()) if pm.size else 0.0
    rhythm_entropy = _safe_entropy_from_bins(durs, bins=16)

    # Densidad de notas por compás aprox (usa el primer compás encontrado o 4/4)
    try:
        ts = list(score.recurse().getElementsByClass(meter.TimeSignature))
        beats = float(ts[0].beatCount) if ts else 4.0
    except Exception:
        beats = 4.0
    notes_per_bar = _safe_div(n_notes, _safe_div(total_qL, beats))

    return np.array([
        float(n_notes),
        float(n_rests),
        total_qL,
        mean_dur,
        std_dur,
        mean_pitch,
        std_pitch,
        range_pitch,
        rhythm_entropy,
        notes_per_bar
    ], dtype=float)

FEATURE_NAMES = [
    "n_notes","n_rests","total_qL","mean_dur","std_dur",
    "mean_pitch","std_pitch","range_pitch","rhythm_entropy","notes_per_bar"
]

def load_features_from_dir(folder):
    """Devuelve (X, names) con X de forma (N, D)."""
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.musicxml"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True))

    X_list, names = [], []
    for p in paths:
        feats = extract_features_from_musicxml(p)
        if feats is not None and np.isfinite(feats).all():
            X_list.append(feats)
            names.append(os.path.basename(p))
    if not X_list:
        raise RuntimeError(f"No se extrajeron rasgos válidos en: {folder}")
    X = np.vstack(X_list)
    return X, names

import os
import numpy as np

def load_features_by_temperature(base_dir, temperature):
    """
    base_dir: productions/[ALGORITHM]
    temperature: str o float (ej. '0.8')

    Devuelve:
        X: np.ndarray (N, D)
        names: lista de nombres de obra
    """
    X_all = []
    names_all = []

    for ts in os.listdir(base_dir):
        ts_path = os.path.join(base_dir, ts)
        if not os.path.isdir(ts_path):
            continue

        for tonality in os.listdir(ts_path):
            ton_path = os.path.join(ts_path, tonality)
            if not os.path.isdir(ton_path):
                continue

            temp_path = os.path.join(ton_path, str(temperature))
            if not os.path.isdir(temp_path):
                continue

            X, names = load_features_from_dir(temp_path)
            if X is not None and len(X) > 0:
                X_all.append(X)
                names_all.extend(
                    [f"{ts}/{tonality}/{temperature}/{n}" for n in names]
                )

    if not X_all:
        raise RuntimeError(f"No features found for temperature={temperature}")

    return np.vstack(X_all), names_all




# ---------------------------
# 2) Estandarización y PCA (NumPy SVD)
# ---------------------------

def zscore(X, eps=1e-12):
    """Estandariza por columna: (X - media) / std. Maneja std=0."""
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu, sd

def pca_fit_transform_numpy(X, n_components=2):
    """
    PCA por SVD:
      Xc = X - mean
      Xc = U S Vt
      Proyección = U[:, :k] * S[:k]
      Var. explicada = S^2 / (n-1) / var_total
    Devuelve: Z (N, k), explained_ratio (k,), mean (D,)
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # SVD compacta
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    Z = U[:, :k] * S[:k]  # proyección 2D
    # var explicada
    eigenvalues = (S**2) / (X.shape[0] - 1)
    explained_ratio = eigenvalues[:k] / eigenvalues.sum()
    return Z, explained_ratio, mean

# ---------------------------
# 3) Visualización
# ---------------------------

def visualize_pca_numpy(charts_dir, corpus_dir, generated_dir, MODEL_NAME, CORPUS, ALGORITHM, TIME_SIGNATURE, TONALITY, TEMPERATURE, font_size = 14, axis_size=12, show_tittle=True, show_chart=True, by_temperature=True):
    # Carga rasgos
    Xc, names_c = load_features_from_dir(corpus_dir)
    if by_temperature:
        algo_dir = f"productions/{MODEL_NAME}/{CORPUS}/{ALGORITHM}"
        Xg, names_g = load_features_by_temperature(algo_dir, TEMPERATURE)
    else:
        Xg, names_g = load_features_from_dir(generated_dir)

    # Combina y estandariza (z-score sobre conjunto combinado)
    X_all = np.vstack([Xc, Xg])
    Xz, mu, sd = zscore(X_all)

    # PCA 2D
    Z, expl, _ = pca_fit_transform_numpy(Xz, n_components=2)

    # Split de vuelta
    n_c = Xc.shape[0]
    Zc = Z[:n_c, :]
    Zg = Z[n_c:, :]

    # Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(Zc[:, 0], Zc[:, 1], s=45, alpha=0.85, c="#1b9e77", edgecolor="black", linewidth=0.4, label="corpus")
    plt.scatter(Zg[:, 0], Zg[:, 1], s=45, alpha=0.85, c="#d95f02", edgecolor="black", linewidth=0.4, label="generated")

    title = f"PCA of the feature space of the corpus and generated works ({CORPUS}, {ALGORITHM}, {TIME_SIGNATURE}, {TONALITY}, {TEMPERATURE})"
    if show_tittle:
        plt.title(title, fontsize=14, fontweight='bold', pad=font_size+2)
    plt.xlabel(f"PC1 ({expl[0]*100:.1f}% var)", fontsize=font_size)
    plt.ylabel(f"PC2 ({expl[1]*100:.1f}% var)", fontsize=font_size)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=False, fontsize=font_size-2)
    plt.tight_layout()

    ax = plt.gca()

    # Tamaño de los números de los ejes
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=axis_size
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    out_png = f"{charts_dir}/PCA_{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}_{TEMPERATURE:.1f}.png"
    out_svg = f"{charts_dir}/PCA_{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}_{TEMPERATURE:.1f}.svg"

    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')

    if show_chart:
        plt.show()

