# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from music21 import converter, note, chord

# =========================================================
# 1. Extracción de características básicas
# =========================================================
def extract_features(path):
    """Extrae un vector simple de rasgos melódico–rítmicos globales."""
    try:
        score = converter.parse(path)
    except Exception:
        return None

    flat = score.flat.notesAndRests
    pitches, durations = [], []

    for el in flat:
        if el.isNote:
            pitches.append(el.pitch.midi)
            durations.append(float(el.quarterLength))
        elif el.isChord:
            pitches.extend([p.midi for p in el.pitches])
            durations.append(float(el.quarterLength))
        elif el.isRest:
            durations.append(float(el.quarterLength))

    if len(durations) == 0:
        return None

    mean_pitch = np.mean(pitches) if pitches else 60.0
    pitch_std = np.std(pitches) if pitches else 0.0
    pitch_range = max(pitches) - min(pitches) if pitches else 0.0
    mean_dur = np.mean(durations)
    dur_std = np.std(durations)

    return np.array([mean_pitch, pitch_std, pitch_range, mean_dur, dur_std], dtype=float)


def load_features_from_dir(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.musicxml"))) + \
            sorted(glob.glob(os.path.join(folder, "*.xml")))
    feats = []
    for p in paths:
        f = extract_features(p)
        if f is not None and np.isfinite(f).all():
            feats.append(f)
    return np.vstack(feats) if feats else None


# =========================================================
# 2. PCA manual (NumPy)
# =========================================================
def zscore(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def pca_numpy(X, n_components=2):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n_components] * S[:n_components]
    return Z, Vt[:n_components]


# =========================================================
# 3. Kernel Density Heatmap (corpus vs generated)
# =========================================================
def kernel_density_heatmap(charts_dir, corpus_dir, generated_dir, gridsize, CORPUS, ALGORITHM, TIME_SIGNATURE, TONALITY, TEMPERATURE, font_size = 14, axis_size=12, show_tittle=True, show_chart=True):
    Xc = load_features_from_dir(corpus_dir)
    Xg = load_features_from_dir(generated_dir)
    if Xc is None or Xg is None:
        raise RuntimeError("No se pudieron cargar características válidas.")

    X_all = np.vstack([Xc, Xg])
    Xz, _, _ = zscore(X_all)
    Z, _ = pca_numpy(Xz, n_components=2)

    n_c = Xc.shape[0]
    Zc = Z[:n_c]
    Zg = Z[n_c:]

    # KDE
    kde_c = gaussian_kde(Zc.T, bw_method=0.3)
    kde_g = gaussian_kde(Zg.T, bw_method=0.3)

    xmin, xmax = Z[:, 0].min(), Z[:, 0].max()
    ymin, ymax = Z[:, 1].min(), Z[:, 1].max()
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, gridsize),
                               np.linspace(ymin, ymax, gridsize))
    grid_coords = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

    Zc_dens = kde_c(grid_coords).reshape(Xgrid.shape)
    Zg_dens = kde_g(grid_coords).reshape(Xgrid.shape)

    # --- Plot ---
    plt.figure(figsize=(7, 6))

    # Corpus heatmap
    plt.contourf(Xgrid, Ygrid, Zc_dens, levels=30, cmap="Greens", alpha=0.6)
    plt.contour(Xgrid, Ygrid, Zc_dens, levels=8, colors='green', linewidths=0.6)

    # Generated heatmap
    plt.contourf(Xgrid, Ygrid, Zg_dens, levels=30, cmap="Oranges", alpha=0.6)
    plt.contour(Xgrid, Ygrid, Zg_dens, levels=8, colors='darkorange', linewidths=0.6)

    # Etiquetas
    if show_tittle:
        plt.title(f"Kernel Density Heatmaps in PCA feature space\n(corpus vs generated) ({CORPUS}, {ALGORITHM}, {TIME_SIGNATURE}, {TONALITY}, {TEMPERATURE})",
                  fontsize=font_size+2, fontweight='bold', pad=12)
    plt.xlabel("PC1", fontsize=font_size)
    plt.ylabel("PC2", fontsize=font_size)

    # Leyenda manual
    plt.scatter([], [], color='green', alpha=0.7, label='corpus')
    plt.scatter([], [], color='darkorange', alpha=0.7, label='generated')
    plt.legend(frameon=False, fontsize=font_size-2)

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
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


    out_png = f"{charts_dir}/KernelDensity_{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}_{TEMPERATURE:.1f}.png"
    out_svg = f"{charts_dir}/KernelDensity_{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}_{TEMPERATURE:.1f}.svg"

    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')

    if show_chart:
        plt.show()




