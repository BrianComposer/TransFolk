
############################################################################
# Histogramas comparativos de una o dos métricas musicales clave
# # (por ejemplo, rango melódico y entropía rítmica).
############################################################################
# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, note, chord, meter

# =========================================================
# 1. Extracción de rasgos globales
# =========================================================
def extract_features_for_histogram(path):
    """Extrae rango melódico y entropía rítmica de una obra MusicXML."""
    try:
        score = converter.parse(path)
    except Exception:
        return None

    flat = score.flat.notesAndRests
    pitches, durations = [], []

    for el in flat:
        dur = getattr(el, "quarterLength", 0.0)
        if el.isNote:
            pitches.append(float(el.pitch.midi))
            durations.append(float(dur))
        elif el.isRest:
            durations.append(float(dur))

    if not durations:
        return None

    # --- rango melódico ---
    if pitches:
        melodic_range = float(max(pitches) - min(pitches))
    else:
        melodic_range = 0.0

    # --- entropía rítmica ---
    def entropy(arr, bins=16):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        hist, _ = np.histogram(arr, bins=bins, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    rhythmic_entropy = entropy(durations, bins=16)
    return dict(melodic_range=melodic_range, rhythmic_entropy=rhythmic_entropy)


def load_features_from_dir(folder):
    """Carga las métricas de todas las obras de una carpeta."""
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.musicxml"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True))

    features = []
    for p in paths:
        f = extract_features_for_histogram(p)
        if f:
            features.append(f)
    if not features:
        raise RuntimeError(f"No se extrajeron rasgos válidos de {folder}")
    return features


# =========================================================
# 2. Función para graficar histogramas comparativos
# =========================================================
def comparative_histograms(corpus_dir, generated_dir, bins=20, save_path=None):
    """
    Genera histogramas comparativos (corpus vs generado)
    para rango melódico y entropía rítmica.
    """
    feats_corpus = load_features_from_dir(corpus_dir)
    feats_gen = load_features_from_dir(generated_dir)

    melodic_corpus = [f["melodic_range"] for f in feats_corpus]
    melodic_gen = [f["melodic_range"] for f in feats_gen]
    entropy_corpus = [f["rhythmic_entropy"] for f in feats_corpus]
    entropy_gen = [f["rhythmic_entropy"] for f in feats_gen]

    # --- CONFIGURACIÓN DE ESTILO ---
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"corpus": "#1b9e77", "generated": "#d95f02"}

    # --- Histograma 1: Rango melódico ---
    ax = axes[0]
    ax.hist(melodic_corpus, bins=bins, color=colors["corpus"], alpha=0.6, label="corpus", density=True)
    ax.hist(melodic_gen, bins=bins, color=colors["generated"], alpha=0.6, label="generated", density=True)
    ax.set_title("Melodic Range Distribution", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Range (MIDI pitch span)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Histograma 2: Entropía rítmica ---
    ax = axes[1]
    ax.hist(entropy_corpus, bins=bins, color=colors["corpus"], alpha=0.6, label="corpus", density=True)
    ax.hist(entropy_gen, bins=bins, color=colors["generated"], alpha=0.6, label="generated", density=True)
    ax.set_title("Rhythmic Entropy Distribution", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Rhythmic entropy (bits)", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')

    plt.suptitle("Comparative histograms of musical features (corpus vs generated)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfica guardada en: {save_path}")
    plt.show()

