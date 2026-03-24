

# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, note, chord, meter

# =========================================================
# 1. Extracción de métricas musicales estándar
# =========================================================
def safe_entropy(values, bins=16):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist))) if hist.size else 0.0


def extract_features_musicxml(path):
    """Extrae métricas globales típicas para análisis comparativo."""
    try:
        score = converter.parse(path)
    except Exception:
        return None

    flat = score.flat.notesAndRests
    pitches, durations, onsets = [], [], []

    onset = 0.0
    for el in flat:
        dur = float(getattr(el, "quarterLength", 0.0))
        if el.isNote:
            pitches.append(float(el.pitch.midi))
            durations.append(dur)
            onsets.append(onset)
        elif el.isChord:
            pitches.extend([float(p.midi) for p in el.pitches])
            durations.append(dur)
            onsets.append(onset)
        elif el.isRest:
            durations.append(dur)
        onset += dur

    if not durations:
        return None

    # --- Métricas ---
    mean_pitch = np.mean(pitches) if pitches else 60.0
    pitch_range = (max(pitches) - min(pitches)) if pitches else 0.0
    pitch_std = np.std(pitches) if pitches else 0.0

    mean_dur = np.mean(durations)
    dur_std = np.std(durations)
    rhythmic_entropy = safe_entropy(durations, bins=16)

    # densidad rítmica (número de eventos por unidad de tiempo)
    total_time = sum(durations)
    note_density = len(pitches) / total_time if total_time > 0 else 0.0

    # intervalo medio entre notas sucesivas
    mean_interval = np.mean(np.abs(np.diff(pitches))) if len(pitches) > 1 else 0.0

    # Entropía melódica (diversidad de alturas)
    melodic_entropy = safe_entropy(pitches, bins=24) if len(pitches) > 0 else 0.0

    return dict(
        mean_pitch=mean_pitch,
        pitch_range=pitch_range,
        pitch_std=pitch_std,
        mean_dur=mean_dur,
        dur_std=dur_std,
        rhythmic_entropy=rhythmic_entropy,
        note_density=note_density,
        mean_interval=mean_interval,
        melodic_entropy=melodic_entropy,
    )


def load_features_from_dir(folder):
    """Carga rasgos de todas las obras MusicXML de una carpeta."""
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.musicxml"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True))
    feats = []
    for p in paths:
        f = extract_features_musicxml(p)
        if f:
            feats.append(f)
    if not feats:
        raise RuntimeError(f"No se extrajeron rasgos válidos en: {folder}")
    return feats


# =========================================================
# 2. Histogramas comparativos de métricas múltiples
# =========================================================
def comparative_histograms_multimetric(charts_dir, corpus_dir, generated_dir, bins, CORPUS, ALGORITHM, TIME_SIGNATURE, TONALITY, TEMPERATURE, font_size = 14, axis_size=12, legend_size=10, show_tittle=True, show_chart=True):
    feats_corpus = load_features_from_dir(corpus_dir)
    feats_gen = load_features_from_dir(generated_dir)

    # --- Métricas a comparar (clave : etiqueta) ---
    metrics = {
        "pitch_range": "Melodic Range (MIDI span)",
        "rhythmic_entropy": "Rhythmic Entropy (bits)",
        "note_density": "Note Density (notes/beat)",
        "mean_interval": "Mean Interval (semitones)",
        "melodic_entropy": "Melodic Entropy (bits)",
        "dur_std": "Duration Variability (std)",
    }

    colors = {"corpus": "#1b9e77", "generated": "#d95f02"}
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'Arial'

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    axes = axes.flatten()

    for i, (key, label) in enumerate(metrics.items()):
        ax = axes[i]
        corpus_vals = [f[key] for f in feats_corpus if np.isfinite(f[key])]
        gen_vals = [f[key] for f in feats_gen if np.isfinite(f[key])]

        # Tamaño de los números de los ejes
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=axis_size
        )
        ax.hist(corpus_vals, bins=bins, color=colors["corpus"], alpha=0.6, label="corpus", density=True)
        ax.hist(gen_vals, bins=bins, color=colors["generated"], alpha=0.6, label="generated", density=True)
        ax.set_title(label, fontsize=font_size-2, fontweight='bold', pad=8)
        ax.set_xlabel("Value", fontsize=font_size)
        ax.set_ylabel("Density", fontsize=font_size)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(frameon=False, fontsize=legend_size)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('black')

    # Eliminar ejes vacíos si sobran
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if show_tittle:
        plt.suptitle(f"Comparative histograms of multiple musical features (corpus vs generated) ({CORPUS}, {ALGORITHM}, {TIME_SIGNATURE}, {TONALITY}, {TEMPERATURE})",
                     fontsize=font_size+2, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_png = f"{charts_dir}/CompHisto_{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}_{TEMPERATURE:.1f}.png"
    out_svg = f"{charts_dir}/CompHisto_{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}_{TEMPERATURE:.1f}.svg"

    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')

    # print(f"Gráfica guardada en: {save_path}")
    if show_chart:
        plt.show()
