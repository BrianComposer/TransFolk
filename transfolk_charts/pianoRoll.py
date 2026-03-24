
# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, note, chord

# =========================================================
# 1. Conversión de una obra a mapa binario pitch–tiempo
# =========================================================
def pianoroll_matrix(path, pitch_min=36, pitch_max=84, step=0.25):
    """
    Devuelve una matriz binaria [n_pitches x n_steps] con 1 si la nota suena.
    step = resolución temporal en quarterLength (0.25 = semicorchea)
    """
    try:
        score = converter.parse(path)
    except Exception:
        return None

    flat = score.flat.notes
    onsets, offsets, pitches = [], [], []

    for el in flat:
        if isinstance(el, note.Note):
            start = float(el.offset)
            dur = float(el.quarterLength)
            onsets.append(start)
            offsets.append(start + dur)
            pitches.append(float(el.pitch.midi))
        elif isinstance(el, chord.Chord):
            for p in el.pitches:
                start = float(el.offset)
                dur = float(el.quarterLength)
                onsets.append(start)
                offsets.append(start + dur)
                pitches.append(float(p.midi))

    if not onsets:
        return None

    max_time = max(offsets)
    n_steps = int(np.ceil(max_time / step))
    n_pitches = pitch_max - pitch_min + 1
    roll = np.zeros((n_pitches, n_steps), dtype=float)

    for o1, o2, p in zip(onsets, offsets, pitches):
        if pitch_min <= p <= pitch_max:
            i = int(p - pitch_min)
            start_idx = int(o1 / step)
            end_idx = int(o2 / step)
            roll[i, start_idx:end_idx] = 1.0
    return roll


# =========================================================
# 2. Construcción de mapas agregados por carpeta
# =========================================================
def aggregate_pianoroll_density(folder, pitch_min=36, pitch_max=84, step=0.25):
    paths = sorted(glob.glob(os.path.join(folder, "*.musicxml"))) + \
            sorted(glob.glob(os.path.join(folder, "*.xml")))
    rolls = []
    for p in paths:
        mat = pianoroll_matrix(p, pitch_min, pitch_max, step)
        if mat is not None:
            rolls.append(mat)
    if not rolls:
        raise RuntimeError(f"No valid scores in {folder}")
    # Normalizar duración promedio
    min_len = min(r.shape[1] for r in rolls)
    rolls = [r[:, :min_len] for r in rolls]
    mean_roll = np.mean(np.stack(rolls, axis=0), axis=0)
    return mean_roll  # [pitches x time]


# =========================================================
# 3. Visualización comparativa
# =========================================================
def pianoroll_density_overlay(corpus_dir, generated_dir,
                              pitch_min=36, pitch_max=84, step=0.25,
                              save_path=None):
    roll_c = aggregate_pianoroll_density(corpus_dir, pitch_min, pitch_max, step)
    roll_g = aggregate_pianoroll_density(generated_dir, pitch_min, pitch_max, step)

    # Igualar dimensiones temporales mediante padding
    n_pitches = roll_c.shape[0]
    max_len = max(roll_c.shape[1], roll_g.shape[1])

    def pad_roll(roll, target_len):
        if roll.shape[1] < target_len:
            pad_width = target_len - roll.shape[1]
            roll = np.pad(roll, ((0, 0), (0, pad_width)), mode='constant')
        elif roll.shape[1] > target_len:
            roll = roll[:, :target_len]
        return roll

    roll_c = pad_roll(roll_c, max_len)
    roll_g = pad_roll(roll_g, max_len)

    n_steps = roll_c.shape[1]
    time_axis = np.arange(n_steps) * step
    pitch_axis = np.arange(pitch_min, pitch_max + 1)

    # Normalizar para visualización
    roll_c /= roll_c.max() if roll_c.max() > 0 else 1
    roll_g /= roll_g.max() if roll_g.max() > 0 else 1

    # Imagen RGB: verde = corpus, rojo = generado
    img = np.zeros((n_pitches, n_steps, 3))
    img[:, :, 1] = roll_c  # canal verde
    img[:, :, 0] = roll_g  # canal rojo

    plt.figure(figsize=(10, 6))
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    diff = roll_g - roll_c
    plt.imshow(np.flipud(diff), aspect="auto", cmap="bwr", interpolation="nearest",
               extent=[time_axis[0], time_axis[-1], pitch_min, pitch_max],
               origin='lower', vmin=-1, vmax=1)
    plt.title("Density Difference Map (generated - corpus)")

    plt.title("Aggregate Pianoroll Density Overlay\n(corpus vs generated)",
              fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Time (quarter lengths)", fontsize=12)
    plt.ylabel("MIDI Pitch", fontsize=12)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=4, label='Corpus'),
        plt.Line2D([0], [0], color='orange', lw=4, label='Generated')
    ], frameon=False, fontsize=11, loc="upper right")
    plt.grid(False)
    plt.tight_layout()

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved pianoroll overlay: {save_path}")
    plt.show()





