# -*- coding: utf-8 -*-

import os
import glob
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from music21 import converter, stream, note, chord



# =============================
# Config de features rítmicas
# =============================
SHORT_DUR_THRESHOLD_QL = 0.5   # <= corchea si negra=1.0
DUR_ROUND_FOR_ENTROPY = 0.25   # agrupar duraciones para entropía
STRONG_BEAT_THRESHOLD = 0.5    # beatStrength >= 0.5 => "fuerte" (proxy)
WEAK_BEAT_THRESHOLD = 0.25     # beatStrength <= 0.25 => "débil" (proxy)


# =============================
# Agrupación por categorías (paper)
# =============================
# FEATURE_GROUPS = {
#     "Ritmo": [
#         "note_density", "mean_dur", "cv_dur", "short_note_ratio",
#         "rhythmic_entropy", "npvi_rhythmic",
#     ],
#     "Métrica y síncopa": [
#         "strong_beat_note_ratio", "syncopation_index",
#     ],
#     "Pitch distribución": [
#         "reciting_pitch_ratio", "pitch_entropy",
#     ],
#     "Cadencia": [
#         "final_tonic_match", "last_note_duration_ratio",
#         "last_interval_abs", "last_interval_is_step",
#     ],
#     "Armonía y tonalidad": [
#         "diatonic_ratio", "best_corr", "key_clarity",
#         "best_key_pc", "best_mode_minor",
#     ],
#     "Intervalos": [
#         "mean_abs_semitones", "max_leap", "step_ratio", "unison_ratio",
#         "interval_2nd_ratio", "interval_3rd_ratio", "interval_4th_ratio",
#         "interval_5th_ratio", "interval_6th_ratio", "interval_7th_ratio",
#         "interval_octave_plus_ratio", "tritone_ratio", "consonant_interval_ratio",
#     ],
#     "Rango y propincuidad": [
#         "range_semitones", "range_p95", "proximity_step_le2", "proximity_inv_mean",
#     ],
#     "Contorno y dirección": [
#         "up_ratio", "down_ratio", "stay_ratio",
#         "num_direction_changes", "climax_pos",
#         "direction_balance", "pitch_time_slope",
#     ],
# }
FEATURE_GROUPS = {
    "Ritmo": [
        "note_density", "mean_dur", "cv_dur", "short_note_ratio",
        "rhythmic_entropy", "npvi_rhythmic",
        "rhythmic_energy"
    ],
    "Métrica y síncopa": [
        "strong_beat_note_ratio", "syncopation_index",
    ],
    "Pitch distribución": [
        "reciting_pitch_ratio", "pitch_entropy",
    ],
    "Cadencia": [
        "final_tonic_match", "last_note_duration_ratio",
        "last_interval_abs", "last_interval_is_step",
    ],
    "Armonía y tonalidad": [
        "diatonic_ratio", "best_corr", "key_clarity",
        "best_key_pc", "best_mode_minor",
    ],
    "Intervalos": [
        "mean_abs_semitones", "interval_std", "max_leap",
        "step_ratio", "unison_ratio",
        "interval_2nd_ratio", "interval_3rd_ratio", "interval_4th_ratio",
        "interval_5th_ratio", "interval_6th_ratio", "interval_7th_ratio",
        "interval_octave_plus_ratio", "tritone_ratio", "consonant_interval_ratio",
    ],
    "Rango y propincuidad": [
        "range_semitones", "range_p95", "range_relative",
        "proximity_step_le2", "proximity_inv_mean",
    ],
    "Contorno y dirección": [
        "up_ratio", "down_ratio", "stay_ratio",
        "num_direction_changes", "climax_pos",
        "direction_balance", "pitch_time_slope",
    ],
    "Estructura temporal": [
        "mean_ioi"
    ],
}

# FEATURE_TITLES = {
#     # Ritmo
#     "note_density": "Densidad de notas",
#     "mean_dur": "Duración media",
#     "cv_dur": "CV de duraciones",
#     "short_note_ratio": "% duraciones cortas",
#     "rhythmic_entropy": "Entropía rítmica",
#     "npvi_rhythmic": "nPVI rítmico",
#
#     # Métrica / síncopa
#     "strong_beat_note_ratio": "% ataques en pulso fuerte",
#     "syncopation_index": "Índice de síncopa (proxy)",
#
#     # Pitch
#     "reciting_pitch_ratio": "Proporción nota recitativa",
#     "pitch_entropy": "Entropía de alturas",
#
#     # Cadencia
#     "final_tonic_match": "Final en tónica",
#     "last_note_duration_ratio": "Proporción duración final",
#     "last_interval_abs": "Intervalo final absoluto",
#     "last_interval_is_step": "Cierre por paso (0/1)",
#
#     # Armonía / tonalidad
#     "diatonic_ratio": "Proporción diatónica",
#     "best_corr": "Correlación tonal (music21)",
#     "key_clarity": "Claridad tonal",
#     "best_key_pc": "Pitch class de tónica",
#     "best_mode_minor": "Modo menor (0/1)",
#
#     # Intervalos
#     "mean_abs_semitones": "Salto medio absoluto",
#     "max_leap": "Salto máximo",
#     "step_ratio": "% movimiento por paso",
#     "unison_ratio": "% unísonos",
#     "interval_2nd_ratio": "% segundas",
#     "interval_3rd_ratio": "% terceras",
#     "interval_4th_ratio": "% cuartas",
#     "interval_5th_ratio": "% quintas",
#     "interval_6th_ratio": "% sextas",
#     "interval_7th_ratio": "% séptimas",
#     "interval_octave_plus_ratio": "% octava o más",
#     "tritone_ratio": "% tritonos",
#     "consonant_interval_ratio": "% intervalos consonantes (proxy)",
#
#     # Rango / propincuidad
#     "range_semitones": "Rango (semitonos)",
#     "range_p95": "Rango robusto (P95-P05)",
#     "proximity_step_le2": "% saltos ≤ 2",
#     "proximity_inv_mean": "Propincuidad (1/(1+salto medio))",
#
#     # Contorno / dirección
#     "up_ratio": "% ascensos",
#     "down_ratio": "% descensos",
#     "stay_ratio": "% repetición",
#     "num_direction_changes": "Cambios de dirección",
#     "climax_pos": "Posición del clímax",
#     "direction_balance": "Balance direccional",
#     "pitch_time_slope": "Pendiente pitch~tiempo",
# }

FEATURE_TITLES = {

    # Ritmo
    "note_density": "Densidad de notas",
    "mean_dur": "Duración media",
    "cv_dur": "CV de duraciones",
    "short_note_ratio": "% duraciones cortas",
    "rhythmic_entropy": "Entropía rítmica",
    "npvi_rhythmic": "nPVI rítmico",
    "rhythmic_energy": "Energía rítmica",

    # Métrica / síncopa
    "strong_beat_note_ratio": "% ataques en pulso fuerte",
    "syncopation_index": "Índice de síncopa (proxy)",

    # Pitch
    "reciting_pitch_ratio": "Proporción nota recitativa",
    "pitch_entropy": "Entropía de alturas",

    # Cadencia
    "final_tonic_match": "Final en tónica",
    "last_note_duration_ratio": "Proporción duración final",
    "last_interval_abs": "Intervalo final absoluto",
    "last_interval_is_step": "Cierre por paso (0/1)",

    # Armonía / tonalidad
    "diatonic_ratio": "Proporción diatónica",
    "best_corr": "Correlación tonal (music21)",
    "key_clarity": "Claridad tonal",
    "best_key_pc": "Pitch class de tónica",
    "best_mode_minor": "Modo menor (0/1)",

    # Intervalos
    "mean_abs_semitones": "Salto medio absoluto",
    "interval_std": "Desviación interválica",
    "max_leap": "Salto máximo",
    "step_ratio": "% movimiento por paso",
    "unison_ratio": "% unísonos",
    "interval_2nd_ratio": "% segundas",
    "interval_3rd_ratio": "% terceras",
    "interval_4th_ratio": "% cuartas",
    "interval_5th_ratio": "% quintas",
    "interval_6th_ratio": "% sextas",
    "interval_7th_ratio": "% séptimas",
    "interval_octave_plus_ratio": "% octava o más",
    "tritone_ratio": "% tritonos",
    "consonant_interval_ratio": "% intervalos consonantes (proxy)",

    # Rango
    "range_semitones": "Rango (semitonos)",
    "range_p95": "Rango robusto (P95-P05)",
    "range_relative": "Rango relativo",
    "proximity_step_le2": "% saltos ≤ 2",
    "proximity_inv_mean": "Propincuidad (1/(1+salto medio))",

    # Contorno
    "up_ratio": "% ascensos",
    "down_ratio": "% descensos",
    "stay_ratio": "% repetición",
    "num_direction_changes": "Cambios de dirección",
    "climax_pos": "Posición del clímax",
    "direction_balance": "Balance direccional",
    "pitch_time_slope": "Pendiente pitch~tiempo",

    # Forma
    "mean_ioi": "IOI medio",
}



# -----------------------------
# File discovery / parsing
# -----------------------------
def _is_musicxml(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".xml", ".musicxml", ".mxl"]


def find_musicxml_files(root: str) -> List[str]:
    patterns = ["**/*.musicxml", "**/*.xml", "**/*.mxl"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return [f for f in sorted(set(files)) if _is_musicxml(f)]


def safe_parse_musicxml(path: str):
    try:
        return converter.parse(path)
    except Exception as e:
        print(f"[WARN] No pude parsear: {path}\n       {type(e).__name__}: {e}")
        return None


def pick_melodic_part(score) -> Optional[stream.Part]:
    if not hasattr(score, "parts") or len(score.parts) == 0:
        return None

    best_part = None
    best_count = -1

    for p in score.parts:
        try:
            elems = list(p.flat.notesAndRests)
            n_notes = sum(1 for x in elems if isinstance(x, (note.Note, chord.Chord)))
            if n_notes > best_count:
                best_count = n_notes
                best_part = p
        except Exception:
            continue

    return best_part


def flatten_melody_events(part: stream.Part) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve arrays (sin rests):
      onsets_ql, durations_ql, midis, beatStrengths
    """
    try:
        p2 = part.stripTies(inPlace=False)
    except Exception:
        p2 = part

    flat = p2.flat
    onsets, durs, midis, bstr = [], [], [], []

    for el in flat.notesAndRests:
        dur = float(getattr(el.duration, "quarterLength", 0.0) or 0.0)
        if dur <= 0:
            continue

        if isinstance(el, note.Rest):
            continue

        if isinstance(el, note.Note):
            m = int(el.pitch.midi)
        elif isinstance(el, chord.Chord):
            m = int(max(p.midi for p in el.pitches))
        else:
            continue

        try:
            bs = float(getattr(el, "beatStrength", np.nan))
        except Exception:
            bs = float("nan")

        onsets.append(float(el.offset))
        durs.append(dur)
        midis.append(m)
        bstr.append(bs)

    if len(midis) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    idx = np.argsort(onsets)
    return (
        np.array(onsets, dtype=float)[idx],
        np.array(durs, dtype=float)[idx],
        np.array(midis, dtype=int)[idx],
        np.array(bstr, dtype=float)[idx],
    )


# -----------------------------
# Utility stats
# -----------------------------
def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else 0.0


def safe_max(x: np.ndarray) -> float:
    return float(np.max(x)) if len(x) else 0.0


def safe_entropy(prob: np.ndarray, base: float = 2.0) -> float:
    p = prob.astype(float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def count_direction_changes(deltas: np.ndarray) -> int:
    if len(deltas) == 0:
        return 0
    s = np.sign(deltas)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def linear_slope_pitch_time(onsets: np.ndarray, midis: np.ndarray) -> float:
    if len(midis) < 2:
        return 0.0
    x = onsets.astype(float)
    y = midis.astype(float)
    if np.std(x) < 1e-12:
        return 0.0
    a, _b = np.polyfit(x, y, 1)
    return float(a)


def npvi(durations: np.ndarray) -> float:
    d = durations.astype(float)
    if len(d) < 2:
        return 0.0
    d1 = d[:-1]
    d2 = d[1:]
    denom = (d1 + d2) / 2.0
    mask = denom > 1e-12
    if not np.any(mask):
        return 0.0
    vals = np.abs(d1[mask] - d2[mask]) / denom[mask]
    return float(100.0 * np.mean(vals))


#########################
def interval_variance_feature(deltas: np.ndarray) -> Dict[str, float]:
    if len(deltas) == 0:
        return {"interval_std": 0.0}

    abs_d = np.abs(deltas).astype(float)

    return {
        "interval_std": float(np.std(abs_d))
    }

def relative_range_feature(midis: np.ndarray) -> Dict[str, float]:
    if len(midis) == 0:
        return {"range_relative": 0.0}

    r = float(np.max(midis) - np.min(midis))
    mean_pitch = float(np.mean(midis))

    if mean_pitch <= 0:
        return {"range_relative": 0.0}

    return {
        "range_relative": float(r / mean_pitch)
    }

def phrase_proxy_features(onsets: np.ndarray) -> Dict[str, float]:
    if len(onsets) < 2:
        return {"mean_ioi": 0.0}

    iois = np.diff(onsets)

    return {
        "mean_ioi": float(np.mean(iois))
    }

def rhythmic_energy_feature(note_density: float, mean_dur: float) -> Dict[str, float]:

    if mean_dur <= 0:
        energy = 0.0
    else:
        energy = note_density * (1.0 / mean_dur)

    return {
        "rhythmic_energy": float(energy)
    }

#########################




# -----------------------------
# Key detection with music21
# -----------------------------
def estimate_key_music21(melodic_stream: stream.Stream, midis: np.ndarray, durs: np.ndarray) -> Dict[str, float]:
    try:
        k = melodic_stream.analyze("key")
    except Exception:
        return {
            "best_key_pc": float("nan"),
            "best_mode_minor": float("nan"),
            "best_corr": float("nan"),
            "key_clarity": float("nan"),
            "diatonic_ratio": float("nan"),
        }

    try:
        tonic_pc = float(k.tonic.pitchClass)
    except Exception:
        tonic_pc = float("nan")

    mode_minor = float(1.0 if getattr(k, "mode", "").lower() == "minor" else 0.0)

    best_corr = getattr(k, "correlationCoefficient", float("nan"))
    try:
        best_corr = float(best_corr)
    except Exception:
        best_corr = float("nan")

    key_clarity = float("nan")
    alts = getattr(k, "alternateInterpretations", None)
    if alts and isinstance(alts, (list, tuple)) and len(alts) >= 2:
        try:
            c1 = float(getattr(alts[0], "correlationCoefficient", float("nan")))
            c2 = float(getattr(alts[1], "correlationCoefficient", float("nan")))
            if np.isfinite(c1) and np.isfinite(c2):
                key_clarity = float(c1 - c2)
        except Exception:
            pass

    diatonic_ratio = float("nan")
    try:
        scale = k.getScale()
        scale_pcs = {p.pitchClass for p in scale.pitches}
        pcs = np.mod(midis, 12).astype(int)
        w = durs.astype(float)
        total_w = float(np.sum(w))
        if total_w > 0:
            diat_w = float(np.sum([ww for pc, ww in zip(pcs, w) if int(pc) in scale_pcs]))
            diatonic_ratio = float(diat_w / total_w)
    except Exception:
        pass

    return {
        "best_key_pc": tonic_pc,
        "best_mode_minor": mode_minor,
        "best_corr": best_corr,
        "key_clarity": key_clarity,
        "diatonic_ratio": diatonic_ratio,
    }


# -----------------------------
# Feature blocks
# -----------------------------
def rhythm_features(onsets: np.ndarray, durs: np.ndarray, beatStrengths: np.ndarray) -> Dict[str, float]:
    if len(durs) == 0:
        return {
            "note_density": 0.0,
            "mean_dur": 0.0,
            "cv_dur": 0.0,
            "short_note_ratio": 0.0,
            "rhythmic_entropy": 0.0,
            "npvi_rhythmic": 0.0,
            "strong_beat_note_ratio": float("nan"),
            "syncopation_index": float("nan"),
        }

    start = float(np.min(onsets))
    end = float(np.max(onsets + durs))
    span = max(1e-9, end - start)
    note_density = float(len(durs) / span)

    mean_dur = float(np.mean(durs))
    std_dur = float(np.std(durs))
    cv_dur = float(std_dur / mean_dur) if mean_dur > 1e-12 else 0.0

    short_note_ratio = float(np.mean(durs <= SHORT_DUR_THRESHOLD_QL))

    d_round = np.round(durs / DUR_ROUND_FOR_ENTROPY) * DUR_ROUND_FOR_ENTROPY
    vals, counts = np.unique(d_round, return_counts=True)
    prob = counts.astype(float) / float(np.sum(counts))
    rhythmic_entropy = safe_entropy(prob, base=2.0)

    npvi_rhythmic = npvi(durs)

    # Métrica / síncopa
    finite_bs = beatStrengths[np.isfinite(beatStrengths)]
    if len(finite_bs) == 0:
        strong_ratio = float("nan")
        syncop = float("nan")
    else:
        strong_ratio = float(np.mean(beatStrengths >= STRONG_BEAT_THRESHOLD))

        weak_mask = (beatStrengths <= WEAK_BEAT_THRESHOLD)
        longish_mask = (durs >= SHORT_DUR_THRESHOLD_QL)
        mask = weak_mask & longish_mask & np.isfinite(beatStrengths)

        if np.any(mask):
            weakness = (WEAK_BEAT_THRESHOLD - beatStrengths[mask]) / max(WEAK_BEAT_THRESHOLD, 1e-6)
            syncop = float(np.sum(weakness * durs[mask]) / np.sum(durs))
        else:
            syncop = 0.0

    return {
        "note_density": note_density,
        "mean_dur": mean_dur,
        "cv_dur": cv_dur,
        "short_note_ratio": short_note_ratio,
        "rhythmic_entropy": rhythmic_entropy,
        "npvi_rhythmic": npvi_rhythmic,
        "strong_beat_note_ratio": strong_ratio,
        "syncopation_index": syncop,
    }


def pitch_distribution_features(midis: np.ndarray, durs: np.ndarray) -> Dict[str, float]:
    if len(midis) == 0 or len(durs) == 0 or np.sum(durs) <= 0:
        return {"reciting_pitch_ratio": 0.0, "pitch_entropy": 0.0}

    total = float(np.sum(durs))
    uniq = {}
    for m, w in zip(midis, durs):
        uniq[int(m)] = uniq.get(int(m), 0.0) + float(w)

    weights = np.array(list(uniq.values()), dtype=float)
    prob = weights / np.sum(weights)
    reciting_pitch_ratio = float(np.max(prob))
    pitch_entropy = safe_entropy(prob, base=2.0)

    return {"reciting_pitch_ratio": reciting_pitch_ratio, "pitch_entropy": pitch_entropy}


def cadential_features(midis: np.ndarray, durs: np.ndarray, tonic_pc: float) -> Dict[str, float]:
    if len(midis) == 0 or len(durs) == 0 or np.sum(durs) <= 0:
        return {
            "final_tonic_match": float("nan"),
            "last_note_duration_ratio": float("nan"),
            "last_interval_abs": float("nan"),
            "last_interval_is_step": float("nan"),
        }

    last_pc = int(midis[-1] % 12)
    if np.isfinite(tonic_pc):
        final_tonic_match = float(1.0 if last_pc == int(tonic_pc) else 0.0)
    else:
        final_tonic_match = float("nan")

    last_note_duration_ratio = float(durs[-1] / np.sum(durs))

    last_interval_abs = float(abs(int(midis[-1]) - int(midis[-2])))
    last_interval_is_step = float(1.0 if last_interval_abs in (1, 2) else 0.0)

    return {
        "final_tonic_match": final_tonic_match,
        "last_note_duration_ratio": last_note_duration_ratio,
        "last_interval_abs": last_interval_abs,
        "last_interval_is_step": last_interval_is_step,
    }


def interval_class_features(deltas: np.ndarray) -> Dict[str, float]:
    if len(deltas) == 0:
        return {k: 0.0 for k in [
            "interval_2nd_ratio", "interval_3rd_ratio", "interval_4th_ratio", "interval_5th_ratio",
            "interval_6th_ratio", "interval_7th_ratio", "interval_octave_plus_ratio", "tritone_ratio",
            "consonant_interval_ratio"
        ]}

    a = np.abs(deltas).astype(int)

    i2 = np.isin(a, [1, 2])
    i3 = np.isin(a, [3, 4])
    i4 = (a == 5)
    tri = (a == 6)
    i5 = (a == 7)
    i6 = np.isin(a, [8, 9])
    i7 = np.isin(a, [10, 11])
    i8p = (a >= 12)

    consonant = (a == 0) | i3 | i5 | i6 | i8p | i4

    n = float(len(a))
    return {
        "interval_2nd_ratio": float(np.sum(i2) / n),
        "interval_3rd_ratio": float(np.sum(i3) / n),
        "interval_4th_ratio": float(np.sum(i4) / n),
        "interval_5th_ratio": float(np.sum(i5) / n),
        "interval_6th_ratio": float(np.sum(i6) / n),
        "interval_7th_ratio": float(np.sum(i7) / n),
        "interval_octave_plus_ratio": float(np.sum(i8p) / n),
        "tritone_ratio": float(np.sum(tri) / n),
        "consonant_interval_ratio": float(np.sum(consonant) / n),
    }


# -----------------------------
# Feature extraction per file
# -----------------------------
def extract_features_musicxml(path: str) -> Optional[Dict[str, float]]:
    score = safe_parse_musicxml(path)
    if score is None:
        return None

    part = pick_melodic_part(score)
    if part is None:
        return None

    onsets, durs, midis, beatStrengths = flatten_melody_events(part)
    if len(midis) < 2:
        return None

    deltas = np.diff(midis).astype(int)
    abs_d = np.abs(deltas)

    tonal = estimate_key_music21(part.flat, midis, durs)
    cad = cadential_features(midis, durs, tonic_pc=tonal["best_key_pc"])
    pdist = pitch_distribution_features(midis, durs)
    rhy = rhythm_features(onsets, durs, beatStrengths)
    iclass = interval_class_features(deltas)

    mean_abs_semitones = safe_mean(abs_d)
    max_leap = safe_max(abs_d)
    step_ratio = float(np.mean(np.isin(abs_d, [1, 2])))
    unison_ratio = float(np.mean(abs_d == 0))

    range_semitones = float(np.max(midis) - np.min(midis))
    range_p95 = float(np.percentile(midis, 95) - np.percentile(midis, 5))

    proximity_step_le2 = float(np.mean(abs_d <= 2))
    proximity_inv_mean = float(1.0 / (1.0 + mean_abs_semitones)) if mean_abs_semitones >= 0 else 0.0

    up_ratio = float(np.mean(deltas > 0))
    down_ratio = float(np.mean(deltas < 0))
    stay_ratio = float(np.mean(deltas == 0))
    num_dir_changes = float(count_direction_changes(deltas))
    climax_pos = float(int(np.argmax(midis)) / (len(midis) - 1)) if len(midis) > 1 else 0.0

    direction_balance = float(np.mean(np.sign(deltas)))
    pitch_time_slope = linear_slope_pitch_time(onsets, midis)

    ######
    ivar = interval_variance_feature(deltas)
    rrange = relative_range_feature(midis)
    phr = phrase_proxy_features(onsets)
    renergy = rhythmic_energy_feature(rhy["note_density"], rhy["mean_dur"])
    ######

    return {
        # Ritmo
        "note_density": rhy["note_density"],
        "mean_dur": rhy["mean_dur"],
        "cv_dur": rhy["cv_dur"],
        "short_note_ratio": rhy["short_note_ratio"],
        "rhythmic_entropy": rhy["rhythmic_entropy"],
        "npvi_rhythmic": rhy["npvi_rhythmic"],

        # Métrica / síncopa
        "strong_beat_note_ratio": rhy["strong_beat_note_ratio"],
        "syncopation_index": rhy["syncopation_index"],

        # Pitch (distribución)
        "reciting_pitch_ratio": pdist["reciting_pitch_ratio"],
        "pitch_entropy": pdist["pitch_entropy"],

        # Cadencia
        "final_tonic_match": cad["final_tonic_match"],
        "last_note_duration_ratio": cad["last_note_duration_ratio"],
        "last_interval_abs": cad["last_interval_abs"],
        "last_interval_is_step": cad["last_interval_is_step"],

        # Armonía / tonalidad
        "diatonic_ratio": tonal["diatonic_ratio"],
        "best_corr": tonal["best_corr"],
        "key_clarity": tonal["key_clarity"],
        "best_key_pc": tonal["best_key_pc"],
        "best_mode_minor": tonal["best_mode_minor"],

        # Intervalos cuantitativos
        "mean_abs_semitones": mean_abs_semitones,
        "max_leap": max_leap,
        "step_ratio": step_ratio,
        "unison_ratio": unison_ratio,

        # Intervalos cualitativos (proxy)
        **iclass,

        # Rango / propincuidad
        "range_semitones": range_semitones,
        "range_p95": range_p95,
        "proximity_step_le2": proximity_step_le2,
        "proximity_inv_mean": proximity_inv_mean,

        # Contorno / dirección
        "up_ratio": up_ratio,
        "down_ratio": down_ratio,
        "stay_ratio": stay_ratio,
        "num_direction_changes": num_dir_changes,
        "climax_pos": climax_pos,
        "direction_balance": direction_balance,
        "pitch_time_slope": pitch_time_slope,
        #**ivar,
        #**rrange,
        #**phr,
        #**renergy,
    }


def load_corpus_features(corpus_dir: str, label: str) -> pd.DataFrame:
    files = find_musicxml_files(corpus_dir)
    if not files:
        raise RuntimeError(f"No encontré MusicXML en: {corpus_dir}")

    rows = []
    for p in files:
        feats = extract_features_musicxml(p)
        if feats is None:
            continue
        feats["file"] = os.path.relpath(p, corpus_dir)
        feats["corpus"] = label
        rows.append(feats)

    if not rows:
        raise RuntimeError(f"No se pudieron extraer rasgos válidos en: {corpus_dir}")

    return pd.DataFrame(rows)