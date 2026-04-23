# -*- coding: utf-8 -*-

import os
import glob

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


from music21 import converter, stream, note, chord, meter


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
        "rhythmic_energy", "triplet_in_binary_ratio", "dotted_rhythm_ratio",
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
        "initial_interval_is_6th",
        "final_leading_tone_appoggiatura",
        "strong_weak_semitone_resolution_ratio",
    ],
    "Armonía y tonalidad": [
        "diatonic_ratio", "best_corr", "key_clarity",
        "best_key_pc", "best_mode_minor",
        "chromatic_usage_ratio",
        "retardo_la_solsharp_ratio",
        "retardo_si_la_ratio",
        "minor_leading_to_mediant_ratio",
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
    "Ornamentación": [
        "grace_note_ratio",
        "short_ornament_window_ratio",
        "turn_like_ratio",
        "appoggiatura_like_ratio",
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
    # Ornamentación
    "grace_note_ratio": "% notas de gracia",
    "short_ornament_window_ratio": "% ventanas ornamentales breves",
    "turn_like_ratio": "% grupetos / floreos",
    "appoggiatura_like_ratio": "% apoyaturas ornamentales",

    # Ritmos específicos
    "triplet_in_binary_ratio": "% tresillos en entorno binario",
    "dotted_rhythm_ratio": "% ritmos apuntillados",

    # Rasgos melódico-cadenciales
    "initial_interval_is_6th": "Inicio con intervalo de sexta (0/1)",
    "final_leading_tone_appoggiatura": "Apoyatura final de sensible (0/1)",
    "strong_weak_semitone_resolution_ratio": "% fuerte-débil con resolución por semitono",

    # Cromatismo / giros idiomáticos
    "chromatic_usage_ratio": "% uso de cromatismos",
    "retardo_la_solsharp_ratio": "% retardos tipo la/la-sol#",
    "retardo_si_la_ratio": "% retardos tipo si/si-la",
    "minor_leading_to_mediant_ratio": "% giro sensible–mediante en menor",
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
# Helpers for idiomatic / local features
# -----------------------------
def _safe_pitch_midi(el):
    if isinstance(el, note.Note):
        return int(el.pitch.midi)
    elif isinstance(el, chord.Chord):
        return int(max(p.midi for p in el.pitches))
    return None


# def _safe_pitch_pc(el):
#     m = _safe_pitch_midi(el)
#     return None if m is None else int(m % 12)


def _safe_dur_ql(el) -> float:
    try:
        return float(getattr(el.duration, "quarterLength", 0.0) or 0.0)
    except Exception:
        return 0.0


def _safe_beat_strength(el) -> float:
    try:
        return float(getattr(el, "beatStrength", np.nan))
    except Exception:
        return float("nan")


def _iter_note_like_events(part: stream.Part):
    """
    Devuelve lista de dicts con notas/acordes (sin rests), incluyendo grace notes
    cuando existan en el MusicXML.
    """
    events = []
    flat = part.flat

    for el in flat.notesAndRests:
        if isinstance(el, note.Rest):
            continue
        if not isinstance(el, (note.Note, chord.Chord)):
            continue

        midi = _safe_pitch_midi(el)
        if midi is None:
            continue

        dur = _safe_dur_ql(el)

        is_grace = False
        try:
            is_grace = bool(getattr(el.duration, "isGrace", False))
        except Exception:
            pass

        # En algunas exportaciones las grace van con ql=0
        events.append({
            "el": el,
            "offset": float(getattr(el, "offset", 0.0) or 0.0),
            "dur": dur,
            "midi": int(midi),
            "pc": int(midi % 12),
            "beatStrength": _safe_beat_strength(el),
            "is_grace": is_grace or dur == 0.0,
        })

    events.sort(key=lambda x: x["offset"])
    return events


def _get_measure_time_signature(meas) -> Optional[meter.TimeSignature]:
    try:
        ts = meas.timeSignature
        if ts is not None:
            return ts
    except Exception:
        pass

    try:
        ctx = meas.getContextByClass(meter.TimeSignature)
        if ctx is not None:
            return ctx
    except Exception:
        pass

    return None


# def _is_binary_metric(ts: Optional[meter.TimeSignature]) -> bool:
#     """
#     Proxy sencillo para 'entorno binario':
#     - 2/4, 4/4, 2/2, 4/2, etc.
#     - excluye compases compuestos típicos 6/8, 9/8, 12/8
#     """
#     if ts is None:
#         return False
#     try:
#         num = int(ts.numerator)
#         den = int(ts.denominator)
#     except Exception:
#         return False
#
#     if den not in (2, 4, 8, 16):
#         return False
#
#     if num in (6, 9, 12):
#         return False
#
#     return True
def _is_binary_metric(ts: Optional[meter.TimeSignature]) -> bool:
    """
    Proxy sencillo para 'entorno binario':
    acepta compases de subdivisión binaria y pulso binario/simple,
    excluyendo ternarios simples y compuestos típicos.
    """
    if ts is None:
        return False
    try:
        num = int(ts.numerator)
        den = int(ts.denominator)
    except Exception:
        return False

    # excluye ternarios simples y compuestos típicos
    if (num, den) in [(3, 4), (3, 8), (6, 8), (9, 8), (12, 8)]:
        return False

    # acepta contextos típicamente binarios
    if (num, den) in [(2, 2), (2, 4), (4, 4), (2, 8), (4, 8), (2, 16), (4, 16)]:
        return True

    # fallback razonable
    return num % 2 == 0 and num not in (6, 12)


# -----------------------------
# Ornamentation features
# -----------------------------
def ornamentation_features(part: stream.Part) -> Dict[str, float]:
    """
    Proxies para:
    - notas de gracia
    - floreos / adornos breves locales
    - grupetos
    - apoyaturas ornamentales explícitas
    """
    events = _iter_note_like_events(part)
    note_events = [e for e in events if e["midi"] is not None]

    if len(note_events) < 2:
        return {
            "grace_note_ratio": 0.0,
            "short_ornament_window_ratio": 0.0,
            "turn_like_ratio": 0.0,
            "appoggiatura_like_ratio": 0.0,
        }

    n = float(len(note_events))

    # 1) notas de gracia
    grace_count = sum(1 for e in note_events if e["is_grace"])
    grace_note_ratio = float(grace_count / n)

    # Para el resto ignoramos grace explícitas y usamos notas "reales"
    core = [e for e in note_events if not e["is_grace"]]
    if len(core) < 3:
        return {
            "grace_note_ratio": grace_note_ratio,
            "short_ornament_window_ratio": 0.0,
            "turn_like_ratio": 0.0,
            "appoggiatura_like_ratio": 0.0,
        }

    core_durs = np.array([e["dur"] for e in core], dtype=float)
    core_midis = np.array([e["midi"] for e in core], dtype=int)
    core_bs = np.array([e["beatStrength"] for e in core], dtype=float)

    valid_durs = core_durs[core_durs > 0]
    local_short_thr = min(SHORT_DUR_THRESHOLD_QL, float(np.median(valid_durs)) if len(valid_durs) else SHORT_DUR_THRESHOLD_QL)

    short_orn_windows = 0
    turn_like = 0
    appoggiatura_like = 0
    total_w3 = max(0, len(core) - 2)

    for i in range(total_w3):
        d = core_durs[i:i+3]
        p = core_midis[i:i+3]
        b = core_bs[i:i+3]

        ints = np.diff(p)
        abs_ints = np.abs(ints)

        # floreo / adorno breve local: tres notas breves y móviles
        if np.all(d <= local_short_thr) and np.all(abs_ints <= 2) and np.any(abs_ints > 0):
            short_orn_windows += 1

        # grupeto / turn-like:
        # patrón alrededor de una nota central: arriba-abajo o abajo-arriba, pasos pequeños
        if len(ints) == 2:
            if abs_ints[0] <= 2 and abs_ints[1] <= 2 and np.sign(ints[0]) == -np.sign(ints[1]) and np.sign(ints[0]) != 0:
                turn_like += 1

        # apoyatura ornamental:
        # nota fuerte no acorde/contextual inmediato que resuelve por paso a nota débil
        # proxy: primera nota fuerte, más larga/igual que la siguiente, resolución por semitono o tono
        if np.isfinite(b[0]) and np.isfinite(b[1]):
            if b[0] >= STRONG_BEAT_THRESHOLD and b[1] <= WEAK_BEAT_THRESHOLD:
                if abs(p[1] - p[0]) in (1, 2) and d[0] >= d[1]:
                    appoggiatura_like += 1

    denom_w3 = float(total_w3) if total_w3 > 0 else 1.0

    return {
        "grace_note_ratio": grace_note_ratio,
        "short_ornament_window_ratio": float(short_orn_windows / denom_w3),
        "turn_like_ratio": float(turn_like / denom_w3),
        "appoggiatura_like_ratio": float(appoggiatura_like / denom_w3),
    }


# -----------------------------
# Rhythmic idioms
# -----------------------------
def triplet_binary_features(part: stream.Part) -> Dict[str, float]:
    """
    Detecta tuplet/triplet en compases binarios.
    """
    note_count_binary = 0
    triplet_count_binary = 0

    try:
        measures = list(part.recurse().getElementsByClass(stream.Measure))
    except Exception:
        measures = []

    if not measures:
        return {"triplet_in_binary_ratio": 0.0}

    for meas in measures:
        ts = _get_measure_time_signature(meas)
        if not _is_binary_metric(ts):
            continue

        for el in meas.flat.notesAndRests:
            if isinstance(el, note.Rest):
                continue
            if not isinstance(el, (note.Note, chord.Chord)):
                continue

            note_count_binary += 1

            is_triplet = False
            try:
                for tup in getattr(el.duration, "tuplets", []):
                    if getattr(tup, "numberNotesActual", None) == 3:
                        is_triplet = True
                        break
            except Exception:
                pass

            # fallback aproximado por duraciones típicas de tresillo
            if not is_triplet:
                ql = _safe_dur_ql(el)
                # valores habituales: 1/3, 2/3, 1/6, 4/3, etc.
                cand = [1/6, 1/3, 2/3, 4/3]
                if any(abs(ql - c) < 0.02 for c in cand):
                    is_triplet = True

            if is_triplet:
                triplet_count_binary += 1

    ratio = float(triplet_count_binary / note_count_binary) if note_count_binary > 0 else 0.0
    return {"triplet_in_binary_ratio": ratio}


def dotted_rhythm_features(durs: np.ndarray) -> Dict[str, float]:
    """
    Detecta ritmos apuntillados típicos:
    relación aproximada 3:1 o 1:3 entre duraciones consecutivas.
    """
    if len(durs) < 2:
        return {"dotted_rhythm_ratio": 0.0}

    count = 0
    total = 0

    for d1, d2 in zip(durs[:-1], durs[1:]):
        if d1 <= 0 or d2 <= 0:
            continue
        total += 1
        r = d1 / d2
        if abs(r - 3.0) / 3.0 < 0.12 or abs(r - (1/3)) / (1/3) < 0.12:
            count += 1

    return {"dotted_rhythm_ratio": float(count / total) if total > 0 else 0.0}


# -----------------------------
# Melodic idioms and cadential figures
# -----------------------------
def initial_sixth_feature(midis: np.ndarray) -> Dict[str, float]:
    if len(midis) < 2:
        return {"initial_interval_is_6th": 0.0}

    first_int = abs(int(midis[1]) - int(midis[0]))
    return {"initial_interval_is_6th": float(1.0 if first_int in (8, 9) else 0.0)}


def final_leading_tone_appoggiatura_feature(midis: np.ndarray,
                                            beatStrengths: np.ndarray,
                                            tonic_pc: float) -> Dict[str, float]:
    """
    Apoyatura de sensible al final de frase/pieza:
    proxy robusto sobre las últimas 4 notas de la melodía.
    """
    if len(midis) < 2 or not np.isfinite(tonic_pc):
        return {"final_leading_tone_appoggiatura": 0.0}

    tonic_pc = int(tonic_pc) % 12
    leading_pc = (tonic_pc - 1) % 12

    start = max(0, len(midis) - 4)
    found = 0.0

    for i in range(start, len(midis) - 1):
        pc1 = int(midis[i] % 12)
        pc2 = int(midis[i + 1] % 12)
        semitone_res = abs(int(midis[i + 1]) - int(midis[i])) == 1
        strong = np.isfinite(beatStrengths[i]) and beatStrengths[i] >= STRONG_BEAT_THRESHOLD

        if pc1 == leading_pc and pc2 == tonic_pc and semitone_res and strong:
            found = 1.0
            break

    return {"final_leading_tone_appoggiatura": found}


def strong_weak_semitone_resolution_feature(midis: np.ndarray,
                                            beatStrengths: np.ndarray) -> Dict[str, float]:
    """
    Parte fuerte -> parte débil con resolución por semitono.
    """
    if len(midis) < 2:
        return {"strong_weak_semitone_resolution_ratio": 0.0}

    count = 0
    total = 0

    for i in range(len(midis) - 1):
        b1 = beatStrengths[i]
        b2 = beatStrengths[i + 1]
        if not (np.isfinite(b1) and np.isfinite(b2)):
            continue
        total += 1
        if b1 >= STRONG_BEAT_THRESHOLD and b2 <= WEAK_BEAT_THRESHOLD:
            if abs(int(midis[i + 1]) - int(midis[i])) == 1:
                count += 1

    return {
        "strong_weak_semitone_resolution_ratio": float(count / total) if total > 0 else 0.0
    }


# -----------------------------
# Chromatic / tonal idioms
# -----------------------------
def chromatic_features(midis: np.ndarray,
                       durs: np.ndarray,
                       melodic_stream: stream.Stream) -> Dict[str, float]:
    """
    % de notas no diatónicas respecto a la escala estimada.
    """
    try:
        k = melodic_stream.analyze("key")
        scale = k.getScale()
        scale_pcs = {p.pitchClass for p in scale.pitches}
    except Exception:
        return {"chromatic_usage_ratio": float("nan")}

    if len(midis) == 0 or len(durs) == 0:
        return {"chromatic_usage_ratio": 0.0}

    pcs = np.mod(midis, 12).astype(int)
    w = durs.astype(float)
    total_w = float(np.sum(w))
    if total_w <= 0:
        return {"chromatic_usage_ratio": 0.0}

    chrom_w = float(np.sum([ww for pc, ww in zip(pcs, w) if int(pc) not in scale_pcs]))
    return {"chromatic_usage_ratio": float(chrom_w / total_w)}


def retardo_tonal_features(midis: np.ndarray,
                           tonic_pc: float) -> Dict[str, float]:
    """
    Retardos tipo:
    - la / la-sol#  => tónica -> sensible inferior (descenso cromático)
    - si / si-la    => supertónica -> tónica
    Generalizados a la tonalidad estimada.
    """
    if len(midis) < 2 or not np.isfinite(tonic_pc):
        return {
            "retardo_la_solsharp_ratio": 0.0,
            "retardo_si_la_ratio": 0.0,
        }

    tonic_pc = int(tonic_pc) % 12
    leading_pc = (tonic_pc - 1) % 12
    supertonic_pc = (tonic_pc + 2) % 12

    pcs = np.mod(midis, 12).astype(int)

    la_solsharp = 0
    si_la = 0
    total = len(pcs) - 1

    for a, b, ma, mb in zip(pcs[:-1], pcs[1:], midis[:-1], midis[1:]):
        # tónica -> sensible inferior, descenso de semitono
        if a == tonic_pc and b == leading_pc and (int(mb) - int(ma) == -1):
            la_solsharp += 1

        # supertónica -> tónica, descenso
        if a == supertonic_pc and b == tonic_pc and (int(mb) < int(ma)):
            si_la += 1

    denom = float(total) if total > 0 else 1.0
    return {
        "retardo_la_solsharp_ratio": float(la_solsharp / denom),
        "retardo_si_la_ratio": float(si_la / denom),
    }


def minor_leading_to_mediant_feature(midis: np.ndarray,
                                     tonic_pc: float,
                                     mode_minor: float) -> Dict[str, float]:
    """
    Giro sensible–mediante en menor:
    sol#-la-si-do en A menor,
    generalizado como:
    sensible -> tónica -> supertónica -> mediante
    """
    if len(midis) < 4 or not np.isfinite(tonic_pc) or not np.isfinite(mode_minor) or int(mode_minor) != 1:
        return {"minor_leading_to_mediant_ratio": 0.0}

    tonic_pc = int(tonic_pc) % 12
    leading_pc = (tonic_pc - 1) % 12
    supertonic_pc = (tonic_pc + 2) % 12
    mediant_pc = (tonic_pc + 3) % 12  # tercera menor

    pcs = np.mod(midis, 12).astype(int)

    count = 0
    total = len(pcs) - 3

    for i in range(total):
        w = pcs[i:i+4]
        if list(w) == [leading_pc, tonic_pc, supertonic_pc, mediant_pc]:
            count += 1

    denom = float(total) if total > 0 else 1.0
    return {"minor_leading_to_mediant_ratio": float(count / denom)}








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

    # tonal = estimate_key_music21(part.flat, midis, durs)
    # cad = cadential_features(midis, durs, tonic_pc=tonal["best_key_pc"])
    # pdist = pitch_distribution_features(midis, durs)
    # rhy = rhythm_features(onsets, durs, beatStrengths)
    # iclass = interval_class_features(deltas)

    tonal = estimate_key_music21(part.flat, midis, durs)
    cad = cadential_features(midis, durs, tonic_pc=tonal["best_key_pc"])
    pdist = pitch_distribution_features(midis, durs)
    rhy = rhythm_features(onsets, durs, beatStrengths)
    iclass = interval_class_features(deltas)

    # Nuevas features idiomáticas
    orn = ornamentation_features(part)
    trip = triplet_binary_features(part)
    dotr = dotted_rhythm_features(durs)
    init6 = initial_sixth_feature(midis)
    final_app = final_leading_tone_appoggiatura_feature(
        midis, beatStrengths, tonal["best_key_pc"]
    )
    swsemi = strong_weak_semitone_resolution_feature(midis, beatStrengths)
    chrom = chromatic_features(midis, durs, part.flat)
    ret = retardo_tonal_features(midis, tonal["best_key_pc"])
    ltm = minor_leading_to_mediant_feature(
        midis, tonal["best_key_pc"], tonal["best_mode_minor"]
    )

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

    # return {
    #     # Ritmo
    #     "note_density": rhy["note_density"],
    #     "mean_dur": rhy["mean_dur"],
    #     "cv_dur": rhy["cv_dur"],
    #     "short_note_ratio": rhy["short_note_ratio"],
    #     "rhythmic_entropy": rhy["rhythmic_entropy"],
    #     "npvi_rhythmic": rhy["npvi_rhythmic"],
    #
    #     # Métrica / síncopa
    #     "strong_beat_note_ratio": rhy["strong_beat_note_ratio"],
    #     "syncopation_index": rhy["syncopation_index"],
    #
    #     # Pitch (distribución)
    #     "reciting_pitch_ratio": pdist["reciting_pitch_ratio"],
    #     "pitch_entropy": pdist["pitch_entropy"],
    #
    #     # Cadencia
    #     "final_tonic_match": cad["final_tonic_match"],
    #     "last_note_duration_ratio": cad["last_note_duration_ratio"],
    #     "last_interval_abs": cad["last_interval_abs"],
    #     "last_interval_is_step": cad["last_interval_is_step"],
    #
    #     # Armonía / tonalidad
    #     "diatonic_ratio": tonal["diatonic_ratio"],
    #     "best_corr": tonal["best_corr"],
    #     "key_clarity": tonal["key_clarity"],
    #     "best_key_pc": tonal["best_key_pc"],
    #     "best_mode_minor": tonal["best_mode_minor"],
    #
    #     # Intervalos cuantitativos
    #     "mean_abs_semitones": mean_abs_semitones,
    #     "max_leap": max_leap,
    #     "step_ratio": step_ratio,
    #     "unison_ratio": unison_ratio,
    #
    #     # Intervalos cualitativos (proxy)
    #     **iclass,
    #
    #     # Rango / propincuidad
    #     "range_semitones": range_semitones,
    #     "range_p95": range_p95,
    #     "proximity_step_le2": proximity_step_le2,
    #     "proximity_inv_mean": proximity_inv_mean,
    #
    #     # Contorno / dirección
    #     "up_ratio": up_ratio,
    #     "down_ratio": down_ratio,
    #     "stay_ratio": stay_ratio,
    #     "num_direction_changes": num_dir_changes,
    #     "climax_pos": climax_pos,
    #     "direction_balance": direction_balance,
    #     "pitch_time_slope": pitch_time_slope,
    #     #**ivar,
    #     #**rrange,
    #     #**phr,
    #     #**renergy,
    # }
    return {
        # Ritmo
        "note_density": rhy["note_density"],
        "mean_dur": rhy["mean_dur"],
        "cv_dur": rhy["cv_dur"],
        "short_note_ratio": rhy["short_note_ratio"],
        "rhythmic_entropy": rhy["rhythmic_entropy"],
        "npvi_rhythmic": rhy["npvi_rhythmic"],
        "triplet_in_binary_ratio": trip["triplet_in_binary_ratio"],
        "dotted_rhythm_ratio": dotr["dotted_rhythm_ratio"],

        # Métrica / síncopa
        "strong_beat_note_ratio": rhy["strong_beat_note_ratio"],
        "syncopation_index": rhy["syncopation_index"],

        # Ornamentación
        "grace_note_ratio": orn["grace_note_ratio"],
        "short_ornament_window_ratio": orn["short_ornament_window_ratio"],
        "turn_like_ratio": orn["turn_like_ratio"],
        "appoggiatura_like_ratio": orn["appoggiatura_like_ratio"],

        # Pitch (distribución)
        "reciting_pitch_ratio": pdist["reciting_pitch_ratio"],
        "pitch_entropy": pdist["pitch_entropy"],

        # Cadencia
        "final_tonic_match": cad["final_tonic_match"],
        "last_note_duration_ratio": cad["last_note_duration_ratio"],
        "last_interval_abs": cad["last_interval_abs"],
        "last_interval_is_step": cad["last_interval_is_step"],
        "initial_interval_is_6th": init6["initial_interval_is_6th"],
        "final_leading_tone_appoggiatura": final_app["final_leading_tone_appoggiatura"],
        "strong_weak_semitone_resolution_ratio": swsemi["strong_weak_semitone_resolution_ratio"],

        # Armonía / tonalidad
        "diatonic_ratio": tonal["diatonic_ratio"],
        "best_corr": tonal["best_corr"],
        "key_clarity": tonal["key_clarity"],
        "best_key_pc": tonal["best_key_pc"],
        "best_mode_minor": tonal["best_mode_minor"],
        "chromatic_usage_ratio": chrom["chromatic_usage_ratio"],
        "retardo_la_solsharp_ratio": ret["retardo_la_solsharp_ratio"],
        "retardo_si_la_ratio": ret["retardo_si_la_ratio"],
        "minor_leading_to_mediant_ratio": ltm["minor_leading_to_mediant_ratio"],

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

        #otras
        "rhythmic_energy": renergy["rhythmic_energy"],
        "interval_std": ivar["interval_std"],
        "range_relative": rrange["range_relative"],
        "mean_ioi": phr["mean_ioi"],
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