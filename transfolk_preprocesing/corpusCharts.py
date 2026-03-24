
import os
import sys
import math
import glob
import warnings
from collections import Counter
from fractions import Fraction

import matplotlib.pyplot as plt

from music21 import converter, meter, key, note, chord, stream

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Utilidades ----------
def safe_parse(path):
    """Parsea una partitura con tolerancia a errores. Devuelve stream.Score o None."""
    try:
        sc = converter.parse(path)
        # Asegurar Score (algunos XML devuelven Opus/Part/Stream)
        if isinstance(sc, stream.Opus):
            sc = sc.scores[0] if sc.scores else None
        return sc
    except Exception as e:
        print(f"[WARN] No se pudo parsear {os.path.basename(path)} ({e})")
        return None

def get_first_time_signature(sc: stream.Score):
    """Devuelve el primer compás (TimeSignature) encontrado como ratioString (p. ej. '4/4')."""
    try:
        ts = sc.recurse().getElementsByClass(meter.TimeSignature)
        if ts:
            return ts[0].ratioString
    except Exception:
        pass
    return "Desconocido"

def analyze_global_key(sc: stream.Score):
    """
    Analiza tonalidad global con music21.analyze('key').
    Devuelve (tonalidad_texto, modo) p.ej. ('C major', 'major') o ('A minor','minor').
    Si no concluyente, devuelve ('Indeterminada','other').
    """
    try:
        k = sc.analyze('key')
        if isinstance(k, key.Key):
            name = f"{k.tonic.name} {k.mode}"
            mode = k.mode if k.mode in ('major', 'minor') else 'other'
            return name, mode
    except Exception:
        pass
    return "Indeterminada", "other"

def first_part(sc: stream.Score):
    """Devuelve la primera Part (primer pentagrama) del Score, o None si no existe."""
    try:
        return sc.parts[0] if sc.parts else None
    except Exception:
        return None

def range_first_part_semitones(p: stream.Part):
    """
    Rango (en semitonos) dentro del primer pentagrama:
    - Ignora silencios y eventos sin duración real (grace, quarterLength <= 0).
    - Si hay acordes, usa únicamente la nota superior (línea superior).
    """
    if p is None:
        return None

    top_midis = []
    for el in p.recurse().notesAndRests:
        if el.isRest:
            continue
        # excluir notas de gracia
        try:
            if float(el.duration.quarterLength) <= 0:
                continue
        except Exception:
            pass

        if isinstance(el, note.Note) and el.pitch is not None:
            top_midis.append(el.pitch.midi)
        elif isinstance(el, chord.Chord) and el.pitches:
            try:
                el.sortAscending(inPlace=False)
                top_midis.append(el.pitches[-1].midi)  # nota superior
            except Exception:
                top_midis.append(max(p_.midi for p_ in el.pitches))

    if not top_midis:
        return None

    return int(round(max(top_midis) - min(top_midis)))

def piece_total_duration_ql(sc: stream.Score):
    """Duración total de la obra en quarterLength (usa highestTime)."""
    try:
        return float(sc.highestTime)
    except Exception:
        return None

def quantize_ql(x, decimals=5):
    """
    Cuantiza una quarterLength a un string estable para usar como clave del histograma.
    Redondea a 'decimals' para evitar ruido por coma flotante (tresillos, puntillos).
    """
    try:
        if isinstance(x, Fraction):
            x = float(x)
        s = f"{round(float(x), decimals):.{decimals}f}"
        return s.rstrip('0').rstrip('.') if '.' in s else s
    except Exception:
        return "NaN"

def collect_all_durations_ql(sc: stream.Score):
    """
    Recolecta TODAS las duraciones de eventos (notas, acordes y silencios) en quarterLength.
    Excluye duraciones <= 0 (grace) y None.
    Devuelve lista de strings cuantizados (p.ej. '0.33333').
    """
    out = []
    try:
        for el in sc.recurse().notesAndRests:
            try:
                ql = el.duration.quarterLength
                if ql is None:
                    continue
                if float(ql) <= 0:
                    continue
                out.append(quantize_ql(ql, decimals=5))
            except Exception:
                continue
    except Exception:
        pass
    return out

def collect_rest_durations_ql(sc: stream.Score):
    """
    Recolecta SOLO las duraciones de silencios en quarterLength.
    Excluye duraciones <= 0 (grace) y None.
    Devuelve lista de strings cuantizados (p.ej. '0.50000').
    """
    out = []
    try:
        for el in sc.recurse().notesAndRests:
            if not getattr(el, 'isRest', False):
                continue
            try:
                ql = el.duration.quarterLength
                if ql is None:
                    continue
                if float(ql) <= 0:
                    continue
                out.append(quantize_ql(ql, decimals=5))
            except Exception:
                continue
    except Exception:
        pass
    return out

# ---------- Pipeline principal ----------
def analyze_folder(folder):
    paths = []
    for ext in ("*.musicxml", "*.xml", "*.mxl", "*.MusicXML", "*.XML", "*.MXL"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(list(set(paths)))

    if not paths:
        print(f"[ERROR] No se han encontrado archivos MusicXML/XML en: {folder}")
        sys.exit(1)

    timeSig_per_piece = []
    mode_per_piece = []
    key_per_piece = []
    final_duration_per_piece = []
    firstpart_range_per_piece = []

    durations_all = []       # TODAS las duraciones (notes + rests) en quarterLength
    rest_durations_all = []  # SOLO duraciones de silencios en quarterLength

    processed = 0
    for pth in paths:
        sc = safe_parse(pth)
        if sc is None:
            continue
        processed += 1

        # Compás (primer TS)
        ts = get_first_time_signature(sc)
        timeSig_per_piece.append(ts)

        # Tonalidad y modo
        kname, kmode = analyze_global_key(sc)
        key_per_piece.append(kname)
        mode_per_piece.append(kmode if kmode in ('major', 'minor') else 'other')

        # Duración total de la obra
        total_ql = piece_total_duration_ql(sc)
        if total_ql is not None:
            final_duration_per_piece.append(total_ql)

        # Rango del PRIMER pentagrama
        p1 = first_part(sc)
        rng = range_first_part_semitones(p1) if p1 is not None else None
        if rng is not None:
            firstpart_range_per_piece.append(rng)

        # Duraciones rítmicas de todos los eventos
        durations_all.extend(collect_all_durations_ql(sc))
        # Duraciones SOLO de silencios
        rest_durations_all.extend(collect_rest_durations_ql(sc))

        print(f"[OK] {os.path.basename(pth)} | TS={ts} | Key={kname} | Mode={kmode} | DurQL={total_ql} | RangoP1={rng}")

    print(f"\nProcesados {processed} de {len(paths)} archivos.")

    return {
        "timeSig_per_piece": timeSig_per_piece,
        "mode_per_piece": mode_per_piece,
        "key_per_piece": key_per_piece,
        "durations_all": durations_all,
        "rest_durations_all": rest_durations_all,
        "final_duration_per_piece": final_duration_per_piece,
        "firstpart_range_per_piece": firstpart_range_per_piece,
        "n_pieces": processed
    }

# ---------- Visualización ----------
def plot_bar_with_percentages(counter: Counter, title: str, n_total_pieces: int, xlabel: str, rotate_labels=True):
    labels, counts = zip(*sorted(counter.items(), key=lambda x: (-x[1], x[0]))) if counter else ([], [])
    percents = [(c / n_total_pieces) * 100 if n_total_pieces > 0 else 0 for c in counts]

    plt.figure()
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45 if rotate_labels else 0, ha='right')
    plt.ylabel("Número de obras")
    plt.xlabel(xlabel)
    plt.title(f"{title}\n(entre paréntesis: porcentaje de obras)")
    # Mostrar porcentajes encima
    for i, (c, p) in enumerate(zip(counts, percents)):
        plt.text(i, c, f"{p:.1f}%", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_histogram_numeric(values, bins, title, xlabel):
    if not values:
        print(f"[INFO] Sin datos para '{title}'")
        return
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

def main(folder):
    results = analyze_folder(folder)

    n_pieces = results["n_pieces"]
    if n_pieces == 0:
        print("[ERROR] No se pudo procesar ninguna obra válida.")
        return

    # 1) Histograma de compases (por obra) con porcentaje
    ts_counter = Counter(results["timeSig_per_piece"])
    plot_bar_with_percentages(
        ts_counter,
        title="Histograma de compases por obra",
        n_total_pieces=n_pieces,
        xlabel="Compás (Time Signature)"
    )

    # 2) Histograma de modalidades (major/minor/other) (por obra) con porcentaje
    mode_counter = Counter(results["mode_per_piece"])
    plot_bar_with_percentages(
        mode_counter,
        title="Histograma de modalidades por obra",
        n_total_pieces=n_pieces,
        xlabel="Modo"
    )

    # 3) Histograma de tonalidades (por obra) con porcentaje
    key_counter = Counter(results["key_per_piece"])
    plot_bar_with_percentages(
        key_counter,
        title="Histograma de tonalidades por obra",
        n_total_pieces=n_pieces,
        xlabel="Tonalidad"
    )

    # 4) Histograma de unidades rítmicas (todas las duraciones en quarterLength)
    dur_counter = Counter(results["durations_all"])

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return math.inf

    dur_items_sorted = sorted(dur_counter.items(), key=lambda kv: _safe_float(kv[0]))
    if dur_items_sorted:
        labels = [k for k, _ in dur_items_sorted]
        counts = [v for _, v in dur_items_sorted]
        plt.figure()
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=90, ha='right')
        plt.ylabel("Frecuencia (eventos)")
        plt.xlabel("Duración en quarterLength")
        plt.title("Histograma de unidades rítmicas (todas las duraciones encontradas)")
        plt.tight_layout()
        plt.show()
    else:
        print("[INFO] Sin datos de duraciones rítmicas.")

    # 4b) Histograma de duraciones de silencios (quarterLength)
    rest_counter = Counter(results["rest_durations_all"])
    rest_items_sorted = sorted(rest_counter.items(), key=lambda kv: _safe_float(kv[0]))
    if rest_items_sorted:
        labels = [k for k, _ in rest_items_sorted]
        counts = [v for _, v in rest_items_sorted]
        plt.figure()
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=90, ha='right')
        plt.ylabel("Frecuencia (silencios)")
        plt.xlabel("Duración de silencio en quarterLength")
        plt.title("Histograma de duraciones de silencios")
        plt.tight_layout()
        plt.show()
    else:
        print("[INFO] Sin datos de silencios.")

    # 5) Histograma de duraciones finales de cada obra (en quarterLength)
    finals = [x for x in results["final_duration_per_piece"] if x is not None]
    bins_finals = max(5, int(math.sqrt(len(finals)))) if finals else 5
    plot_histogram_numeric(
        finals,
        bins=bins_finals,
        title="Histograma de duraciones finales (por obra)",
        xlabel="Duración total en quarterLength"
    )

    # 6) Histograma de rangos del primer pentagrama por obra (en semitonos)
    ranges = [x for x in results["firstpart_range_per_piece"] if x is not None]
    bins_ranges = max(5, int(math.sqrt(len(ranges)))) if ranges else 5
    plot_histogram_numeric(
        ranges,
        bins=bins_ranges,
        title="Histograma de rangos (primer pentagrama) por obra",
        xlabel="Rango (semitonos, max - min)"
    )

if __name__ == "__main__":
    # 👇 Escribe aquí la ruta de la carpeta que quieres analizar
    carpeta = r"C:\Todos"

    if not os.path.isdir(carpeta):
        print(f"[ERROR] Carpeta no válida: {carpeta}")
        sys.exit(1)

    main(carpeta)





