from pathlib import Path
from fractions import Fraction
from music21 import converter, stream, note, chord, meter, key, metadata, harmony
from music21 import pitch as m21pitch
from music21.duration import quarterConversion
from copy import deepcopy
from music21 import stream, note, meter, chord
from .common import is_grace

from copy import deepcopy
from music21 import stream, meter, note
import copy
from music21 import stream, meter, note

from music21 import meter, note

from music21 import stream, meter, note


def get_part1_all_voices(input_stream):
    """
    Recibe un music21.stream.Stream y devuelve un music21.stream.Part.
    Para cada compás, si Voice 1 está vacía (solo silencios) pero hay música
    en Voice 2, 3 o 4, copia la primera Voice con contenido musical a Voice 1.
    """

    # 1. Obtener la primera parte
    parts = input_stream.getElementsByClass(stream.Part)
    if not parts:
        raise ValueError("El stream no contiene ninguna Part")

    original_part = parts[0]

    # 2. Comprobamos que la parte no este vacia
    orig_measures = list(original_part.getElementsByClass(stream.Measure))
    if len(orig_measures) == 0:
        raise Exception(f"⚠️ Sin compases detectados. Omitido.")

    # 3. Crear nueva Part de salida
    new_part = stream.Part()
    new_part.id = original_part.id
    new_part.insert(0, original_part.getInstrument(returnDefault=True))

    # 4. Iterar por compases
    for measure in original_part.getElementsByClass(stream.Measure):

        new_measure = stream.Measure(number=measure.number)

        # Copiar atributos (clave, compás, armadura, etc.)
        for attr in measure.flat:
            pass  # (no-op, seguridad)

        if measure.timeSignature:
            new_measure.insert(0, measure.timeSignature)
        if measure.keySignature:
            new_measure.insert(0, measure.keySignature)
        if measure.clef:
            new_measure.insert(0, measure.clef)

        voices = list(measure.getElementsByClass(stream.Voice))

        # Si no hay voces explícitas, copiamos el contenido tal cual
        if not voices:
            for el in measure.notesAndRests:
                new_measure.append(el)
            new_part.append(new_measure)
            continue

        # Asegurar orden Voice 1, 2, 3, 4...
        voices.sort(key=lambda v: v.id if v.id is not None else 0)

        def voice_has_music(v):
            return any(n.isNote or (n.isChord and n.notes) for n in v.notes)

        # Voice 1
        voice1 = voices[0]

        if voice_has_music(voice1):
            # Caso normal: Voice 1 tiene música
            for el in voice1.notesAndRests:
                new_measure.append(el)

        else:
            # Voice 1 solo silencios → buscar fallback
            fallback_voice = None
            for v in voices[1:]:
                if voice_has_music(v):
                    fallback_voice = v
                    break

            if fallback_voice is not None:
                for el in fallback_voice.notesAndRests:
                    new_measure.append(el)
            else:
                # Todas las voces son silencios → copiar Voice 1 tal cual
                for el in voice1.notesAndRests:
                    new_measure.append(el)

        new_part.append(new_measure)

    return new_part

def copy_key_signature(score, new_part):
    try:
        orig_key = score.recurse().getElementsByClass(key.KeySignature)[0]
        new_part.insert(0.0, key.KeySignature(orig_key.sharps))
        print(f"🎯 Armadura copiada: {orig_key.sharps} alteraciones.")
    except IndexError:
        print("⚠️ Sin armadura detectada (se omitirá).")
    except Exception as e:
        print(f"⚠️ Error leyendo armadura (se omitirá): {e}")

def trim_score(orig_measures):
    if not orig_measures:
        raise Exception(f"⚠️ solo compases vacíos, omitido.")

    # --- Eliminar compases iniciales vacíos o de solo silencios ---
    while orig_measures:
        first_measure = orig_measures[0]
        has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in first_measure.notesAndRests)
        if not has_note:
            orig_measures.pop(0)
        else:
            break

    # --- Eliminar compases finales vacíos o de solo silencios ---
    while orig_measures:
        last_measure = orig_measures[-1]
        has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in last_measure.notesAndRests)
        if not has_note:
            orig_measures.pop(-1)
        else:
            break

def trim_score2(orig_measures):
    if not orig_measures:
        raise Exception(f"⚠️ solo compases vacíos, omitido.")

    # --- Eliminar compases iniciales vacíos o de solo silencios ---
    while orig_measures:
        first_measure = orig_measures[0]
        has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in first_measure.notesAndRests)
        if not has_note:
            orig_measures.pop(0)
        else:
            break

    # --- Eliminar compases finales vacíos o de solo silencios ---
    while orig_measures:
        last_measure = orig_measures[-1]
        has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in last_measure.notesAndRests)
        if not has_note:
            orig_measures.pop(-1)
        else:
            break
    return orig_measures
    # ============================================================
    # NUEVA FUNCIONALIDAD (corregida):
    # Si queda un "bloque musical" muy corto seguido de un bloque
    # largo de compases vacíos, eliminar ambos y repetir.
    # ============================================================

    MIN_EMPTY_BLOCK = 2          # "varios compases vacíos" = 2 o más
    MAX_SHORT_MEASURES = 2       # cuántos compases con notas consideramos "suelto"
    MAX_SHORT_NOTES = 4          # cuántas notas totales consideramos "suelto"

    def note_count(m):
        return sum(1 for el in m.notesAndRests if isinstance(el, (note.Note, chord.Chord)))

    def is_empty_measure(m):
        return note_count(m) == 0

    # Repetir por si hay más de un "arranque falso"
    while True:
        if not orig_measures:
            raise Exception(f"⚠️ solo compases vacíos, omitido.")

        # 1) Primer bloque musical (desde 0 hasta antes del primer bloque largo de silencios)
        i = 0
        short_measures = 0
        short_notes = 0

        # Consumir compases NO vacíos iniciales (bloque musical inicial)
        while i < len(orig_measures) and not is_empty_measure(orig_measures[i]):
            short_measures += 1
            short_notes += note_count(orig_measures[i])
            i += 1

        # Si no hay música al principio, no tiene sentido (ya debería estar trimmeado)
        if short_measures == 0:
            break

        # 2) Contar bloque de compases vacíos que viene justo después
        j = i
        empty_len = 0
        while j < len(orig_measures) and is_empty_measure(orig_measures[j]):
            empty_len += 1
            j += 1

        # 3) Regla: si el bloque musical es "muy corto" y va seguido de un silencio largo,
        # eliminar bloque musical + silencio, y volver a evaluar desde el nuevo inicio.
        if empty_len >= MIN_EMPTY_BLOCK and short_measures <= MAX_SHORT_MEASURES and short_notes <= MAX_SHORT_NOTES:
            # eliminamos [0 : j)
            orig_measures[:] = orig_measures[j:]

            # tras recortar, vuelve a limpiar silencios iniciales por si quedaron
            while orig_measures:
                first_measure = orig_measures[0]
                has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in first_measure.notesAndRests)
                if not has_note:
                    orig_measures.pop(0)
                else:
                    break

            continue

        break

    return orig_measures

# def clean_sparse_measures(measures, density_threshold=0.35):
#     """
#     Elimina compases con baja densidad de notas rodeados de compases vacíos o con silencios.
#
#     Parámetros
#     ----------
#     measures : list[music21.stream.Measure]
#         Lista de compases.
#     density_threshold : float
#         Umbral mínimo de densidad de notas.
#
#     Returns
#     -------
#     list[music21.stream.Measure]
#         Nueva lista de compases limpia.
#     """
#
#     # -----------------------------
#     # FUNCIONES AUXILIARES
#     # -----------------------------
#
#     def measure_duration(m):
#         return float(m.barDuration.quarterLength)
#
#     def note_duration(m):
#         return sum(el.quarterLength for el in m.notes)
#
#     def density(m):
#         total = measure_duration(m)
#         if total == 0:
#             return 0
#         return note_duration(m) / total
#
#     def is_empty(m):
#         return len(m.notesAndRests) == 0
#
#     def only_rests(m):
#         if len(m.notesAndRests) == 0:
#             return False
#         return all(isinstance(el, note.Rest) for el in m.notesAndRests)
#
#     def low_density(m):
#         return density(m) < density_threshold
#
#     # ---------------------------------
#     # 1–2 Eliminación de compases sparse
#     # ---------------------------------
#
#     cleaned = []
#
#     for i, m in enumerate(measures):
#
#         left = measures[i - 1] if i > 0 else None
#         right = measures[i + 1] if i < len(measures) - 1 else None
#
#         remove = False
#
#         if low_density(m):
#
#             if i == len(measures) - 1:  # último compás
#                 if left and (is_empty(left) or only_rests(left)):
#                     remove = True
#
#             else:
#                 cond_left = left and (is_empty(left) or only_rests(left))
#                 cond_right = right and (is_empty(right) or only_rests(right))
#
#                 if cond_left or cond_right:
#                     remove = True
#
#         if not remove:
#             cleaned.append(m)
#
#     # ---------------------------------
#     # 3 eliminar compases completamente vacíos
#     # ---------------------------------
#
#     cleaned = [m for m in cleaned if not is_empty(m)]
#
#     # ---------------------------------
#     # 4 eliminar compases con solo silencios
#     # ---------------------------------
#
#     cleaned = [m for m in cleaned if not only_rests(m)]
#
#     return cleaned
#

def clean_sparse_measures(measures, density_threshold=0.35):
    """
    Limpia una lista de compases eliminando compases con baja densidad de notas
    rodeados de silencios o compases vacíos.

    Parameters
    ----------
    measures : list[music21.stream.Measure]
    density_threshold : float

    Returns
    -------
    list[music21.stream.Measure]
    """

    # -----------------------------
    # FUNCIONES AUXILIARES
    # -----------------------------

    def measure_duration(m):
        if m.barDuration:
            return float(m.barDuration.quarterLength)
        return sum(el.quarterLength for el in m.notesAndRests)

    def note_duration(m):
        return sum(el.quarterLength for el in m.notes)

    def density(m):
        total = measure_duration(m)
        if total == 0:
            return 0
        return note_duration(m) / total

    def is_empty(m):
        return len(m.notesAndRests) == 0

    def only_rests(m):
        if len(m.notesAndRests) == 0:
            return False
        return all(isinstance(el, note.Rest) for el in m.notesAndRests)

    def has_notes(m):
        return len(m.notes) > 0

    def low_density(m):
        return density(m) < density_threshold

    # ---------------------------------
    # PASO 1–3 eliminación compases sparse
    # ---------------------------------

    cleaned = []

    for i, m in enumerate(measures):

        left = measures[i - 1] if i > 0 else None
        right = measures[i + 1] if i < len(measures) - 1 else None

        remove = False

        if low_density(m):

            # -------------------------
            # caso último compás
            # -------------------------
            if i == len(measures) - 1:
                if left and only_rests(left):
                    remove = True

            else:

                # condición derecha
                right_silent = right and (only_rests(right) or is_empty(right))

                # condición izquierda silenciosa
                left_silent = left and (only_rests(left) or is_empty(left))

                # condición izquierda con notas
                left_has_notes = left and has_notes(left)

                # regla 1
                if right_silent and not left_has_notes:
                    remove = True

                # regla 2
                if left_silent:
                    remove = True

        if not remove:
            cleaned.append(m)

    # ---------------------------------
    # PASO 4 eliminar compases vacíos
    # ---------------------------------

    cleaned = [m for m in cleaned if not is_empty(m)]

    # ---------------------------------
    # PASO 5 eliminar compases solo silencios
    # ---------------------------------

    cleaned = [m for m in cleaned if not only_rests(m)]

    return cleaned



def estimate_list_time_signatures(orig_measures):
    time_signatures = []
    for m in orig_measures:
        total = 0.0
        for el in m.notesAndRests:
            if isinstance(el, harmony.ChordSymbol):
                continue
            if is_grace(el):
                continue
            try:
                total += float(el.quarterLength)
            except Exception:
                pass
        if total > 0:
            time_signatures.append(calculate_time_signature(total))

    if not time_signatures:
        raise Exception(f"⚠️ sin contenido rítmico (tras ignorar graces). Omitido.")

    return time_signatures

def looks_like_grace(el) -> bool:
    """
    Heurística robusta: en music21, una grace suele tener duration.isGrace True o quarterLength 0
    (ojo: hay notas normales con ql=0 en casos raros, pero esto suele ser correcto para MusicXML).
    """
    try:
        if hasattr(el, "duration") and getattr(el.duration, "isGrace", False):
            return True
    except Exception:
        pass
    try:
        if hasattr(el, "quarterLength") and float(el.quarterLength) == 0.0:
            # Muchos MusicXML marcan grace como ql=0
            return True
    except Exception:
        pass
    return False

def ql_to_frac(x: float) -> Fraction:
    return Fraction(x).limit_denominator(96)

def calculate_time_signature(measure_len: float):
    # Falta poder distinguir un 3/4de un 6/8

    candidates = [
        (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4),(10,4),(11,4),(12,4),(13,4),(14,4),(15,4),(16,4),(17,4),(18,4),(19,4),
        (1,8),(3,8),(5,8),(6,8),(7,8),(9,8),(11,8),(12,8),(13,8),
        (14,8),(15,8),(17,8),(21,8),(23,8),(15,16),(17,16),(19,16)
    ]

    try:
        measure_len = float(measure_len)

        def bar_len(n, d):
            return 4 * n / d

        num, den = min(
            candidates,
            key=lambda s: abs(measure_len - bar_len(*s))
        )

        return meter.TimeSignature(f"{num}/{den}")

    except Exception as e:
        print(f"Error al calcular el time signature {measure_len}: {e}")
        return None

def expected_len_from_ts(ts: meter.TimeSignature) -> Fraction:
        return ql_to_frac(ts.numerator * (4 / ts.denominator))

def list_musicxml_files(root: Path):
    return [p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".xml", ".mxl", ".musicxml")]

def closest_allowed(frac_dur: Fraction, allowed_durations) -> Fraction:
    allowed_fracs = [ql_to_frac(d) for d in allowed_durations]
    return min(allowed_fracs, key=lambda a: abs(a - frac_dur))

def clamp_midi(midi_min, midi_max, midi_val: int) -> int:
    while midi_val < midi_min:
        midi_val += 12
    while midi_val > midi_max:
        midi_val -= 12
    return midi_val

def clamp_pitch_octave(p: m21pitch.Pitch, midi_min: int, midi_max: int) -> m21pitch.Pitch:
    """
    Devuelve un Pitch con el MISMO nombre (step + accidental),
    ajustando solo la octava para que su MIDI caiga en [midi_min, midi_max].
    """
    # new_p = p.clone()
    new_p = m21pitch.Pitch(p.nameWithOctave)

    while new_p.midi < midi_min:
        new_p.octave += 1

    while new_p.midi > midi_max:
        new_p.octave -= 1

    return new_p

def create_grace_note(el, midi_min, midi_max):
    midi_val=60
    if isinstance(el, note.Note):
        midi_val = clamp_pitch_octave(el.pitch, midi_min, midi_max)
    elif isinstance(el, chord.Chord):
        top_pitch = max(el.pitches, key=lambda p: p.midi)
        midi_val = clamp_midi(midi_min, midi_max, int(top_pitch.midi))
    elif isinstance(el, note.Unpitched):
        midi_val=60
        try:
            top_pitch = max(el.pitches, key=lambda p: p.midi)
            midi_val = clamp_midi(midi_min, midi_max, int(top_pitch.midi))
        except Exception:
            pass
        try:
            midi_val = el.displayName
        except Exception:
            pass
    gn = note.Note(midi_val).getGrace()
    tipo = '16th'  # Forzamos la duracion del grace a semicorcheas
    gn = note.Note(midi_val, type=tipo).getGrace()
    gn.duration.slash = True  # barra diagonal
    return gn

def create_note_or_rest(el, dur_q, midi_min, midi_max, respect_ties):
    # Rest
    if isinstance(el, note.Rest):
        return note.Rest(quarterLength=float(dur_q))

    # Note
    elif isinstance(el, note.Note):
        # midi_val = clamp_midi(int(el.pitch.midi))
        p = clamp_pitch_octave(el.pitch, midi_min, midi_max)
        n = note.Note(p, quarterLength=float(dur_q))
        if respect_ties:
            n.tie=el.tie
        return n

    # Chord -> top pitch only
    elif isinstance(el, chord.Chord):
        top_pitch = max(el.pitches, key=lambda p: p.midi)
        midi_val = clamp_midi(midi_min, midi_max, int(top_pitch.midi))
        n = note.Note(midi_val, quarterLength=float(dur_q))
        if respect_ties:
            n.tie=el.tie
        return n

    elif isinstance(el, note.Unpitched):
        midi_val = el.displayName
        n = note.Note(midi_val, quarterLength=float(dur_q))
        if respect_ties:
            n.tie = el.tie
        return n

def sort_anacruse(elems):
    rests = [e for e in elems if isinstance(e, note.Rest)]
    notes = [e for e in elems if isinstance(e, note.Note)]

    if rests and notes:
        elems[:] = rests + notes
        print(f"🟢 c. Silencios reordenados en anacrusa.")
        return True
    else:
        return False

def fill_anacruse(elems, exp_len):
    rests = [e for e in elems if isinstance(e, note.Rest)]
    notes = [e for e in elems if isinstance(e, note.Note)]

    #Comprueba si a la anacrusa le faltan silencios
    suma = 0
    for rest in rests:
        suma = suma + rest.quarterLength
    for nota in notes:
        suma = suma + nota.quarterLength
    dif = exp_len-suma
    if dif>0:
        rests.insert(0, note.Rest(quarterLength=dif))
        print(f"🟢 c. Silencios añadidos a anacrusa.")


    if rests and notes:
        elems[:] = rests + notes
        print(f"🟢 c. Silencios reordenados en anacrusa.")
        return True
    else:
        return False

def sum_total_len_mesure(elems):
    total_len = Fraction(0, 1)
    for e in elems:
        if not is_grace(e):
            try:
                total_len += ql_to_frac(e.quarterLength)
            except Exception:
                pass

    return total_len

def remove_last_note_tie(events):
    """
    Elimina el tie de la última nota de una lista de notas/silencios.

    Parameters
    ----------
    events : list[music21.note.Note | music21.note.Rest]

    Returns
    -------
    list
        La misma lista con el tie eliminado si la última nota lo tenía.
    """

    last_note = None

    for el in reversed(events):
        if isinstance(el, note.Note):
            last_note = el
            break

    if last_note and last_note.tie is not None:
        last_note.tie = None

    return events



from music21 import stream, note

def clean_invalid_ties(measures):
    """
    Recorre una lista de music21.stream.Measure y elimina ligaduras incorrectas.
    """

    modified = False

    # Construir stream temporal con jerarquía válida
    temp_stream = stream.Stream()
    for m in measures:
        temp_stream.append(m)

    # Obtener notas en orden temporal fiable
    notes = list(temp_stream.recurse().getElementsByClass(note.Note))

    for i in range(len(notes) - 1):
        n = notes[i]
        next_n = notes[i + 1]

        if n.tie is not None:
            if n.pitch != next_n.pitch:
                n.tie = None
                modified = True

    return modified



def fix_edge_time_signatures(measures):

    if not measures or len(measures) < 2:
        return measures

    # -----------------------------
    # Helpers
    # -----------------------------
    def get_ts(measure):
        ts_list = measure.flatten().getElementsByClass(meter.TimeSignature)

        # Caso sin TimeSignature explícita
        if len(ts_list) == 0:
            print("Compás sin TS!!!!!!!!!!!")
            ts = meter.TimeSignature("4/4")
            measure.timeSignature = ts
            return ts

        # Asegurar que coges la primera real por offset
        ts_list = sorted(ts_list, key=lambda x: x.offset)

        return ts_list[0]

    def measure_duration(measure):
        return sum(el.quarterLength for el in measure.notesAndRests)

    def measure_end_offset(measure):
        if not measure.notesAndRests:
            return 0.0
        return max(el.offset + el.quarterLength for el in measure.notesAndRests)

    def ts_duration(ts):
        return ts.barDuration.quarterLength

    def add_rest_left(measure, dur):
        if dur <= 0:
            return

        # desplazar contenido
        for el in measure.notesAndRests:
            el.offset += dur

        # insertar silencio al inicio
        r = note.Rest(quarterLength=dur)
        measure.insert(0.0, r)

    def add_rest_right(measure, dur):
        if dur <= 0:
            return

        # calcular final REAL del compás
        end = measure_end_offset(measure)

        # insertar justo al final
        r = note.Rest(quarterLength=dur)
        measure.insert(end, r)

    # -----------------------------
    # PRIMER COMPÁS
    # -----------------------------
    first = measures[0]
    second = measures[1]

    ts_first = get_ts(first)
    ts_second = get_ts(second)

    dur_first = ts_first.barDuration.quarterLength
    dur_second = ts_second.barDuration.quarterLength

    if (dur_first < dur_second) and (dur_first in [1.0, 1.5]):
        first.timeSignature = meter.TimeSignature(ts_second.ratioString)

        target_dur = ts_duration(ts_second)
        current_dur = measure_duration(first)
        missing = target_dur - current_dur

        if missing > 1e-6:
            add_rest_left(first, missing)

    # -----------------------------
    # ÚLTIMO COMPÁS
    # -----------------------------
    last = measures[-1]
    penultimate = measures[-2]

    ts_last = get_ts(last)
    ts_prev = get_ts(penultimate)

    if ts_last.ratioString != ts_prev.ratioString:
        dur_last = ts_duration(ts_last)
        dur_prev = ts_duration(ts_prev)

        if dur_last < dur_prev:
            last.timeSignature = meter.TimeSignature(ts_prev.ratioString)
            ts_last = last.timeSignature

    # asegurar completitud SIEMPRE
    target_dur_last = ts_duration(ts_last)
    current_dur_last = measure_duration(last)

    missing_last = target_dur_last - current_dur_last

    if missing_last > 1e-6:
        add_rest_right(last, missing_last)

    return measures



def normalize_musicxml_corpus_new(
    data_dir,
    output_dir,
    allowed_durations,
    overwrite=False,
    delete_grace_notes=True,
    midi_min=57,
    midi_max=83,
    create_title=False,
    respect_time_signature_changes=False,
    respect_ties=True
):
    """
    Limpieza y normalización métrica y rítmica avanzada de corpus MusicXML.

    Fix importante:
    - Las grace notes NO se cuantizan, NO cuentan para el tiempo del compás y NO deben avanzar offsets.
      Si se preservan (delete_grace_notes=False), se insertan como objetos grace puros (duración métrica 0).
    """

    # -------------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------------
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔤 DATA NORMALIZATION: {data_dir}")

    # Obtiene todas los xml del directorio
    files = list_musicxml_files(data_dir)
    print(f"\n🎼 Archivos detectados: {len(files)}\n")

    total_read = 0
    total_processed = 0
    total_ignored = 0
    total_errors = 0
    total_saved = 0

    for src in files:
        print(f"──────────────────────────────────────────────")
        print(f"Procesando: {src.name}")

        out_path = output_dir / src.name
        if out_path.exists() and not overwrite:
            print(f"⏭️ Ya existe en salida y overwrite=False → omitido: {out_path.name}")
            total_ignored += 1
            continue

        try:
            score = converter.parse(src)
            total_read += 1
        except Exception as e:
            print(f"⛔ Error al leer {src.name}: {e}")
            total_errors += 1
            continue

        # 1. Crea la partitura nueva
        new_score = stream.Score()
        new_part = stream.Part()
        measures = []

        # 2. Copia el Título = nombre del archivo
        if create_title:
            if new_score.metadata is None:
                new_score.insert(0, metadata.Metadata())
            new_score.metadata.title = src.stem

        # 3. Copiar armadura
        copy_key_signature(score, new_part)


        # 4. Obtener primera parte, revisando las voces 2, 3 y 4
        try:
            orig_part = get_part1_all_voices(score)
        except Exception as e:
            print(f"⚠️ {src.name}: Error obteniendo parts ({e}). Omitido.")
            total_ignored += 1
            continue


        # 5.A.  Eliminar compases iniciales/finales vacíos o de solo silencios ---
        orig_measures=[]
        try:
            orig_measures = list(orig_part.getElementsByClass(stream.Measure))
            orig_measures = trim_score2(orig_measures)
        except Exception as e:
            print(f"⚠️ {src.name}: Error eliminando compases vacios ({e}). Omitido.")
            total_ignored += 1
            continue

        # 5.A.  Eliminar compases con muy baja densidad de notas/notas sueltas ---
        try:
            orig_measures = clean_sparse_measures(orig_measures, density_threshold=0.35)
        except Exception as e:
            print(f"⚠️ {src.name}: Error eliminando compases vacios ({e}). Omitido.")
            total_ignored += 1
            continue




        # 6. Estimar TIME SIGNATURE global (ignorando grace notes para el cálculo de longitudes) ---
        ts_to_use = ""
        time_signatures = []
        try:
            # ts_to_use, exp_len = estimate_time_signature(score, orig_measures)
            time_signatures = estimate_list_time_signatures(orig_measures)
            # new_part.insert(0.0, meter.TimeSignature(ts_to_use.ratioString))
            # print(f"✅ Compás asignado: {ts_to_use.ratioString} (duración esperada {float(exp_len):.2f})")
        except Exception as e:
            print(f"⚠️ {src.name}: Error estimando la lista global de TS ({e}). Omitido.")
            total_ignored += 1
            continue


        modified = False

        # ---------------------------------------------------------------------
        # 7. Procesar todos los compases
        # ---------------------------------------------------------------------

        for mi, m in enumerate(orig_measures):
            new_measure = stream.Measure(number=mi+1)
            elems = []
            ts_mes=None

            #Detectamos cambios de Time signature
            try:
                if len(m.flat.getElementsByClass("TimeSignature"))==0:
                    #Si el compás no tiene definida el time signature, asignamos el estimado
                    ts_mes=time_signatures[mi]
                    new_measure.insert(0.0, meter.TimeSignature(ts_mes.ratioString))
                else:
                    ts_mes = m.flat.getElementsByClass("TimeSignature")[0]
                    if ts_mes.ratioString!=time_signatures[mi].ratioString:
                        ts_mes=time_signatures[mi]
                        print(f"🔧 TS inconsistente. Asignamos: {ts_mes.ratioString}")
                    new_measure.insert(0.0, meter.TimeSignature(ts_mes.ratioString))

            except Exception as e:
                print(f"Error al acceder a la time signature {e}")
                pass
            exp_len = expected_len_from_ts(ts_mes)

            # 8. Metemos notas, silencios y si procede notas de paso
            for el in m.notesAndRests:
                if isinstance(el, harmony.ChordSymbol):
                    continue

                grace = is_grace(el)
                if grace:
                    # -------------------------
                    # GRACE NOTES (NO CUANTIZAR)
                    # -------------------------
                    if delete_grace_notes:
                        modified = True
                        continue
                    else:
                        gn = create_grace_note(el, midi_min, midi_max)
                        elems.append(gn)
                        continue
                else:
                    # -------------------------
                    # NOTAS Y SILENCIOS (cuantizar)
                    # -------------------------
                    # Aproxima la duracion a las duraciones permitidas
                    dur_frac = ql_to_frac(float(el.quarterLength))
                    dur_q = closest_allowed(dur_frac, allowed_durations)
                    if abs(dur_q - dur_frac) > Fraction(1, 192):
                        print(f"🔧 c.{m.number}: dur {float(dur_frac):.3f} ajustada a {float(dur_q):.3f}")
                        modified = True
                    elems.append(create_note_or_rest(el, dur_q, midi_min, midi_max, respect_ties))


            # =====================================================
            # 9. PRIMER COMPÁS: reordenar ANACRUSA
            # =====================================================
            if mi == 0 and len(elems) > 0:
                if sort_anacruse(elems):
                    modified = True


            # 10. Calcular la Longitud total del compas (sin contar graces) ---
            total_len = sum_total_len_mesure(elems)

            # =====================================================
            # 11 ULTIMO COMPÁS: revisar ties
            # =====================================================
            if mi == len(orig_measures) and len(elems) > 1:
                elems= remove_last_note_tie(elems)



            for e in elems:
                new_measure.append(e)

            measures.append(new_measure)





        # =====================================================
        # 12 COMPROBAR LIGADURAS INCONEXAS
        # =====================================================
        clean_invalid_ties(measures)
        # --- Guardar ---

        # =====================================================
        # 13 Detectar y corregir todas las anacrusas, inicio, fin
        # =====================================================
        measures = fix_edge_time_signatures(measures)


        new_part.append(measures)

        new_score.append(new_part)
        try:
            new_score.write('musicxml', fp=str(out_path).replace(".mxl",".xml"))
            print(f"💾 Guardado → {out_path.name}")
            total_saved += 1
            total_processed += 1
        except Exception as e:
            print(f"❌ Error guardando {out_path.name}: {e}")
            total_errors += 1

    # --- Informe final ---
    print("\n═══════════════════════════════════════════════")
    print("📊 INFORME FINAL")
    print(f" Archivos encontrados:   {len(files)}")
    print(f" Leídos correctamente:   {total_read}")
    print(f" Procesados y guardados: {total_processed}")
    print(f" Ignorados:              {total_ignored}")
    print(f" Errores:                {total_errors}")
    print(f" Archivos escritos:      {total_saved}")
    print("═══════════════════════════════════════════════")
    print("✅ Limpieza de corpus completada.")




