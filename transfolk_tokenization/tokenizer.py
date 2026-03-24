# ------------------------------
# tokenization/tokenizer.py
# ------------------------------

# ------------------------------
# TO DO PROBLEMS
# Notas ligadas que atraviesan compas
# Comprobar compases que no están completos
# ------------------------------

import os
from sympy import false
from tqdm import tqdm
from music21 import *

from fractions import Fraction


def closest_duration(value, allowed_durations):
    return min(allowed_durations, key=lambda x: abs(x - value))

def build_vocabulary(token_sequences):
    vocab = {"PAD": 0, "START": 1, "END": 2}
    idx = 3
    for seq in token_sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab[token] for token in tokens if token in vocab]

def ids_to_tokens(ids, inv_vocab):
    return [inv_vocab[i] for i in ids if i in inv_vocab]

def extract_tokens_from_musicxml(xml_path, time_signature, tonality, allowed_durations):
    try:
        score = converter.parse(xml_path)
        #1. Comprobar compás
        try:
            ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
            if ts.ratioString != time_signature:
                #print(f"⛔ Saltando {xml_path}: compás {ts.ratioString} ≠ requerido {TIME_SIGNATURE}")
                return None
        except IndexError:
            #print(f"⚠️ Sin compás encontrado en {xml_path}")
            return None

        #2. Analizar tonalidad y comprobar modo
        try:
            original_key = score.analyze('key')
            if original_key.mode != tonality:
                # print(f"⛔ Saltando {xml_path}: modo {original_key.mode} ≠ requerido {TONALITY}")
                return None
        except:
            #print(f"⚠️ No se pudo analizar la tonalidad en {xml_path}")
            return None

        #3. Transportar a Do mayor o La menor
        target_key = key.Key('C') if original_key.mode == 'major' else key.Key('A')
        intvl = interval.Interval(original_key.tonic, target_key.tonic)
        score = score.transpose(intvl)

        #4. Extraer tokens de notas y silencios
        melody = []
        for el in score.flatten().notesAndRests:
            dur = closest_duration(round(el.quarterLength, 6), allowed_durations)
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                # Transposición para asegurar rango [43, 71]
                while midi < 43:
                    midi += 12
                while midi > 71:
                    midi -= 12
                melody.append(f"NOTE_ON_{midi}")
                melody.append(f"DUR_{dur}")
            elif isinstance(el, note.Rest):
                melody.append("REST")
                melody.append(f"DUR_{dur}")

        melody.append("END")
        return melody
    except:
        return None

def extract_tokens_with_meter(xml_path, time_signature, tonality, allowed_durations):
    try:
        score = converter.parse(xml_path)

        try:
            ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
            if ts.ratioString != time_signature:
                # print(f"⛔ Saltando {xml_path}: compás {ts.ratioString} ≠ requerido {TIME_SIGNATURE}")
                return None
        except IndexError:
            # print(f"⚠️ Sin compás encontrado en {xml_path}")
            return None

        # 2. Analizar tonalidad y comprobar modo
        try:
            original_key = score.analyze('key')
            if original_key.mode != tonality:
                # print(f"⛔ Saltando {xml_path}: modo {original_key.mode} ≠ requerido {TONALITY}")
                return None
        except IndexError:
            # print(f"⚠️ No se pudo analizar la tonalidad en {xml_path}")
            return None

        # 3. Transportar a Do mayor o La menor
        target_key = key.Key('C') if original_key.mode == 'major' else key.Key('A')
        intvl = interval.Interval(original_key.tonic, target_key.tonic)
        score = score.transpose(intvl)

        # 4. Extraer tokens de notas y silencios
        melody = []

        # 5 Recorrer todos los compases
        part = score.parts[0]
        for measure in part.getElementsByClass(stream.Measure):
            melody.append("BAR") #añade el token BAR al inicio de cada compas

            for el in measure.notesAndRests:
                dur = closest_duration(round(el.quarterLength, 6), allowed_durations)
                # 6) Añadir nota o silencio
                if isinstance(el, note.Note):
                    midi = el.pitch.midi
                    # Transposición para asegurar rango [57,83]
                    while midi < 57:
                        midi += 12
                    while midi > 83:
                        midi -= 12
                    melody.append(f"NOTE_ON_{midi}")
                    melody.append(f"DUR_{dur}")
                elif isinstance(el, note.Rest):
                    melody.append("REST")
                    melody.append(f"DUR_{dur}")


        melody.append("END")
        return melody
    except Exception as e:
        return None

def extract_tokens_with_meter_modulation(xml_path, time_signature, tonality, allowed_durations):

    score = converter.parse(xml_path)
    denominatorLengh = 0
    try:
        ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
        denominatorLengh = 4 / ts.denominator
        if ts.ratioString != time_signature:
            # print(f"⛔ Saltando {xml_path}: compás {ts.ratioString} ≠ requerido {TIME_SIGNATURE}")
            return None
    except IndexError:
        # print(f"⚠️ Sin compás encontrado en {xml_path}")
        return None

    # 2. Analizar tonalidad y comprobar modo
    try:
        original_key = score.analyze('key')
        if original_key.mode != tonality:
            # print(f"⛔ Saltando {xml_path}: modo {original_key.mode} ≠ requerido {TONALITY}")
            return None
    except IndexError:
        # print(f"⚠️ No se pudo analizar la tonalidad en {xml_path}")
        return None

    # 3. Transportar a Do mayor o La menor
    if original_key.mode == 'major':
        target_key = key.Key('C')
        tonic_midi_target = 60  # C4
    else:
        target_key = key.Key('A')
        tonic_midi_target = 69  # A4

    # intervalo de transposición hacia la tonalidad base (DoM o Lam)
    intvl_to_target = interval.Interval(original_key.tonic, target_key.tonic)
    score = score.transpose(intvl_to_target)

    # ajuste final: garantizar que la nota tónica quede exactamente en el número MIDI deseado
    actual_tonic_midi = note.Note(target_key.tonic).pitch.midi
    adjust_intvl = tonic_midi_target - actual_tonic_midi
    score = score.transpose(adjust_intvl)

    # 4. Extraer tokens de notas y silencios
    melody = []

    # 5 Recorrer todos los compases
    part = score.parts[0]

    measures = part.getElementsByClass(stream.Measure)
    measures_num = len(measures)
    for i in range(measures_num):
    # for measure in part.getElementsByClass(stream.Measure):
        measure = measures[i]
        #añade el token BAR al inicio de cada compas
        melody.append("BAR")

        # comprobamos las modulaciones
        # analizamos la tonalidad considerando un bloque de tres compases (i, i+1, i+2)
        try:
            if i <= measures_num - 4:
                local_window = stream.Stream()
                local_window.append(measures[i])
                local_window.append(measures[i + 1])
                local_window.append(measures[i + 2])
                local_window.append(measures[i + 3])

                window_key = local_window.analyze('key')

                # si difiere de la tonalidad principal, consideramos modulación
                if window_key.tonic.name != target_key.tonic.name or window_key.mode != target_key.mode:
                    melody.append(f"MOD_{window_key.tonic.name}_{window_key.mode}")
                    target_key = window_key
        except Exception as e:
            # print(e)
            pass


        beat = 1
        for el in measure.notesAndRests:
            if beat == 1:
                melody.append(f"BEAT_{beat}")
            elif beat % denominatorLengh==0:
                melody.append(f"BEAT_{int(beat/ denominatorLengh)}")

            dur = closest_duration(round(el.quarterLength, 6), allowed_durations)
            # 6) Añadir nota o silencio
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                # Transposición para asegurar rango [57,83]
                while midi < 57:
                    midi += 12
                while midi > 83:
                    midi -= 12
                melody.append(f"NOTE_ON_{midi}")
                melody.append(f"DUR_{dur}")
                #Comprobamos si existe ligadura
                # if el.tie is not None and el.tie.type == 'start':
                #     melody.append("TIE_START")
                beat += dur

            elif isinstance(el, note.Rest):
                melody.append("REST")
                melody.append(f"DUR_{dur}")
                beat += dur


    melody.append("END")
    return melody

def extract_tokens_with_meter_modulation_strict(xml_path, time_signature, tonality,
                                         allowed_durations,
                                         strict_durations=False):

    score = converter.parse(xml_path)
    denominatorLengh = 0
    try:
        ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
        denominatorLengh = 4 / ts.denominator
        if ts.ratioString != time_signature:
            return None
    except IndexError:
        return None

    try:
        original_key = score.analyze('key')
        if original_key.mode != tonality:
            return None
    except IndexError:
        return None

    if original_key.mode == 'major':
        target_key = key.Key('C')
        tonic_midi_target = 60
    else:
        target_key = key.Key('A')
        tonic_midi_target = 69

    intvl_to_target = interval.Interval(original_key.tonic, target_key.tonic)
    score = score.transpose(intvl_to_target)

    actual_tonic_midi = note.Note(target_key.tonic).pitch.midi
    adjust_intvl = tonic_midi_target - actual_tonic_midi
    score = score.transpose(adjust_intvl)

    part = score.parts[0]
    measures = part.getElementsByClass(stream.Measure)
    measures_num = len(measures)

    if strict_durations:
        for m in measures:
            for el in m.notesAndRests:
                dur_raw = round(el.quarterLength, 6)
                dur = closest_duration(dur_raw, allowed_durations)
                if dur_raw not in allowed_durations:
                    print(
                        f"⛔ Saltando {xml_path}: duración no permitida {dur_raw} "
                        f"(permitidas: {sorted(allowed_durations)})"
                    )
                    return None

    melody = []

    for i in range(measures_num):
        measure = measures[i]
        melody.append("BAR")

        try:
            if i <= measures_num - 4:
                local_window = stream.Stream()
                local_window.append(measures[i])
                local_window.append(measures[i + 1])
                local_window.append(measures[i + 2])
                local_window.append(measures[i + 3])

                window_key = local_window.analyze('key')

                if window_key.tonic.name != target_key.tonic.name or window_key.mode != target_key.mode:
                    melody.append(f"MOD_{window_key.tonic.name}_{window_key.mode}")
                    target_key = window_key
        except:
            pass

        beat = 1
        for el in measure.notesAndRests:
            if beat == 1:
                melody.append(f"BEAT_{beat}")
            elif beat % denominatorLengh == 0:
                melody.append(f"BEAT_{int(beat / denominatorLengh)}")

            dur = closest_duration(round(el.quarterLength, 6), allowed_durations)

            if isinstance(el, note.Note):
                midi = el.pitch.midi
                while midi < 57:
                    midi += 12
                while midi > 83:
                    midi -= 12
                melody.append(f"NOTE_ON_{midi}")
                melody.append(f"DUR_{dur}")
                beat += dur

            elif isinstance(el, note.Rest):
                melody.append("REST")
                melody.append(f"DUR_{dur}")
                beat += dur

    melody.append("END")
    return melody

def compute_pulse_beat(ts) -> float:
    """
    Calcula la duración del pulso (en quarterLength) a partir de una TimeSignature.

    - Compases simples: pulso = unidad base (negra, corchea, etc.)
    - Compases compuestos (6/8, 9/8, 12/8...): pulso = grupo de 3 subdivisiones

    Parameters
    ----------
    ts : music21.meter.TimeSignature

    Returns
    -------
    float
        Duración del pulso en unidades de negra (quarterLength)
    """

    denominator_length = 4 / ts.denominator

    is_compound = (ts.numerator % 3 == 0) and (ts.numerator > 3)

    if is_compound:
        return denominator_length * 3
    else:
        return denominator_length

def validate_measure_duration(measure, expected_length, allowed_durations, tol=1e-6):
    """
    Comprueba si la suma de duraciones cuantizadas del compás coincide con su duración esperada.

    Parameters
    ----------
    measure : music21.stream.Measure
    expected_length : float
        Duración esperada del compás (en quarterLength)
    allowed_durations : list[float]
    tol : float
        Tolerancia numérica

    Returns
    -------
    is_valid : bool
    beat_sum : float
    diff : float
    """

    beat_sum = 0.0

    for el in measure.notesAndRests:
        if el.isNote and not el.duration.isGrace:
            dur = closest_duration(el.quarterLength, allowed_durations)
            beat_sum += dur
        elif el.isRest:
            dur = closest_duration(el.quarterLength, allowed_durations)
            beat_sum += dur


    diff = expected_length - beat_sum

    is_valid = abs(diff) <= tol

    return is_valid, beat_sum, diff

def log_error(errors, xml_path, error_type, **kwargs):
    file_name = os.path.basename(xml_path)

    if file_name not in errors:
        errors[file_name] = []

    entry = {"type": error_type}
    entry.update(kwargs)

    errors[file_name].append(entry)

def extract_tokens_with_meter_modulation_full_measures(
        xml_path, time_signature, tonality, allowed_durations, errors,
        mark_bars=True,
        strict_measure_rejection=True,
        strict_modality_rejection=True,
        mark_ts_changes=False,
        mark_grace_notes=False,
        mark_beats=False):

    score = converter.parse(xml_path)
    melody = []

    # --------------------------------------------
    # 1. Time Signature inicial (robusto + tipado)
    # --------------------------------------------
    ts_obj = score.recurse().getElementsByClass(meter.TimeSignature).first()
    if not isinstance(ts_obj, meter.TimeSignature):
        log_error(errors, xml_path, "no_time_signature_found")
        print(f"\n⚠️ Excluded file {xml_path}: No Time signature.")
        return None

    ts: meter.TimeSignature = ts_obj

    if strict_measure_rejection and ts.ratioString != time_signature:
        log_error(errors, xml_path, "no_ts_match", found=ts.ratioString, expected=time_signature)
        print(f"\n⚠️ Excluded file {xml_path}: Time Signature does not match with required one.")
        return None

    melody.append(f"TS_{ts.ratioString}")

    # --------------------------------------------
    # 2. Tonalidad
    # --------------------------------------------
    try:
        original_key = score.analyze('key')
    except Exception:
        log_error(errors, xml_path, "no_tonality_found")
        print(f"\n⚠️ Excluded file {xml_path}: Tonality does not found.")
        return None

    if strict_modality_rejection and original_key.mode != tonality:
        log_error(errors, xml_path, "no_tonality_match", found=original_key.mode, expected=tonality)
        print(f"\n⚠️ Excluded file {xml_path}: Tonality does not match with required one.")
        return None

    melody.append(f"MODE_{original_key.mode}")

    # --------------------------------------------
    # 3. Transposición a C/A
    # --------------------------------------------
    if original_key.mode == 'major':
        target_key = key.Key('C')
        tonic_midi_target = 60
    elif original_key.mode == 'minor':
        target_key = key.Key('A')
        tonic_midi_target = 69
    else:
        log_error(errors, xml_path, "tonality_not_supported", found=original_key.mode, expected=tonality)
        print(f"\n⚠️ Excluded file {xml_path}: Tonality {original_key.mode} not supported.")
        return None

    intvl_to_target = interval.Interval(original_key.tonic, target_key.tonic)
    score = score.transpose(intvl_to_target)

    actual_tonic_midi = note.Note(target_key.tonic).pitch.midi
    adjust_intvl = tonic_midi_target - actual_tonic_midi
    score = score.transpose(adjust_intvl)

    # --------------------------------------------
    # 4. Iteración por compases
    # --------------------------------------------
    part = score.parts[0]
    measures = part.getElementsByClass(stream.Measure)

    for i, measure in enumerate(measures):

        ts_changed = False

        # --------------------------------------------
        # 4.1 Detección de cambio de compás (tipado seguro)
        # --------------------------------------------
        ts_in_measure_obj = measure.getElementsByClass(meter.TimeSignature).first()

        if isinstance(ts_in_measure_obj, meter.TimeSignature):
            if ts_in_measure_obj.ratioString != ts.ratioString:
                ts = ts_in_measure_obj
                ts_changed = True

        # --------------------------------------------
        # 5. Métrica
        # --------------------------------------------
        denominator_length = 4 / ts.denominator
        pulse_beat = compute_pulse_beat(ts)
        expected_length = float(ts.numerator * denominator_length)

        # --------------------------------------------
        # 6. Validación del compás
        # --------------------------------------------
        is_valid, beat_sum, diff = validate_measure_duration(
            measure, expected_length, allowed_durations
        )

        if not is_valid:
            log_error(errors, xml_path, f"measure_inconsistent", measure=i+1, found=round(beat_sum, 6), expected=expected_length)
            print(f"\n⛔ Compás {i + 1} excluido: duración {round(beat_sum, 6)} ≠ {expected_length}")
            continue

        # --------------------------------------------
        # 7. Tokens estructurales
        # --------------------------------------------
        if mark_bars:
            melody.append("BAR")

        if mark_ts_changes and ts_changed:
            melody.append(f"TS_{ts.ratioString}")

        # --------------------------------------------
        # 8. Eventos del compás
        # --------------------------------------------
        beat = 0.0

        for el in measure.notesAndRests:

            # --------------------------------------------
            # 8.1 Marcado de beats
            # --------------------------------------------
            if mark_beats:
                beat_idx = beat / pulse_beat
                if abs(beat_idx - round(beat_idx)) < 1e-6:
                    melody.append(f"BEAT_{int(round(beat_idx)) + 1}")

            # --------------------------------------------
            # 8.2 Notas
            # --------------------------------------------
            if isinstance(el, note.Note):
                midi = el.pitch.midi
                while midi < 57:
                    midi += 12
                while midi > 83:
                    midi -= 12

                # Grace notes → nunca afectan al beat
                if el.duration.isGrace:
                    if mark_grace_notes:
                        melody.append(f"NOTE_ON_{midi}")
                        melody.append("DUR_0")
                    continue

                melody.append(f"NOTE_ON_{midi}")

                dur = closest_duration(el.quarterLength, allowed_durations)
                melody.append(f"DUR_{dur}")
                beat += dur

            # --------------------------------------------
            # 8.3 Silencios
            # --------------------------------------------
            elif isinstance(el, note.Rest):

                dur = closest_duration(el.quarterLength, allowed_durations)
                melody.append("REST")
                melody.append(f"DUR_{dur}")
                beat += dur

    melody.append("END")
    return melody

def extract_tokens_with_meter_modulation_full_measures2(xml_path, time_signature, tonality, allowed_durations):
    """
    Tokenizador robusto con control métrico fuerte:
    - Verifica compás y tonalidad.
    - Transporta a C mayor / A menor.
    - SOLO usa allowed_durations tanto en notas reales como en rellenos.
    - Si un compás queda corto: añade tantas figuras permitidas como haga falta.
    - Si queda largo: elimina o recorta la última figura usando allowed_durations.
    - El pitch del relleno siempre sigue la última nota encontrada.
    """

    print(f"\n═══════════════════════════════")
    print(f"Procesando archivo: {xml_path}")
    print(f"═══════════════════════════════")

    score = converter.parse(xml_path)

    # ================================================================
    # 1) Verificar compás
    # ================================================================
    try:
        ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
        denominatorLengh = 4 / ts.denominator
        if ts.ratioString != time_signature:
            print(f"⛔ Saltado: compás {ts.ratioString} esperado {time_signature}")
            return None
        print(f"✔ Compás correcto: {ts.ratioString}")
    except IndexError:
        print("⛔ Sin compás encontrado.")
        return None

    # ================================================================
    # 2) Verificar tonalidad
    # ================================================================
    try:
        original_key = score.analyze('key')
        if original_key.mode != tonality:
            print(f"⛔ Tonalidad rechazada: {original_key.mode}, esperaba {tonality}")
            return None
        print(f"✔ Tonalidad correcta: {original_key.tonic.name} {original_key.mode}")
    except Exception:
        print("⛔ Error analizando tonalidad.")
        return None

    # ================================================================
    # 3) Transporte a C mayor o A menor
    # ================================================================
    if original_key.mode == 'major':
        target_key = key.Key('C')
        tonic_midi_target = 60
    else:
        target_key = key.Key('A')
        tonic_midi_target = 69

    print("→ Transportando…")
    intvl_to_target = interval.Interval(original_key.tonic, target_key.tonic)
    score = score.transpose(intvl_to_target)

    actual_tonic_midi = note.Note(target_key.tonic).pitch.midi
    adjust_intvl = tonic_midi_target - actual_tonic_midi
    score = score.transpose(adjust_intvl)
    print("✔ Transporte completado")

    # ================================================================
    # 4) Preparación inicial
    # ================================================================
    melody = []
    part = score.parts[0]
    measures = part.getElementsByClass(stream.Measure)
    expected_length = float(ts.numerator * denominatorLengh)

    print(f"Duración esperada de cada compás: {expected_length}")

    # ================================================================
    # 5) Procesar cada compás
    # ================================================================
    for i, measure in enumerate(measures, start=1):
        print(f"\n--- COMPÁS {i} -----------------------------------")
        melody.append("BAR")

        # ----------------------------
        # Detección de modulación
        # ----------------------------
        try:
            if i <= len(measures) - 3:
                local_window = stream.Stream()
                for j in range(4):
                    local_window.append(measures[i - 1 + j])
                window_key = local_window.analyze('key')

                if window_key.tonic.name != target_key.tonic.name or window_key.mode != target_key.mode:
                    print(f"⚠ Modulación detectada → {window_key.tonic.name} {window_key.mode}")
                    melody.append(f"MOD_{window_key.tonic.name}_{window_key.mode}")
                    target_key = window_key
        except Exception:
            pass

        beat = 0.0
        last_pitch = None
        last_is_rest = False

        # ================================================================
        # 6) Tokenizar elementos reales del compás
        # ================================================================
        for el in measure.notesAndRests:

            if beat == 0:
                melody.append("BEAT_1")
            elif beat % denominatorLengh == 0:
                melody.append(f"BEAT_{int(beat / denominatorLengh) + 1}")

            # cuantización usando allowed_durations
            d_real = round(el.quarterLength, 6)
            dur = closest_duration(d_real, allowed_durations)

            if abs(d_real - dur) > 1e-6:
                print(f"  ↳ Ajuste duración {d_real} → {dur}")

            # --------------------------
            # Nota
            # --------------------------
            if isinstance(el, note.Note):
                midi = el.pitch.midi

                # normalizar registro
                while midi < 57: midi += 12
                while midi > 83: midi -= 12

                last_pitch = midi
                last_is_rest = False

                melody.append(f"NOTE_ON_{midi}")
                melody.append(f"DUR_{dur}")
                beat += dur

            # --------------------------
            # Silencio
            # --------------------------
            elif isinstance(el, note.Rest):
                last_is_rest = True
                melody.append("REST")
                melody.append(f"DUR_{dur}")
                beat += dur

        # ================================================================
        # 7) Cierre de compás SOLO con allowed_durations
        # ================================================================
        diff = expected_length - beat
        print(f"→ Longitud compás antes de cierre: {beat} (objetivo {expected_length})")

        # ------------------------------------------------------------------
        # CASO A: COMPÁS CORTO → añadir figuras de allowed_durations
        # ------------------------------------------------------------------
        if diff > 1e-6:
            print(f"  🟢 Compás incompleto → rellenando {diff}")

            remaining = diff
            while remaining > 1e-6:

                candidates = [d for d in allowed_durations if d <= remaining + 1e-6]
                if candidates:
                    d = max(candidates)
                else:
                    d = min(allowed_durations)

                if last_is_rest or last_pitch is None:
                    print(f"     + REST {d}")
                    melody.append("REST")
                else:
                    print(f"     + NOTE_ON_{last_pitch} dur={d}")
                    melody.append(f"NOTE_ON_{last_pitch}")

                melody.append(f"DUR_{d}")
                remaining -= d

        # ------------------------------------------------------------------
        # CASO B: COMPÁS LARGO → recortar o eliminar última figura
        # ------------------------------------------------------------------
        elif diff < -1e-6:
            exceso = -diff
            print(f"  🔴 Compás sobrepasado → reducir {exceso}")

            j = len(melody) - 1
            while j >= 0 and not melody[j].startswith("DUR_"):
                j -= 1

            if j < 0:
                print("⚠ No se encontró DUR para ajustar. Se continúa.")
                continue

            old_dur = float(melody[j].replace("DUR_", ""))

            reduce_candidates = [d for d in allowed_durations if d <= old_dur]

            if not reduce_candidates:
                print(f"  ❌ Eliminando última figura {old_dur}")
                del melody[j - 1: j + 1]
            else:
                target_dur = min(reduce_candidates, key=lambda d: abs((old_dur - exceso) - d))
                print(f"  🟠 Ajuste DUR {old_dur} → {target_dur}")
                melody[j] = f"DUR_{target_dur}"

    # ================================================================
    # 8) Finalizar
    # ================================================================
    melody.append("END")
    print("\n✔ Archivo procesado correctamente\n")
    return melody

def pattern_to_string(elements):
    """
    Convierte una secuencia de notas/silencios de music21 en string con nombre y duración.
    """
    parts = []
    for el in elements:
        if isinstance(el, note.Note):
            dur = round(el.quarterLength, 6)
            parts.append(f"{el.nameWithOctave}:{dur}")
        elif isinstance(el, note.Rest):
            dur = round(el.quarterLength, 6)
            parts.append(f"Rest:{dur}")
    return "|".join(parts)

def process_musicxml_directory(directory_path, max_files, algorithm, time_signature, tonality, allowed_durations):
    token_sequences = []

    filenames = [
        f for f in os.listdir(directory_path)
        if f.endswith(".xml") or f.endswith(".musicxml")
    ]

    # Si se especifica max_files, limitar la lista de archivos
    if max_files is not None:
        filenames = filenames[:max_files]

    progress_bar = tqdm(filenames, desc="Procesando obras", unit="obra", dynamic_ncols=True)
    excluded = 0
    processed = 0
    errors = {}
    for filename in progress_bar:
        progress_bar.set_description(f"🎼 Procesando: {filename}")
        filepath = os.path.join(directory_path, filename)

        tokens = process_musicxml_file(filepath, algorithm, time_signature, tonality, allowed_durations, errors)
        token_sequences.append(tokens)
        # try:
        #     if algorithm=="patterns" or algorithm=="standard":
        #         tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
        #                                                                     allowed_durations, errors,
        #                                                                     strict_modality_rejection=True,
        #                                                                     strict_measure_rejection=True,
        #                                                                     mark_bars=True,
        #                                                                     mark_beats=True,
        #                                                                     mark_ts_changes=False,
        #                                                                     mark_grace_notes=False)
        #     elif algorithm=="baseline":
        #         tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
        #                                                                     allowed_durations, errors,
        #                                                                     strict_modality_rejection=True,
        #                                                                     strict_measure_rejection=True,
        #                                                                     mark_bars=False,
        #                                                                     mark_beats=False,
        #                                                                     mark_ts_changes=False,
        #                                                                     mark_grace_notes=False)
        #     elif algorithm=="chm":
        #         tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
        #                                                                     allowed_durations, errors,
        #                                                                     strict_modality_rejection=True,
        #                                                                     strict_measure_rejection=False,
        #                                                                     mark_bars=True,
        #                                                                     mark_beats=True,
        #                                                                     mark_ts_changes=True,
        #                                                                     mark_grace_notes=True)
        #     elif algorithm=="momet":
        #         tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
        #                                                                     allowed_durations, errors,
        #                                                                     strict_modality_rejection=False,
        #                                                                     strict_measure_rejection=False,
        #                                                                     mark_bars=True,
        #                                                                     mark_beats=True,
        #                                                                     mark_ts_changes=True,
        #                                                                     mark_grace_notes=True)

        if tokens is not None:
            processed+=1
        else:
            excluded+=1

    print(f"Obras procesadas:{processed}, excluidas:{excluded}.")

    vocab = build_vocabulary(token_sequences)
    token_id_sequences = [tokens_to_ids(seq, vocab) for seq in token_sequences]
    return token_id_sequences, vocab, errors

def process_musicxml_file(filepath, algorithm, time_signature, tonality, allowed_durations, errors):
    try:
        tokens = None
        if algorithm == "patterns" or algorithm == "standard":
            tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
                                                                        allowed_durations, errors,
                                                                        strict_modality_rejection=False,
                                                                        strict_measure_rejection=False,
                                                                        mark_bars=True,
                                                                        mark_beats=True,
                                                                        mark_ts_changes=False,
                                                                        mark_grace_notes=False)
        elif algorithm == "baseline":
            tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
                                                                        allowed_durations, errors,
                                                                        strict_modality_rejection=False,
                                                                        strict_measure_rejection=False,
                                                                        mark_bars=False,
                                                                        mark_beats=False,
                                                                        mark_ts_changes=False,
                                                                        mark_grace_notes=False)
        elif algorithm == "chm":
            tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
                                                                        allowed_durations, errors,
                                                                        strict_modality_rejection=True,
                                                                        strict_measure_rejection=False,
                                                                        mark_bars=True,
                                                                        mark_beats=True,
                                                                        mark_ts_changes=True,
                                                                        mark_grace_notes=True)
        elif algorithm == "momet":
            tokens = extract_tokens_with_meter_modulation_full_measures(filepath, time_signature, tonality,
                                                                        allowed_durations, errors,
                                                                        strict_modality_rejection=False,
                                                                        strict_measure_rejection=False,
                                                                        mark_bars=True,
                                                                        mark_beats=True,
                                                                        mark_ts_changes=True,
                                                                        mark_grace_notes=True)
    except Exception as e:
        return None

    if tokens is not None:
        return ["START"] + tokens
    else:
        return None






