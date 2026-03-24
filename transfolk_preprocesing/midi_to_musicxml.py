import os
import math
import copy
from collections import defaultdict

from music21 import converter, meter, stream, note, chord, tempo, key


def closest_duration(value, allowed_durations):
    """
    Devuelve la duración permitida más cercana.
    """
    return min(allowed_durations, key=lambda x: abs(float(x) - float(value)))


def get_existing_time_signatures(score):
    """
    Devuelve una lista de pares (offset, TimeSignature) ordenados por offset.
    Si el MIDI contiene cambios de compás legibles por music21, aparecerán aquí.
    """
    ts_items = []
    for el in score.recurse():
        if isinstance(el, meter.TimeSignature):
            ts_items.append((float(el.getOffsetInHierarchy(score)), el))

    ts_items.sort(key=lambda x: x[0])

    # eliminar duplicados exactos consecutivos
    cleaned = []
    prev = None
    for off, ts in ts_items:
        sig = ts.ratioString
        if prev is None or prev[0] != off or prev[1] != sig:
            cleaned.append((off, ts))
            prev = (off, sig)

    return cleaned


def strip_to_single_part(score):
    """
    Convierte cualquier score/partitura a una única Part monoflujo de eventos,
    preservando offsets globales de notesAndRests.
    """
    part = stream.Part()

    for el in score.flatten().notesAndRests:
        cloned = copy.deepcopy(el)
        part.insert(float(el.offset), cloned)

    return part


def quantize_part(part, allowed_durations, min_duration=0.125):
    """
    Cuantiza offsets y duraciones de una Part.
    - Redondea offsets a rejilla basada en la menor duración permitida.
    - Ajusta quarterLength a la duración permitida más cercana.
    """
    qpart = stream.Part()

    grid = min(float(x) for x in allowed_durations if float(x) > 0)
    grid = max(grid, min_duration)

    for el in part.flatten().notesAndRests:
        new_el = copy.deepcopy(el)

        ql = float(el.quarterLength)
        if ql <= 0:
            continue

        ql_q = closest_duration(ql, allowed_durations)

        # cuantización del offset
        off = float(el.offset)
        off_q = round(off / grid) * grid

        new_el.quarterLength = ql_q
        qpart.insert(off_q, new_el)

    return qpart


def collect_onsets_by_candidate_bar(part, ts_candidates):
    """
    Para cada compás candidato, calcula un score basado en:
    - cuántos eventos caen cerca del inicio de compás
    - cuántos eventos caen cerca de pulsos internos regulares
    """
    onsets = [float(el.offset) for el in part.flatten().notesAndRests]
    if not onsets:
        return {}

    scores = {}

    for ts_str in ts_candidates:
        ts = meter.TimeSignature(ts_str)
        bar_len = float(ts.barDuration.quarterLength)

        # pulsos internos según el denominador
        # para 6/8 tratamos agrupación binaria de negra con puntillo
        if ts_str == "6/8":
            beat_positions = [0.0, 1.5, 3.0, 4.5]
            strong_positions = [0.0, 3.0]
        else:
            beat_len = 4.0 / ts.denominator
            beat_positions = [i * beat_len for i in range(int(round(bar_len / beat_len)))]
            strong_positions = [0.0]

        score = 0.0

        for off in onsets:
            pos = off % bar_len

            # proximidad a inicio de compás
            d_bar = min(abs(pos - 0.0), abs(pos - bar_len))
            if d_bar < 0.10:
                score += 3.0

            # proximidad a pulsos fuertes
            for sp in strong_positions:
                if abs(pos - sp) < 0.10:
                    score += 1.5

            # proximidad a beats
            for bp in beat_positions:
                if abs(pos - bp) < 0.10:
                    score += 0.5

        scores[ts_str] = score

    return scores


def infer_global_time_signature(part, candidate_time_signatures=None):
    """
    Infiere un compás global si el MIDI no contiene TimeSignature utilizable.
    """
    if candidate_time_signatures is None:
        candidate_time_signatures = ["2/4", "3/4", "4/4", "6/8", "9/8", "12/8"]

    scores = collect_onsets_by_candidate_bar(part, candidate_time_signatures)

    if not scores:
        return meter.TimeSignature("4/4")

    best = max(scores.items(), key=lambda x: x[1])[0]
    return meter.TimeSignature(best)


def make_ts_map(existing_ts, fallback_ts, total_length):
    """
    Construye un mapa de cambios de compás:
    lista ordenada de (offset_inicio, TimeSignature).
    """
    if existing_ts:
        ts_map = [(off, copy.deepcopy(ts)) for off, ts in existing_ts]
        if ts_map[0][0] > 0:
            ts_map.insert(0, (0.0, copy.deepcopy(ts_map[0][1])))
        return ts_map

    return [(0.0, copy.deepcopy(fallback_ts))]


def get_ts_for_offset(ts_map, offset):
    """
    Devuelve la TS vigente en un offset dado.
    """
    current_ts = ts_map[0][1]
    for off, ts in ts_map:
        if offset >= off:
            current_ts = ts
        else:
            break
    return current_ts


def rebuild_score_with_measures(
    part,
    ts_map,
    carry_metadata=True,
    infer_anacrusis=False
):
    """
    Reconstruye una Part con compases explícitos y cambios de compás.

    Supuestos:
    - part ya está cuantizada
    - ts_map = [(offset, TimeSignature), ...]
    """
    new_score = stream.Score()
    new_part = stream.Part()

    # metadatos musicales útiles
    if carry_metadata:
        first_tempo = part.flatten().getElementsByClass(tempo.MetronomeMark).first()
        first_key = part.flatten().getElementsByClass(key.KeySignature).first()
        if first_tempo is not None:
            new_part.insert(0, copy.deepcopy(first_tempo))
        if first_key is not None:
            new_part.insert(0, copy.deepcopy(first_key))

    events = list(part.flatten().notesAndRests)
    if not events:
        new_score.append(new_part)
        return new_score

    total_length = max(float(el.offset + el.quarterLength) for el in events)

    measure_number = 1
    cursor = 0.0

    # anacrusa opcional muy simple
    initial_ts = get_ts_for_offset(ts_map, 0.0)
    initial_bar_len = float(initial_ts.barDuration.quarterLength)

    if infer_anacrusis and events:
        first_onset = float(events[0].offset)
        if 0 < first_onset < initial_bar_len:
            m = stream.Measure(number=measure_number)
            m.timeSignature = copy.deepcopy(initial_ts)
            measure_number += 1
            new_part.append(m)

    idx = 0
    while cursor < total_length + 1e-6:
        current_ts = get_ts_for_offset(ts_map, cursor)
        bar_len = float(current_ts.barDuration.quarterLength)

        m = stream.Measure(number=measure_number)
        measure_number += 1

        # insertar TS cuando cambia o en primer compás
        prev_ts = None if len(new_part.getElementsByClass(stream.Measure)) == 0 else \
            new_part.getElementsByClass(stream.Measure).last().timeSignature

        if prev_ts is None or prev_ts.ratioString != current_ts.ratioString:
            m.timeSignature = copy.deepcopy(current_ts)

        measure_start = cursor
        measure_end = cursor + bar_len

        while idx < len(events):
            el = events[idx]
            el_start = float(el.offset)

            if el_start >= measure_end - 1e-9:
                break

            if el_start < measure_start - 1e-9:
                idx += 1
                continue

            local_offset = el_start - measure_start
            remaining_in_measure = measure_end - el_start
            dur = float(el.quarterLength)

            new_el = copy.deepcopy(el)

            # cortar evento si atraviesa el compás
            if dur > remaining_in_measure + 1e-9:
                new_el.quarterLength = remaining_in_measure
                # actualizar el original restante para compás siguiente
                events[idx].quarterLength = dur - remaining_in_measure
                events[idx].offset = measure_end
            else:
                idx += 1

            m.insert(local_offset, new_el)

        new_part.append(m)
        cursor = measure_end

    new_score.append(new_part)
    return new_score


def clean_empty_trailing_measures(score):
    """
    Elimina compases finales vacíos si los hubiera.
    """
    parts = score.parts
    if not parts:
        return score

    for p in parts:
        measures = list(p.getElementsByClass(stream.Measure))
        while measures:
            last_m = measures[-1]
            if len(last_m.notesAndRests) == 0:
                p.remove(last_m)
                measures.pop()
            else:
                break
    return score


def midi_to_musicxml_with_inferred_meter(
    midi_path,
    output_path,
    allowed_durations=None,
    candidate_time_signatures=None,
    use_existing_ts=True,
    infer_ts_when_missing=True,
    infer_anacrusis=False,
    verbose=True
):
    """
    Convierte un MIDI a MusicXML intentando reconstruir la métrica.

    Parámetros
    ----------
    midi_path : str
    output_path : str
    allowed_durations : iterable[float] | None
        Duraciones permitidas para cuantización.
    candidate_time_signatures : list[str] | None
        Candidatos para inferencia global.
    use_existing_ts : bool
        Si True, usa TimeSignature del MIDI cuando existan.
    infer_ts_when_missing : bool
        Si True, infiere una TS global cuando no existan TS embebidas.
    infer_anacrusis : bool
        Inferencia simple de anacrusa.
    verbose : bool
    """
    if allowed_durations is None:
        allowed_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    if candidate_time_signatures is None:
        candidate_time_signatures = ["2/4", "3/4", "4/4", "6/8", "9/8", "12/8"]

    score = converter.parse(midi_path)

    existing_ts = get_existing_time_signatures(score) if use_existing_ts else []

    flat_part = strip_to_single_part(score)
    quantized_part = quantize_part(flat_part, allowed_durations=allowed_durations)

    if existing_ts:
        fallback_ts = existing_ts[0][1]
        if verbose:
            ts_strs = ", ".join([f"{off:.3f}:{ts.ratioString}" for off, ts in existing_ts])
            print(f"[TS embebidas] {os.path.basename(midi_path)} -> {ts_strs}")
    else:
        if infer_ts_when_missing:
            fallback_ts = infer_global_time_signature(
                quantized_part,
                candidate_time_signatures=candidate_time_signatures
            )
            if verbose:
                print(f"[TS inferida] {os.path.basename(midi_path)} -> {fallback_ts.ratioString}")
        else:
            fallback_ts = meter.TimeSignature("4/4")
            if verbose:
                print(f"[TS default] {os.path.basename(midi_path)} -> 4/4")

    total_length = 0.0
    for el in quantized_part.flatten().notesAndRests:
        total_length = max(total_length, float(el.offset + el.quarterLength))

    ts_map = make_ts_map(existing_ts, fallback_ts, total_length)

    rebuilt = rebuild_score_with_measures(
        quantized_part,
        ts_map=ts_map,
        carry_metadata=True,
        infer_anacrusis=infer_anacrusis
    )

    rebuilt = clean_empty_trailing_measures(rebuilt)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    rebuilt.write("musicxml", fp=output_path)
    return rebuilt


def midi_folder_to_musicxml(
    input_folder,
    output_folder,
    allowed_durations=None,
    candidate_time_signatures=None,
    recursive=False,
    verbose=True,
    corpus_name=None
):
    """
    Recorre una carpeta de MIDI y los convierte a MusicXML.
    """
    if allowed_durations is None:
        allowed_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    if candidate_time_signatures is None:
        candidate_time_signatures = ["2/4", "3/4", "4/4", "6/8", "9/8", "12/8"]

    os.makedirs(output_folder, exist_ok=True)

    converted = 0
    failed = []

    if recursive:
        walker = []
        for root, _, files in os.walk(input_folder):
            for fname in files:
                walker.append((root, fname))
    else:
        walker = [(input_folder, fname) for fname in os.listdir(input_folder)]

    for root, fname in walker:
        if not fname.lower().endswith((".mid", ".midi")):
            continue

        in_path = os.path.join(root, fname)

        rel_root = os.path.relpath(root, input_folder)
        out_subdir = output_folder if rel_root == "." else os.path.join(output_folder, rel_root)
        os.makedirs(out_subdir, exist_ok=True)

        if corpus_name:
            out_name = corpus_name + os.path.splitext(fname)[0] + ".xml"
        else:
            out_name = os.path.splitext(fname)[0] + ".xml"

        out_path = os.path.join(out_subdir, out_name)

        try:
            midi_to_musicxml_with_inferred_meter(
                midi_path=in_path,
                output_path=out_path,
                allowed_durations=allowed_durations,
                candidate_time_signatures=candidate_time_signatures,
                use_existing_ts=True,
                infer_ts_when_missing=True,
                infer_anacrusis=False,
                verbose=verbose
            )
            converted += 1

        except Exception as e:
            failed.append((in_path, str(e)))
            if verbose:
                print(f"[ERROR] {in_path}: {e}")

    if verbose:
        print(f"\nConvertidos: {converted}")
        print(f"Errores: {len(failed)}")
        if failed:
            for path, err in failed:
                print(f" - {path}: {err}")

    return {
        "converted": converted,
        "failed": failed
    }