
from music21 import stream, note, meter, duration, metadata, key
from datetime import datetime


def filer_tokens(tokens, patrones, ignore_case=True):
    """
    Elimina de 'tokens' los elementos que CONTENGAN cualquiera de las subcadenas en 'patrones'.
    - tokens: lista de strings
    - patrones: lista de strings a bloquear (p. ej. ["BAR", "MOD"])
    - ignore_case: si True, compara sin distinguir mayúsculas/minúsculas
    """
    if ignore_case:
        patrones_norm = [p.lower() for p in patrones]
        def contiene_patron(t):
            tl = t.lower()
            return any(p in tl for p in patrones_norm)
    else:
        def contiene_patron(t):
            return any(p in t for p in patrones)

    return [t for t in tokens if not contiene_patron(t)]

def tokens_to_music21_stream_with_ts(tokens, time_signature, allowed_durations):
    s = stream.Part()

    # Añadir compás al principio
    ts = meter.TimeSignature(time_signature)
    s.append(ts)

    #Eliminamos los tokens innecesarios
    tokens = filer_tokens(tokens, ["BAR", "BEAT", "MOD", "TIE"], ignore_case=True)

    i = 0
    while i < len(tokens):
        try:
            if tokens[i].startswith("NOTE_ON") and i + 1 < len(tokens) and tokens[i+1].startswith("DUR_"):
                pitch = int(tokens[i].split("_")[-1])
                dur = float(eval(tokens[i+1].split("_")[-1]))
                if dur not in allowed_durations:
                    i += 2
                    continue
                n = note.Note(pitch)
                n.duration = duration.Duration(dur)
                if n.pitch.accidental is not None:
                    n.pitch.accidental.displayStatus = False  # oculta el becuadro innecesario
                s.append(n)
                i += 2
            elif tokens[i] == "REST" and i + 1 < len(tokens) and tokens[i+1].startswith("DUR_"):
                dur = float(eval(tokens[i+1].split("_")[-1]))
                if dur not in allowed_durations:
                    i += 2
                    continue
                r = note.Rest()
                r.duration = duration.Duration(dur)
                s.append(r)
                i += 2
            else:
                i += 1
        except:
            i += 1
    return s

def tokens_to_music21_stream(tokens, allowed_durations, verbose_warnings=True):
    """
    Convierte una secuencia de tokens en un music21.stream.Score.

    Tokens soportados:
        - START, END, PAD     -> ignorados
        - BAR                 -> crea nuevo compás
        - MODE_*              -> ignorado
        - TS_X/Y              -> actualiza compás
        - BEAT_*              -> ignorado
        - NOTE_ON_XX          -> debe ir seguido de DUR_X
        - REST                -> debe ir seguido de DUR_X
        - DUR_X               -> duración
        - DUR_0               -> grace note

    Devuelve:
        music21.stream.Score
    """

    score = stream.Score()
    score.append(key.Key('C'))
    part = stream.Part()


    current_measure = None
    measure_num = 1
    current_ts = None
    last_ts = None
    measure_duration_target = 0.0 #float(current_ts.barDuration.quarterLength)
    measure_duration_accum = 0.0

    pending_event = None  # ("note", Note) o ("rest", Rest)

    def warn(msg):
        if verbose_warnings:
            print(f"[WARN] {msg}")

    def normalize_allowed_durations(values):
        if values is None:
            warn("allowed_durations es None; no se validarán duraciones.")
            return set()

        # --- NUEVO: soportar objeto AllowedDurations ---
        if hasattr(values, "durations"):
            values = values.durations
        elif hasattr(values, "values"):
            values = values.values

        if isinstance(values, str):
            warn("allowed_durations es string; no se validarán duraciones.")
            return set()

        try:
            return {float(v) for v in values}
        except Exception as e:
            warn(f"Error normalizando allowed_durations: {e}")
            return set()
    allowed_durations_set = normalize_allowed_durations(allowed_durations)

    def close_current_measure_if_needed():
        nonlocal current_measure, measure_duration_accum

        if current_measure is not None:
            if abs(measure_duration_accum - measure_duration_target) > 1e-4:
                if measure_duration_accum < measure_duration_target:
                    warn(f"Compás incompleto: {measure_duration_accum:.3f} / {measure_duration_target:.3f}")
                else:
                    warn(f"Compás excedido: {measure_duration_accum:.3f} / {measure_duration_target:.3f}")

    def start_new_measure():
        nonlocal current_measure, measure_duration_accum, last_ts, measure_num

        close_current_measure_if_needed()

        current_measure = stream.Measure(number=measure_num)
        measure_num+=1
        #Cambiamos el TS solo cuando cambia el ts
        if last_ts is None or current_ts.ratioString != last_ts.ratioString:
            current_measure.timeSignature = meter.TimeSignature(current_ts.ratioString)
            last_ts = current_ts
        measure_duration_accum = 0.0

    for i, tok in enumerate(tokens):

        # --- IGNORAR TOKENS ESPECIALES ---
        if tok in {"START", "END", "PAD"}:
            continue

        # --- CONTROL DE ESTADO (evento → DUR) ---
        if pending_event is not None and not (isinstance(tok, str) and tok.startswith("DUR_")):
            warn(f"Se esperaba DUR tras evento, pero se encontró '{tok}' en posición {i}")
            pending_event = None

        # --- BAR ---
        if tok == "BAR":
            start_new_measure()
            part.append(current_measure)
            continue

        # --- TIME SIGNATURE ---
        if isinstance(tok, str) and tok.startswith("TS_"):
            try:
                ts_str = tok.replace("TS_", "", 1)
                current_ts = meter.TimeSignature(ts_str)
                measure_duration_target = float(current_ts.barDuration.quarterLength)
            except Exception as e:
                warn(f"Time signature inválido '{tok}': {e}")
            continue

        # --- IGNORADOS ---
        if isinstance(tok, str) and (tok.startswith("MODE_") or tok.startswith("BEAT_")):
            continue

        # --- NOTE ---
        if isinstance(tok, str) and tok.startswith("NOTE_ON_"):
            try:
                midi = int(tok.replace("NOTE_ON_", "", 1))
                n = note.Note(midi)
                n.pitch.accidental.displayStatus = None
                n.pitch.accidental.displayType = 'normal'
                pending_event = ("note", n)
            except Exception as e:
                warn(f"NOTE_ON inválido '{tok}': {e}")
                pending_event = None
            continue

        # --- REST ---
        if tok == "REST":
            pending_event = ("rest", note.Rest())
            continue

        # --- DUR ---
        if isinstance(tok, str) and tok.startswith("DUR_"):

            if pending_event is None:
                warn(f"DUR sin evento previo: '{tok}' en posición {i}")
                continue

            try:
                dur_val = float(tok.replace("DUR_", "", 1))
            except Exception as e:
                warn(f"DUR inválido '{tok}': {e}")
                pending_event = None
                continue

            if dur_val != 0.0 and allowed_durations_set and dur_val not in allowed_durations_set:
                warn(f"Duración no permitida: {dur_val}")

            if current_measure is None:
                start_new_measure()
                part.append(current_measure)

            ev_type, ev_obj = pending_event

            # --- GRACE NOTE ---
            if dur_val == 0.0:
                ev_obj.duration = duration.Duration(0.0)
                gn = note.Note(ev_obj.pitch, type='16th').getGrace()
                gn.duration.slash = True  # barra diagonal
                # ev_obj = ev_obj.getGrace()
                current_measure.append(gn)
                pending_event = None
                continue

            # --- NORMAL ---
            ev_obj.duration = duration.Duration(dur_val)
            current_measure.append(ev_obj)

            measure_duration_accum += dur_val

            if measure_duration_accum - measure_duration_target > 1e-4:
                warn(f"Overflow inmediato en compás: {measure_duration_accum:.3f} / {measure_duration_target:.3f}")

            pending_event = None
            continue

        # --- DESCONOCIDO ---
        warn(f"Token desconocido: '{tok}'")

    # --- EVENTO COLGANTE ---
    if pending_event is not None:
        warn("La secuencia termina con evento sin DUR asociado")

    close_current_measure_if_needed()




    # ---- AÑADIR AL SCORE ----
    score.append(part)

    # METADATA
    score.insert(0, metadata.Metadata())
    score.metadata.title = "Generated Melody"
    score.metadata.composer = "TransFolk"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score.metadata.movementName = f"Generated on {now}"
    score.metadata.composer = "TransFolk"

    return score