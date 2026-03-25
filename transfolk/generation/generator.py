# ------------------------------
# generation/generator.py
# ------------------------------
import torch
import torch.nn.functional as F
from datetime import datetime
from music21 import stream, note, meter, duration, instrument, metadata


def sample_next_token(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def sample_next_token_top_k_p(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    Selecciona el siguiente token usando sampling con temperatura,
    top-k y/o nucleus (top-p) sampling.

    Args:
        logits: Tensor con logits del modelo (vocab_size,).
        temperature: Factor de suavizado (bajo = más determinista).
        top_k: Mantener solo los k tokens más probables (0 = desactivado).
        top_p: Mantener tokens cuya prob acumulada <= p (0.0 = desactivado).

    Returns:
        Índice del token seleccionado.
    """
    # Escalar por temperatura
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # --- Top-k sampling ---
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))  # evitar errores
        values, indices = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs).scatter_(-1, indices, values)
        probs = probs / probs.sum()  # renormalizar

    # --- Top-p (nucleus) sampling ---
    if top_p > 0.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # filtrar tokens fuera de la masa acumulada p
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_probs[cutoff] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum()  # renormalizar

    # muestrear token
    return torch.multinomial(probs, num_samples=1).item()

def sample_next_token_top_k_p_penalty(
    logits,
    temperature=1.0,
    generated=None,
    repetition_penalty=1.2,
    top_k=0,
    top_p=0.0,
    greedy=False
):
    # --- GREEDY DECODING ---
    if greedy:
        # logits, no probs: equivalente a temperature=0
        return torch.argmax(logits).item()

    # 1. aplicar temperatura
    probs = F.softmax(logits / temperature, dim=-1)

    # 2. aplicar penalización de repetición
    if generated is not None and repetition_penalty > 1.0:
        for token_id in set(generated):
            if token_id < len(probs):
                probs[token_id] /= repetition_penalty
        probs = probs / probs.sum()  # renormalizar

    # 3. aplicar top-k
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k)
        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(0, indices, values)
        probs = probs_filtered / probs_filtered.sum()

    # 4. aplicar top-p (nucleus sampling)
    if top_p > 0.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mantener solo hasta que sume >= top_p
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_probs[cutoff] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)
        probs = probs / probs.sum()

    # 5. muestreo final
    return torch.multinomial(probs, num_samples=1).item()


def generate_sequence(model, start_token_id, max_len, vocab, inv_vocab, device, temperature, top_k=25, top_p=0.9, penalty=1.1 ):
    model.eval()
    generated = [start_token_id]
    for _ in range(max_len):
        input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
        next_token_logits = output[0, -1, :]
        #next_token = sample_next_token(next_token_logits, temperature)
        # next_token = sample_next_token_top_k_p(
        #     next_token_logits,
        #     temperature=TEMPERATURE,
        #     top_k=15,  # limita a los k tokens más probables
        #     top_p=0.9  # nucleus sampling al 90%
        # )
        next_token = sample_next_token_top_k_p_penalty(
            next_token_logits,
            temperature=temperature,
            generated=generated[-20:],  # solo últimas 20 notas
            repetition_penalty=penalty,
            top_k=top_k,
            top_p=top_p
        )

        generated.append(next_token)
        if next_token == vocab["END"]:
            break
    return [inv_vocab.get(t, "UNK") for t in generated]

def generate_sequence_from_prompt(model, start_token_id_list, max_len, vocab, inv_vocab, device, temperature, top_k=25, top_p=0.9, penalty=1.1):
    model.eval()
    start_id = vocab["START"]

    # Comprobación de que la lista de prompt tokens contiene START
    if not start_token_id_list or start_token_id_list[0] != start_id:
        generated = [start_id] + list(start_token_id_list)
    else:
        generated = list(start_token_id_list)
    for _ in range(max_len):
        input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
        next_token_logits = output[0, -1, :]

        next_token = sample_next_token_top_k_p_penalty(
            next_token_logits,
            temperature=temperature,
            generated=generated[-20:],  # solo últimas 20 notas
            repetition_penalty=penalty,
            top_k=top_k,
            top_p=top_p
        )

        generated.append(next_token)
        if next_token == vocab["END"]:
            break
    return [inv_vocab.get(t, "UNK") for t in generated]

#
#
# def filer_tokens(tokens, patrones, ignore_case=True):
#     """
#     Elimina de 'tokens' los elementos que CONTENGAN cualquiera de las subcadenas en 'patrones'.
#     - tokens: lista de strings
#     - patrones: lista de strings a bloquear (p. ej. ["BAR", "MOD"])
#     - ignore_case: si True, compara sin distinguir mayúsculas/minúsculas
#     """
#     if ignore_case:
#         patrones_norm = [p.lower() for p in patrones]
#         def contiene_patron(t):
#             tl = t.lower()
#             return any(p in tl for p in patrones_norm)
#     else:
#         def contiene_patron(t):
#             return any(p in t for p in patrones)
#
#     return [t for t in tokens if not contiene_patron(t)]
#
# def tokens_to_music21_stream_with_ts(tokens, time_signature, allowed_durations):
#     s = stream.Part()
#
#     # Añadir compás al principio
#     ts = meter.TimeSignature(time_signature)
#     s.append(ts)
#
#     #Eliminamos los tokens innecesarios
#     tokens = filer_tokens(tokens, ["BAR", "BEAT", "MOD", "TIE"], ignore_case=True)
#
#     i = 0
#     while i < len(tokens):
#         try:
#             if tokens[i].startswith("NOTE_ON") and i + 1 < len(tokens) and tokens[i+1].startswith("DUR_"):
#                 pitch = int(tokens[i].split("_")[-1])
#                 dur = float(eval(tokens[i+1].split("_")[-1]))
#                 if dur not in allowed_durations:
#                     i += 2
#                     continue
#                 n = note.Note(pitch)
#                 n.duration = duration.Duration(dur)
#                 if n.pitch.accidental is not None:
#                     n.pitch.accidental.displayStatus = False  # oculta el becuadro innecesario
#                 s.append(n)
#                 i += 2
#             elif tokens[i] == "REST" and i + 1 < len(tokens) and tokens[i+1].startswith("DUR_"):
#                 dur = float(eval(tokens[i+1].split("_")[-1]))
#                 if dur not in allowed_durations:
#                     i += 2
#                     continue
#                 r = note.Rest()
#                 r.duration = duration.Duration(dur)
#                 s.append(r)
#                 i += 2
#             else:
#                 i += 1
#         except:
#             i += 1
#     return s
#
# def tokens_to_music21_stream(tokens, allowed_durations, verbose_warnings=True):
#     """
#     Convierte una secuencia de tokens en un music21.stream.Score.
#
#     Tokens soportados:
#         - START, END, PAD     -> ignorados
#         - BAR                 -> crea nuevo compás
#         - MODE_*              -> ignorado
#         - TS_X/Y              -> actualiza compás
#         - BEAT_*              -> ignorado
#         - NOTE_ON_XX          -> debe ir seguido de DUR_X
#         - REST                -> debe ir seguido de DUR_X
#         - DUR_X               -> duración
#         - DUR_0               -> grace note
#
#     Devuelve:
#         music21.stream.Score
#     """
#
#     score = stream.Score()
#     part = stream.Part()
#
#
#     current_measure = None
#     current_ts = meter.TimeSignature("4/4")
#     measure_duration_target = float(current_ts.barDuration.quarterLength)
#     measure_duration_accum = 0.0
#
#     pending_event = None  # ("note", Note) o ("rest", Rest)
#
#     def warn(msg):
#         if verbose_warnings:
#             print(f"[WARN] {msg}")
#
#     def normalize_allowed_durations(values):
#         if values is None:
#             warn("allowed_durations es None; no se validarán duraciones.")
#             return set()
#
#         # --- NUEVO: soportar objeto AllowedDurations ---
#         if hasattr(values, "durations"):
#             values = values.durations
#         elif hasattr(values, "values"):
#             values = values.values
#
#         if isinstance(values, str):
#             warn("allowed_durations es string; no se validarán duraciones.")
#             return set()
#
#         try:
#             return {float(v) for v in values}
#         except Exception as e:
#             warn(f"Error normalizando allowed_durations: {e}")
#             return set()
#     allowed_durations_set = normalize_allowed_durations(allowed_durations)
#
#     def close_current_measure_if_needed():
#         nonlocal current_measure, measure_duration_accum
#
#         if current_measure is not None:
#             if abs(measure_duration_accum - measure_duration_target) > 1e-4:
#                 if measure_duration_accum < measure_duration_target:
#                     warn(f"Compás incompleto: {measure_duration_accum:.3f} / {measure_duration_target:.3f}")
#                 else:
#                     warn(f"Compás excedido: {measure_duration_accum:.3f} / {measure_duration_target:.3f}")
#
#     def start_new_measure():
#         nonlocal current_measure, measure_duration_accum
#
#         close_current_measure_if_needed()
#
#         current_measure = stream.Measure()
#         #current_measure.timeSignature = current_ts
#         current_measure.timeSignature = meter.TimeSignature(current_ts.ratioString)
#         measure_duration_accum = 0.0
#
#     for i, tok in enumerate(tokens):
#
#         # --- IGNORAR TOKENS ESPECIALES ---
#         if tok in {"START", "END", "PAD"}:
#             continue
#
#         # --- CONTROL DE ESTADO (evento → DUR) ---
#         if pending_event is not None and not (isinstance(tok, str) and tok.startswith("DUR_")):
#             warn(f"Se esperaba DUR tras evento, pero se encontró '{tok}' en posición {i}")
#             pending_event = None
#
#         # --- BAR ---
#         if tok == "BAR":
#             start_new_measure()
#             part.append(current_measure)
#             continue
#
#         # --- TIME SIGNATURE ---
#         if isinstance(tok, str) and tok.startswith("TS_"):
#             try:
#                 ts_str = tok.replace("TS_", "", 1)
#                 current_ts = meter.TimeSignature(ts_str)
#                 measure_duration_target = float(current_ts.barDuration.quarterLength)
#             except Exception as e:
#                 warn(f"Time signature inválido '{tok}': {e}")
#             continue
#
#         # --- IGNORADOS ---
#         if isinstance(tok, str) and (tok.startswith("MODE_") or tok.startswith("BEAT_")):
#             continue
#
#         # --- NOTE ---
#         if isinstance(tok, str) and tok.startswith("NOTE_ON_"):
#             try:
#                 midi = int(tok.replace("NOTE_ON_", "", 1))
#                 pending_event = ("note", note.Note(midi))
#             except Exception as e:
#                 warn(f"NOTE_ON inválido '{tok}': {e}")
#                 pending_event = None
#             continue
#
#         # --- REST ---
#         if tok == "REST":
#             pending_event = ("rest", note.Rest())
#             continue
#
#         # --- DUR ---
#         if isinstance(tok, str) and tok.startswith("DUR_"):
#
#             if pending_event is None:
#                 warn(f"DUR sin evento previo: '{tok}' en posición {i}")
#                 continue
#
#             try:
#                 dur_val = float(tok.replace("DUR_", "", 1))
#             except Exception as e:
#                 warn(f"DUR inválido '{tok}': {e}")
#                 pending_event = None
#                 continue
#
#             if dur_val != 0.0 and allowed_durations_set and dur_val not in allowed_durations_set:
#                 warn(f"Duración no permitida: {dur_val}")
#
#             if current_measure is None:
#                 start_new_measure()
#                 part.append(current_measure)
#
#             ev_type, ev_obj = pending_event
#
#             # --- GRACE NOTE ---
#             if dur_val == 0.0:
#                 ev_obj.duration = duration.Duration(0.0)
#                 ev_obj = ev_obj.getGrace()
#                 current_measure.append(ev_obj)
#                 pending_event = None
#                 continue
#
#             # --- NORMAL ---
#             ev_obj.duration = duration.Duration(dur_val)
#             current_measure.append(ev_obj)
#
#             measure_duration_accum += dur_val
#
#             if measure_duration_accum - measure_duration_target > 1e-4:
#                 warn(f"Overflow inmediato en compás: {measure_duration_accum:.3f} / {measure_duration_target:.3f}")
#
#             pending_event = None
#             continue
#
#         # --- DESCONOCIDO ---
#         warn(f"Token desconocido: '{tok}'")
#
#     # --- EVENTO COLGANTE ---
#     if pending_event is not None:
#         warn("La secuencia termina con evento sin DUR asociado")
#
#     close_current_measure_if_needed()
#
#
#
#
#     # ---- AÑADIR AL SCORE ----
#     score.append(part)
#
#     # METADATA
#     score.insert(0, metadata.Metadata())
#     score.metadata.title = "Generated Melody"
#     score.metadata.composer = "TransFolk"
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     score.metadata.movementName = f"Generated on {now}"
#     score.metadata.composer = "TransFolk"
#
#
#     return score