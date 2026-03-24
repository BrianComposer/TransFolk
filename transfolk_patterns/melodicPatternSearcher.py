from collections import Counter

# ============================================================
# 1. UTILIDADES BÁSICAS
# ============================================================

def get_note_tokens(vocab: dict) -> dict:
    """Devuelve los tokens NOTE_ON del vocabulario."""
    return {k: v for k, v in vocab.items() if k.startswith("NOTE_ON")}


def extract_pitch_sequence(seq, vocab):
    """Devuelve una lista de alturas absolutas de NOTE_ON en la secuencia."""
    inv_vocab = {v: int(k.split("_")[2]) for k, v in vocab.items() if k.startswith("NOTE_ON")}
    return [inv_vocab[tok] for tok in seq if tok in inv_vocab]


def sequence_to_intervals(seq, vocab):
    """Convierte una secuencia en lista de intervalos (Δpitch) ignorando REST."""
    pitches = extract_pitch_sequence(seq, vocab)
    if len(pitches) < 2:
        return []
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


# ============================================================
# 2. DETECCIÓN DE PATRONES MELÓDICOS POR INTERVALOS
# ============================================================

def count_melodic_patterns(sequences, vocab, n_min=3, n_max=8):
    """
    Extrae n-gramas de intervalos melódicos y cuenta sus frecuencias.
    Devuelve lista [{"n": n, "counts": {...}}, ...].
    """
    results = []
    for n in range(n_max, n_min - 1, -1):
        counter = Counter()
        for seq in sequences:
            intervals = sequence_to_intervals(seq, vocab)
            if len(intervals) < n:
                continue
            for i in range(len(intervals) - n + 1):
                ngram = tuple(intervals[i:i + n])
                counter[ngram] += 1
        results.append({"n": n, "counts": dict(counter)})
    return results


def show_top_melodic_patterns(pattern_counts, top=20):
    """Muestra las combinaciones de intervalos más frecuentes."""
    for entry in sorted(pattern_counts, key=lambda x: x["n"], reverse=True):
        n = entry["n"]
        sorted_counts = sorted(entry["counts"].items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n=== Motivos melódicos de {n} intervalos ===")
        for combo, freq in sorted_counts[:top]:
            print(f"{combo}: {freq}")


# ============================================================
# 3. ANOTACIÓN JERÁRQUICA (melódicos dentro de rítmicos)
# ============================================================

def annotate_melodic_patterns(sequences, vocab, patterns, min_count=10):
    """
    Inserta <MEL_Mxxx_START> ... <MEL_Mxxx_END> dentro de las secuencias,
    priorizando los patrones más largos y sólo los que superan min_count.
    Las marcas se insertan dentro de los límites de ritmo si existen.
    """
    annotated_sequences = []
    new_vocab = vocab.copy()

    # Filtrado por frecuencia
    filtered_patterns = {
        k: v["pattern"] for k, v in patterns.items() if v.get("count", 0) >= min_count
    }

    # Orden descendente por longitud
    ordered_patterns = sorted(filtered_patterns.items(), key=lambda kv: len(kv[1]), reverse=True)

    # Añadir nuevos tokens al vocabulario
    for motif_name, _ in ordered_patterns:
        for tag in (f"<{motif_name}_START>", f"<{motif_name}_END>"):
            if tag not in new_vocab:
                new_vocab[tag] = max(new_vocab.values()) + 1

    note_ids = {v for k, v in vocab.items() if k.startswith("NOTE_ON")}

    for seq in sequences:
        new_seq = []
        i = 0
        # extraer alturas para referencia
        inv_vocab = {v: int(k.split("_")[2]) for k, v in vocab.items() if k.startswith("NOTE_ON")}
        note_positions = [idx for idx, tok in enumerate(seq) if tok in note_ids]
        pitches = [inv_vocab[tok] for tok in seq if tok in note_ids]

        # recorrer secuencia y buscar patrones
        while i < len(seq):
            tok = seq[i]
            if tok in note_ids:
                matched = False
                for motif_name, motif_intervals in ordered_patterns:
                    n = len(motif_intervals)
                    # extrae subsecuencia de notas a partir de posición i
                    start_note_idx = None
                    for k, pos in enumerate(note_positions):
                        if pos == i:
                            start_note_idx = k
                            break
                    if start_note_idx is None or start_note_idx + n >= len(pitches):
                        continue
                    intervals_local = [
                        pitches[start_note_idx + j + 1] - pitches[start_note_idx + j]
                        for j in range(n)
                    ]
                    if intervals_local == motif_intervals:
                        start_tok_id = new_vocab[f"<{motif_name}_START>"]
                        end_tok_id = new_vocab[f"<{motif_name}_END>"]
                        new_seq.append(start_tok_id)
                        # inserta tokens originales desde nota inicial hasta nota final del motivo
                        last_note_pos = note_positions[start_note_idx + n]
                        new_seq.extend(seq[i:last_note_pos + 1])
                        new_seq.append(end_tok_id)
                        i = last_note_pos + 1
                        matched = True
                        break
                if not matched:
                    new_seq.append(tok)
                    i += 1
            else:
                new_seq.append(tok)
                i += 1
        annotated_sequences.append(new_seq)

    return annotated_sequences, new_vocab


# ============================================================
# 4. PIPELINE COMPLETO DE DETECCIÓN Y ANOTACIÓN MELODICA
# ============================================================
#
# def searchMelodicPatterns(sequences, vocab, n_min=3, n_max=5, min_count=10, top_show=20, show=False):
#     """
#     Pipeline:
#       1. Detecta patrones melódicos mediante intervalos relativos.
#       2. Muestra los más frecuentes.
#       3. Crea diccionario de motivos con frecuencia >= min_count.
#       4. Inserta anotaciones jerárquicas <MEL_*> en el corpus.
#     """
#     pattern_counts = count_melodic_patterns(sequences, vocab, n_min, n_max)
#     if show:
#         show_top_melodic_patterns(pattern_counts, top_show)
#
#     # Crear diccionario de patrones frecuentes (tomando el n más largo)
#     top_counts = pattern_counts[0]["counts"]
#     patterns = {}
#     for i, (combo, freq) in enumerate(sorted(top_counts.items(), key=lambda kv: kv[1], reverse=True)):
#         if freq < min_count:
#             break
#         motif_name = f"MEL_M{i+1:03d}"
#         patterns[motif_name] = {"pattern": list(combo), "count": freq}
#
#     if show:
#         print(f"\n{len(patterns)} patrones melódicos seleccionados con >= {min_count} ocurrencias.")
#
#     annotated, new_vocab = annotate_melodic_patterns(sequences, vocab, patterns, min_count=min_count)
#
#     if show:
#         print("Corpus anotado con motivos melódicos jerárquicos.")
#     return annotated, new_vocab

def searchMelodicPatterns(sequences, vocab, n_min=3, n_max=5, min_count=10, top_show=20, show=True):
    try:
        pattern_counts = count_melodic_patterns(sequences, vocab, n_min, n_max)
        if show:
            show_top_melodic_patterns(pattern_counts, top_show)

        top_counts = pattern_counts[0]["counts"]
        patterns = {}
        for i, (combo, freq) in enumerate(sorted(top_counts.items(), key=lambda kv: kv[1], reverse=True)):
            if freq < min_count:
                break
            motif_name = f"MEL_M{i+1:03d}"
            patterns[motif_name] = {"pattern": list(combo), "count": freq}

        if show:
            print(f"\n{len(patterns)} patrones melódicos seleccionados con >= {min_count} ocurrencias.")

        annotated, new_vocab = annotate_melodic_patterns(sequences, vocab, patterns, min_count=min_count)

        if show:
            print("Corpus anotado con motivos melódicos jerárquicos.")

        return annotated, new_vocab

    except Exception as e:
        print(f"Error en searchMelodicPatterns: {e}")
        return None, None

