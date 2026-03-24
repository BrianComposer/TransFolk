# ------------------------------
# pattern/rhythmicPatternSearcher.py
# ------------------------------


# ==============================================================
# 1. TOKENS DE DURACIÓN
# ==============================================================

def get_duration_tokens(vocab: dict) -> dict:
    """Devuelve sólo las entradas del vocabulario que contienen 'DUR' en la clave."""
    return {k: v for k, v in vocab.items() if "DUR" in k}


# ==============================================================
# 2. GENERAR TODAS LAS COMBINACIONES DE TOKENS DUR
# ==============================================================

def generate_duration_combinations(vocab, n_min=3, n_max=8):
    """
    Genera todas las combinaciones posibles de los tokens de duración
    (ordenadas y con repetición) entre longitudes n_max → n_min.
    """
    dur_tokens = list(get_duration_tokens(vocab).values())
    results = []
    for n in range(n_max, n_min - 1, -1):
        combos = list(itertools.product(dur_tokens, repeat=n))
        results.append({"n": n, "combinations": combos})
    return results


# ==============================================================
# 3. CONTAR OCURRENCIAS DE CADA COMBINACIÓN EN EL CORPUS
# ==============================================================

def count_duration_combinations_old(sequences, vocab, combinations):
    """
    Recorre las secuencias y cuenta cuántas veces aparece cada combinación de DUR,
    ignorando los tokens intercalados que no sean de duración.
    """
    dur_values = set(get_duration_tokens(vocab).values())
    results = []

    for entry in combinations:
        n = entry["n"]
        combos = entry["combinations"]
        counts = Counter()

        for seq in sequences:
            dur_seq = [tok for tok in seq if tok in dur_values]
            if len(dur_seq) < n:
                continue
            dur_ngrams = [tuple(dur_seq[i:i + n]) for i in range(len(dur_seq) - n + 1)]
            local_counter = Counter(dur_ngrams)
            for combo in combos:
                if combo in local_counter:
                    counts[combo] += local_counter[combo]

        results.append({"n": n, "counts": dict(counts)})

    return results

def count_duration_combinations(sequences, vocab, combinations):
    """
    1. Filtra cada secuencia dejando solo BAR y DUR.
    2. Separa por BAR y convierte cada fragmento en un compás formado solo por tokens DUR.
    3. Cuenta cuántas veces aparece cada combinación de DUR dentro de todos los compases.
    4. Devuelve para cada n un diccionario ordenado de mayor a menor frecuencia.
    """
    dur_values = set(get_duration_tokens(vocab).values())
    bar_token = 0
    try:
        bar_token = vocab["BAR"]
    except Exception as e:
        raise Exception("Vocabulary doesn't contain BAR token")

    results = []

    for entry in combinations:
        n = entry["n"]
        combos = entry["combinations"]

        counts = Counter()

        for seq in sequences:
            filtered = [tok for tok in seq if tok == bar_token or tok in dur_values]

            measures = []
            current = []

            for tok in filtered:
                if tok == bar_token:
                    if current:
                        measures.append(current)
                    current = []
                else:
                    current.append(tok)

            if current:
                measures.append(current)

            for m in measures:
                if len(m) < n:
                    continue
                ngrams = [tuple(m[i:i+n]) for i in range(len(m) - n + 1)]
                local_counter = Counter(ngrams)
                for combo in combos:
                    if combo in local_counter:
                        counts[combo] += local_counter[combo]

        ordered = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        results.append({"n": n, "counts": ordered})

    return results


# ==============================================================
# 4. MOSTRAR LOS PATRONES MÁS FRECUENTES
# ==============================================================

def show_top_combinations(ngram_counts, top=None):
    """Muestra las combinaciones detectadas ordenadas por frecuencia."""
    for entry in sorted(ngram_counts, key=lambda x: x["n"], reverse=True):
        n = entry["n"]
        counts = entry["counts"]
        sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if top:
            sorted_counts = sorted_counts[:top]
        print(f"\n=== N-gramas de longitud {n} ===")
        for combo, freq in sorted_counts:
            print(f"{combo}: {freq}")


# ==============================================================
# 5. ANOTAR PATRONES RÍTMICOS FRECUENTES EN EL CORPUS
# ==============================================================

def annotate_rhythm_patterns(sequences, vocab, patterns, min_count=1):
    """
    Inserta etiquetas <RHY_Mxxx_START> … <RHY_Mxxx_END> alrededor de los patrones rítmicos detectados.
    Prioriza los de mayor longitud y sólo anota los que aparecen >= min_count veces.
    """
    annotated_sequences = []
    new_vocab = vocab.copy()

    # Filtrar patrones por frecuencia
    filtered_patterns = {
        k: v["pattern"] for k, v in patterns.items() if v.get("count", 0) >= min_count
    }

    # Ordenar por longitud descendente
    ordered_patterns = sorted(filtered_patterns.items(), key=lambda kv: len(kv[1]), reverse=True)

    # Añadir tokens de inicio/fin al vocabulario
    for motif_name, _ in ordered_patterns:
        for tag in (f"<{motif_name}_START>", f"<{motif_name}_END>"):
            if tag not in new_vocab:
                new_vocab[tag] = max(new_vocab.values()) + 1

    dur_ids = {v for k, v in vocab.items() if "DUR" in k}

    for seq in sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            tok = seq[i]
            if tok in dur_ids:
                matched = False
                for motif_name, motif_pattern in ordered_patterns:
                    n = len(motif_pattern)
                    dur_seq = []
                    j = i
                    while j < len(seq) and len(dur_seq) < n:
                        if seq[j] in dur_ids:
                            dur_seq.append(seq[j])
                        j += 1
                    if dur_seq == motif_pattern:
                        start_tok_id = new_vocab[f"<{motif_name}_START>"]
                        end_tok_id = new_vocab[f"<{motif_name}_END>"]
                        new_seq.append(start_tok_id)
                        new_seq.extend(seq[i:j])
                        new_seq.append(end_tok_id)
                        i = j
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




def estimate_min_occurrences(T, abs_min=100, rel_factor=0.001):
    """
    Calcula el umbral N_min = max(abs_min, rel_factor * T)
    donde:
        T = número total de tokens de duración en el corpus
        abs_min = mínimo absoluto (por defecto 100)
        rel_factor = mínimo relativo (por defecto 0.001)
    """
    rel_min = int(T * rel_factor)
    return max(abs_min, rel_min)


def select_relevant_patterns(all_counts, sequences, vocab,
                             abs_min=100, rel_factor=0.001):
    """
    Filtra los patrones rítmicos manteniendo la separación por longitud.
    Usa la regla:
        n_i >= N_min  donde N_min = max(100, 0.001*T)

    all_counts: salida de count_duration_combinations
                formato:
                [
                  {"n": n, "counts": { combo: freq, ... }},
                  ...
                ]
    """
    dur_values = set(get_duration_tokens(vocab).values())

    # Cálculo de T: total de tokens de duración en el corpus
    T = 0
    for seq in sequences:
        T += sum(1 for tok in seq if tok in dur_values)

    N_min = estimate_min_occurrences(T, abs_min=abs_min, rel_factor=rel_factor)

    filtered = []
    for entry in all_counts:
        n = entry["n"]
        counts = entry["counts"]

        kept = {combo: c for combo, c in counts.items() if c >= N_min}

        filtered.append({"n": n, "counts": kept})

    return filtered


from collections import Counter
import itertools

def annotate_rhythm_patterns_no_overlap(sequences, vocab, patterns):
    """
    Para cada secuencia:
      1) Identifica cada compás (BAR).
      2) Dentro del compás extrae DUR en orden.
      3) Recorre el compás de izquierda a derecha.
      4) Para cada posición, prueba patrones desde el más largo al más corto.
      5) Si coincide un patrón:
           - Obtiene el rango real del patrón dentro del compás.
           - Comprueba que ese rango no contiene previamente tokens OPEN/CLOSE.
           - Si está limpio: inserta OPEN + tokens originales + CLOSE.
           - Avanza el cursor hasta el final del patrón.
      6) Si no coincide ningún patrón: copia el token original.
    Sin solapamientos. Sin anidaciones.
    """

    # ----------------------------------------------------------------------
    # 1. Ampliar vocabulario con tokens OPEN / CLOSE
    # ----------------------------------------------------------------------
    new_vocab = dict(vocab)
    cur = max(new_vocab.values())

    dur_tokens = set(get_duration_tokens(vocab).values())
    bar = vocab["BAR"]

    for name, pdata in patterns.items():
        cur += 1
        open_tok = cur
        cur += 1
        close_tok = cur

        new_vocab[f"{name}_OPEN"] = open_tok
        new_vocab[f"{name}_CLOSE"] = close_tok

        pdata["open"] = open_tok
        pdata["close"] = close_tok

    # ----------------------------------------------------------------------
    # 2. Preparar lista de patrones de mayor a menor longitud
    # ----------------------------------------------------------------------
    pat_list = []
    for name, pdata in patterns.items():
        p = pdata["pattern"]
        Lp = len(p)
        pat_list.append((Lp, p, pdata["open"], pdata["close"]))
    pat_list.sort(key=lambda x: x[0], reverse=True)

    # ----------------------------------------------------------------------
    # 3. Función auxiliar para comprobar si un rango contiene OPEN/CLOSE
    # ----------------------------------------------------------------------
    def contains_pattern_tokens(seq, start, end):
        for t in seq[start:end+1]:
            for pdata in patterns.values():
                if t == pdata["open"] or t == pdata["close"]:
                    return True
        return False

    # ----------------------------------------------------------------------
    # 4. Procesamiento principal
    # ----------------------------------------------------------------------
    annotated = []

    for seq in sequences:
        L = len(seq)

        # Localizar compases mediante índices
        measure_ranges = []
        start_idx = 0
        for i, tok in enumerate(seq):
            if tok == bar:
                if start_idx <= i - 1:
                    measure_ranges.append((start_idx, i - 1))
                start_idx = i + 1
        if start_idx <= L - 1:
            measure_ranges.append((start_idx, L - 1))

        new_seq = []
        last_end = -1

        for m_start, m_end in measure_ranges:

            # Copiar todo lo anterior al compás
            for pos in range(last_end + 1, m_start):
                new_seq.append(seq[pos])

            # Tokens del compás
            comp = seq[m_start:m_end + 1]

            # Mapa de posiciones DUR dentro del compás: [(idx_global, tok), ...]
            dur_positions = [(j, seq[j]) for j in range(m_start, m_end + 1)
                             if seq[j] in dur_tokens]
            if not dur_positions:
                # Compás sin duraciones: copiar tal cual
                for pos in range(m_start, m_end + 1):
                    new_seq.append(seq[pos])
                last_end = m_end
                continue

            # Convertir a secuencia local DUR únicamente
            dur_seq = [tok for _, tok in dur_positions]

            # Cursor local en dur_seq
            k = 0

            # Reconstrucción del compás
            rebuilt = []
            # Mapeo inverso a posición original
            dur_indices = [idx for idx, _ in dur_positions]

            while k < len(dur_seq):
                matched = False

                # Probar patrones desde el más largo
                for Lp, pat, open_tok, close_tok in pat_list:
                    if k + Lp <= len(dur_seq) and dur_seq[k:k + Lp] == pat:
                        global_start = dur_indices[k]
                        global_end = dur_indices[k + Lp - 1]

                        # Comprobar ausencia previa de tokens de patrón
                        if not contains_pattern_tokens(seq, global_start, global_end):

                            # Insertar patrón
                            rebuilt.append(open_tok)
                            for pos in range(global_start, global_end + 1):
                                rebuilt.append(seq[pos])
                            rebuilt.append(close_tok)

                            k += Lp
                            matched = True
                            break

                if not matched:
                    # No coincide ningún patrón: copiar token DUR original
                    rebuilt.append(dur_seq[k])
                    k += 1

            # Ahora insertar reconstrucción del compás en orden real
            # No copiamos directamente comp, sino la versión procesada:

            # rebuilt contiene:
            #   - tokens OPEN
            #   - tokens DUR originales
            #   - tokens CLOSE
            # El compás original puede contener otro material no DUR dentro del rango.
            # Implementación mínima: sustituimos únicamente los DUR por rebuilt,
            # manteniendo los no-DUR en su posición aproximada.
            #
            # Necesitamos intercalar: reconstrucción DUR + tokens no DUR preservados.

            dur_iter = iter(rebuilt)
            for pos in range(m_start, m_end + 1):
                if seq[pos] in dur_tokens:
                    # poner siguiente token de rebuilt
                    new_seq.append(next(dur_iter))
                else:
                    new_seq.append(seq[pos])

            last_end = m_end

        # Copiar resto de la secuencia si queda
        for pos in range(last_end + 1, L):
            new_seq.append(seq[pos])

        annotated.append(new_seq)

    return annotated, new_vocab





def searchRhythmicPatterns_(sequences, vocab, n_min=3, n_max=5, min_count=10, show=True):
    try:
        dur_combos = generate_duration_combinations(vocab, n_min=n_min, n_max=n_max)
        counts = count_duration_combinations(sequences, vocab, dur_combos)
        relevant = select_relevant_patterns(counts, sequences, vocab)

        if show:
            show_top_combinations(counts, top=20)

        top_counts = counts[0]["counts"]
        patterns = {}
        for i, (combo, freq) in enumerate(sorted(top_counts.items(), key=lambda kv: kv[1], reverse=True)):
            if freq < min_count:
                break
            motif_name = f"RHY_M{i + 1:03d}"
            patterns[motif_name] = {"pattern": list(combo), "count": freq}

        if show:
            print(f"\n{len(patterns)} patrones seleccionados con >= {min_count} ocurrencias.")

        annotated, new_vocab = annotate_rhythm_patterns(sequences, vocab, patterns, min_count=min_count)

        if show:
            print("Corpus anotado y vocabulario extendido generados correctamente.")

        return annotated, new_vocab

    except Exception as e:
        print(f"Error en searchRhythmicPatterns: {e}")
        return None, None


def searchRhythmicPatterns(sequences, vocab,
                           n_min=3, n_max=5,
                           min_count=10, show=True):
    try:
        dur_combos = generate_duration_combinations(vocab, n_min=n_min, n_max=n_max)
        counts = count_duration_combinations(sequences, vocab, dur_combos)
        relevant = select_relevant_patterns(counts, sequences, vocab)

        if show:
            show_top_combinations(counts, top=20)

        patterns = {}
        for entry in relevant:
            n = entry["n"]
            for combo, freq in entry["counts"].items():
                motif_name = f"RHY_N{n}_M{len(patterns)+1:03d}"
                patterns[motif_name] = {"pattern": list(combo), "count": freq}

        if show:
            print(f"\n{len(patterns)} patrones relevantes seleccionados.")

        annotated, new_vocab = annotate_rhythm_patterns_no_overlap(
            sequences, vocab, patterns
        )

        if show:
            print("Corpus anotado sin solapamientos y con delimitadores.")

        return annotated, new_vocab

    except Exception as e:
        print(f"Error en searchRhythmicPatterns: {e}")
        return None, None
