from music21 import stream, note, meter,duration
# ==============================================
# VARIABLES GLOBALES
# ==============================================
TIME_SIGNATURE = "2/4"

secuencia = [1, 3, 4, 5, 6, 7, 6, 8, 9, 8, 3, 10, 5, 10, 7, 10, 8, 11, 7, 11, 7, 3, 11, 7, 10, 8, 10, 7, 10, 8, 10, 7, 11, 7, 3, 10, 5, 9, 7, 9, 8, 9, 7, 10, 7, 3, 6, 8, 4, 7, 10, 7, 11, 8, 10, 8, 3, 9, 7, 10, 7, 9, 7, 6, 7, 12, 13, 3, 12, 5, 6, 7, 9, 8, 10, 8, 3, 6, 7, 9, 7, 6, 7, 12, 7, 14, 13, 3, 14, 5, 15, 7, 15, 8, 15, 8, 3, 15, 13, 16, 13, 3, 16, 5, 16, 7, 15, 8, 17, 8, 3, 15, 18, 3, 15, 8, 4, 7, 19, 7, 19, 8, 17, 8, 3, 15, 8, 20, 13, 3, 20, 8, 9, 7, 10, 7, 20, 8, 15, 8, 3, 10, 18, 3, 10, 5, 9, 7, 10, 8, 20, 8, 3, 20, 13, 10, 13, 3, 20, 8, 9, 7, 10, 7, 20, 8, 21, 8, 3, 15, 13, 10, 13, 3, 10, 5, 9, 7, 22, 8, 12, 8, 3, 14, 18, 3, 14, 8, 4, 8, 4, 13, 2]

vocabulario = {"PAD": 0, "START": 1, "END": 2, "BAR": 3, "REST": 4, "DUR_0.75": 5, "NOTE_ON_64": 6, "DUR_0.25": 7, "DUR_0.5": 8, "NOTE_ON_65": 9, "NOTE_ON_67": 10, "NOTE_ON_69": 11, "NOTE_ON_62": 12, "DUR_1.0": 13, "NOTE_ON_60": 14, "NOTE_ON_72": 15, "NOTE_ON_71": 16, "NOTE_ON_74": 17, "DUR_2.0": 18, "NOTE_ON_75": 19, "NOTE_ON_68": 20, "NOTE_ON_70": 21, "NOTE_ON_63": 22}



# ==============================================
# FUNCIÓN DE RECONSTRUCCIÓN
# ==============================================
def reconstruir_partitura(secuencia, vocabulario):

    # Invertir el diccionario para mapear números -> tokens
    id_a_token = {v: k for k, v in vocabulario.items()}

    # Sustituir en la secuencia
    tokens = [id_a_token.get(num, f"UNK_{num}") for num in secuencia]


    s = stream.Score()
    part = stream.Part()

    # Añadir compás inicial con la métrica
    ts = meter.TimeSignature(TIME_SIGNATURE)
    part.append(ts)

    i = 0
    while i < len(tokens):
        try:
            if tokens[i].startswith("NOTE_ON") and i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                pitch = int(tokens[i].split("_")[-1])
                dur = float(eval(tokens[i + 1].split("_")[-1]))
                if dur not in ALLOWED_DURATIONS:
                    i += 2
                    continue
                n = note.Note(pitch)
                n.duration = duration.Duration(dur)
                part.append(n)
                i += 2
            elif tokens[i] == "REST" and i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
                dur = float(eval(tokens[i + 1].split("_")[-1]))
                if dur not in ALLOWED_DURATIONS:
                    i += 2
                    continue
                r = note.Rest()
                r.duration = duration.Duration(dur)
                part.append(r)
                i += 2
            else:
                i += 1
        except:
            i += 1
    s.append(part)
    return s


def test_beats_alignment(sequences_path: str, vocab_path: str, subdivision: float = 1.0):
    """
    Comprueba la correcta colocación de tokens de BEAT en las secuencias.
    Muestra para cada obra su índice, el porcentaje de beats correctos y los compases donde hay error.
    """
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}

    dur_tokens = {idx: float(token.split('_')[1]) for idx, token in inv_vocab.items() if 'DUR' in token}
    beat_tokens = {idx: token for idx, token in inv_vocab.items() if 'BEAT' in token}
    bar_token = next((idx for idx, token in inv_vocab.items() if token == 'BAR'), None)

    total_pieces = len(sequences)
    correct_pieces = 0
    total_measures = 0
    wrong_measures = 0

    for piece_idx, seq in enumerate(sequences):
        measure_dur = 0.0
        piece_measures = 0
        piece_wrong = 0
        measure_count = 1
        wrong_measure_list = []

        for token_idx in seq:
            token = inv_vocab[token_idx]

            # Nuevo compás
            if bar_token is not None and token_idx == bar_token:
                measure_dur = 0.0
                measure_count += 1
                continue

            if token_idx in dur_tokens:
                measure_dur += dur_tokens[token_idx]

            if token_idx in beat_tokens:
                piece_measures += 1
                total_measures += 1
                # comprobar duración acumulada
                if abs(measure_dur - subdivision) > 1e-6:
                    piece_wrong += 1
                    wrong_measures += 1
                    wrong_measure_list.append(measure_count)
                measure_dur = 0.0

        piece_correct_pct = 100 * (1 - piece_wrong / piece_measures) if piece_measures > 0 else 0
        if piece_wrong == 0:
            correct_pieces += 1

        print(f"Obra {piece_idx + 1}: {piece_correct_pct:.2f}% correcto", end='')
        if wrong_measure_list:
            print(f" | Compases erróneos: {wrong_measure_list}")
        else:
            print(" | Sin errores")

    pct_correct_pieces = 100 * correct_pieces / total_pieces
    pct_wrong_measures = 100 * wrong_measures / max(total_measures, 1)

    print("\nResumen general:")
    print(f"Obras correctamente tokenizadas: {pct_correct_pieces:.2f}%")
    print(f"Compases incorrectamente tokenizados: {pct_wrong_measures:.2f}%")


def test_beats_alignment2(sequences_path: str, vocab_path: str, subdivision: float = 1.0):
    """
    Comprueba la correcta colocación de tokens de BEAT en las secuencias
    y muestra cada obra reconstruida solo con los tokens BAR, BEAT y DUR.
    """
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}

    dur_tokens = {idx: float(token.split('_')[1]) for idx, token in inv_vocab.items() if 'DUR' in token}
    beat_tokens = {idx: token for idx, token in inv_vocab.items() if 'BEAT' in token}
    bar_token = next((idx for idx, token in inv_vocab.items() if token == 'BAR'), None)

    total_pieces = len(sequences)
    correct_pieces = 0
    total_measures = 0
    wrong_measures = 0

    for piece_idx, seq in enumerate(sequences):
        measure_dur = 0.0
        piece_measures = 0
        piece_wrong = 0
        measure_count = 1
        wrong_measure_list = []
        filtered_tokens = []  # para reconstruir solo BAR, BEAT y DUR

        for token_idx in seq:
            token = inv_vocab[token_idx]

            if bar_token is not None and token_idx == bar_token:
                filtered_tokens.append("BAR")
                measure_dur = 0.0
                measure_count += 1
                continue

            if token_idx in dur_tokens:
                filtered_tokens.append(f"DUR_{dur_tokens[token_idx]}")
                measure_dur += dur_tokens[token_idx]

            if token_idx in beat_tokens:
                filtered_tokens.append(token)
                piece_measures += 1
                total_measures += 1
                if abs(measure_dur - subdivision) > 1e-6:
                    piece_wrong += 1
                    wrong_measures += 1
                    wrong_measure_list.append(measure_count)
                measure_dur = 0.0

        piece_correct_pct = 100 * (1 - piece_wrong / piece_measures) if piece_measures > 0 else 0
        if piece_wrong == 0:
            correct_pieces += 1

        print(f"\nObra {piece_idx + 1}: {piece_correct_pct:.2f}% correcto", end='')
        if wrong_measure_list:
            print(f" | Compases erróneos: {wrong_measure_list}")
        else:
            print(" | Sin errores")

        print("Reconstrucción:")
        print(" ".join(filtered_tokens))

    pct_correct_pieces = 100 * correct_pieces / total_pieces
    pct_wrong_measures = 100 * wrong_measures / max(total_measures, 1)

    print("\nResumen general:")
    print(f"Obras correctamente tokenizadas: {pct_correct_pieces:.2f}%")
    print(f"Compases incorrectamente tokenizados: {pct_wrong_measures:.2f}%")


def reconstruir_y_comprobar_beats(sequences_path, vocab_path, subdivision=1.0):
    """
    Reconstruye las obras mostrando solo los tokens BAR, BEAT y DUR.
    Cada compás se muestra como BAR_X, donde X es el número de compás.
    Llama a check_beats_piece() y calcula el porcentaje global de obras con errores.
    """
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}

    total_obras = len(sequences)
    obras_con_errores = 0

    for i, seq in enumerate(sequences, start=1):
        filtered_tokens = []
        compas = 1

        for idx in seq:
            token = inv_vocab[idx]
            if "BAR" in token:
                filtered_tokens.append(f"BAR_{compas}")
                compas += 1
            elif "BEAT" in token or "DUR" in token:
                filtered_tokens.append(token)

        print(f"\n=== Obra {i} ===")
        print("Reconstrucción:")
        print(" ".join(filtered_tokens))
        print("\nComprobación de beats:")
        resultado = check_beats_piece(filtered_tokens, subdivision=subdivision)
        if not resultado:
            obras_con_errores += 1

    if total_obras > 0:
        porcentaje_error = 100 * obras_con_errores / total_obras
        print(f"\n{porcentaje_error:.2f}% de obras con errores ({obras_con_errores}/{total_obras}).")
    else:
        print("No se encontraron obras.")

def reconstruir_y_comprobar_beats2(sequences_path, vocab_path, subdivision=1.0):
    """
    Reconstruye las obras mostrando solo los tokens BAR, BEAT y DUR.
    Cada compás se muestra como BAR_X, donde X es el número de compás.
    Llama a check_beats_piece() y además detecta si algún BAR no va seguido de BEAT_1.
    Muestra porcentajes globales de obras con errores de tokenización y de estructura.
    """
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}

    total_obras = len(sequences)
    obras_con_errores = 0
    obras_con_bar_mal = 0

    for i, seq in enumerate(sequences, start=1):
        filtered_tokens = []
        compas = 1

        for idx in seq:
            token = inv_vocab[idx]
            if "BAR" in token:
                filtered_tokens.append(f"BAR_{compas}")
                compas += 1
            elif "BEAT" in token or "DUR" in token:
                filtered_tokens.append(token)

        print(f"\n=== Obra {i} ===")
        print("Reconstrucción:")
        print(" ".join(filtered_tokens))

        # 1. Comprobación de beats
        print("\nComprobación de beats:")
        resultado = check_beats_piece(filtered_tokens, subdivision=subdivision)
        if not resultado:
            obras_con_errores += 1

        # 2. Comprobación de que cada BAR vaya seguido de BEAT_1
        print("\nComprobación de estructura BAR–BEAT_1:")
        bar_error = False
        for j, token in enumerate(filtered_tokens[:-1]):
            if token.startswith("BAR"):
                siguiente = filtered_tokens[j + 1]
                if not siguiente.startswith("BEAT_1"):
                    print(f"  Error: {token} no va seguido de BEAT_1 (siguiente = {siguiente})")
                    bar_error = True
        if bar_error:
            obras_con_bar_mal += 1
        else:
            print("  Todos los compases comienzan correctamente con BEAT_1.")

    # --- Resumen global ---
    if total_obras > 0:
        porcentaje_error = 100 * obras_con_errores / total_obras
        porcentaje_bar_error = 100 * obras_con_bar_mal / total_obras
        print(f"\nResumen general:")
        print(f"  {porcentaje_error:.2f}% de obras con errores de beats ({obras_con_errores}/{total_obras})")
        print(f"  {porcentaje_bar_error:.2f}% de obras con errores de estructura BAR–BEAT_1 ({obras_con_bar_mal}/{total_obras})")
    else:
        print("No se encontraron obras.")

import json
import math

def reconstruir_y_comprobar_beats3(sequences_path, vocab_path, subdivision=1.0):
    """
    Reconstruye las obras mostrando solo los tokens BAR, BEAT y DUR.
    Cada compás se muestra como BAR_X, donde X es el número de compás.
    Llama a check_beats_piece() y calcula:
      - porcentaje global de obras con errores
      - número medio de compases erróneos por obra con su desviación estándar
    """
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}

    total_obras = len(sequences)
    obras_con_errores = 0
    errores_por_obra = []

    for i, seq in enumerate(sequences, start=1):
        filtered_tokens = []
        compas = 1

        for idx in seq:
            token = inv_vocab[idx]
            if "BAR" in token:
                filtered_tokens.append(f"BAR_{compas}")
                compas += 1
            elif "BEAT" in token or "DUR" in token:
                filtered_tokens.append(token)

        print(f"\n=== Obra {i} ===")
        print("Reconstrucción:")
        print(" ".join(filtered_tokens))

        print("\nComprobación de beats:")
        errores = check_beats_piece(filtered_tokens, subdivision=subdivision)
        errores_por_obra.append(errores)
        if errores > 0:
            obras_con_errores += 1

    if total_obras == 0:
        print("No se encontraron obras.")
        return

    # Estadísticas globales
    porcentaje_error = 100 * obras_con_errores / total_obras
    media_errores = sum(errores_por_obra) / total_obras
    # desviación estándar poblacional
    desviacion = math.sqrt(sum((x - media_errores) ** 2 for x in errores_por_obra) / total_obras)

    print("\nResumen global:")
    print(f"  {porcentaje_error:.2f}% de obras con errores ({obras_con_errores}/{total_obras})")
    print(f"  Número medio de compases erróneos por obra: {media_errores:.2f} ± {desviacion:.2f}")


def check_beats_piece(tokens, subdivision=1.0):
    dur = 0.0
    measure_number = 0
    beat_number = 0
    errores = 0
    totales = 0

    for token in tokens:
        if "BAR" in token:
            if dur > 0 and abs(dur - subdivision*2) > 0.01:
                print(f"Error en compás {measure_number} (dur acumulada = {dur})")
                errores += 1
            totales += 1
            measure_number += 1
            beat_number = 0
            dur = 0

        elif "DUR" in token:
            dur += float(token.replace("DUR_", ""))

        elif "BEAT" in token:
            beat_number += 1

    if totales == 0:
        print("Obra sin beats detectados.")
        return 0

    if errores > 0:
        print(f"Obra con {100 * errores / totales:.2f}% de beats con error ({errores}/{totales}).")
        return errores
    else:
        print("Obra correctamente tokenizada.")
        return 0



# ==============================================
# PROGRAMA PRINCIPAL
# ==============================================
if __name__ == "__main__":
    # partitura = reconstruir_partitura(secuencia, vocabulario)
    # partitura.write("musicxml", fp="test.xml")
    reconstruir_y_comprobar_beats3(rf"{ROOT_DIR}\models\todos\standard\sequences_major_2_4.json",
                         rf"{ROOT_DIR}\models\todos\standard\vocab_major_2_4.json",
                         1.0)


