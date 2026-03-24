from music21 import note, chord, stream, meter


def is_grace(el, epsilon=1e-5):
    """
    Detecta si un elemento musical (Note o Chord) es grace note.

    Parámetros:
        el: objeto music21 (Note o Chord)
        epsilon: tolerancia para detectar duraciones casi cero

    Devuelve:
        True si se considera grace note, False en caso contrario
    """

    # Solo tiene sentido para notas o acordes
    if not isinstance(el, (note.Note, chord.Chord, note.Unpitched)):
        return False

    # -------------------------------------------------
    # 1. Método oficial de music21
    # -------------------------------------------------
    try:
        if el.duration.isGrace:
            return True
    except Exception:
        pass

    # -------------------------------------------------
    # 2. Duración exactamente cero
    # -------------------------------------------------
    try:
        if el.quarterLength == 0:
            return True
    except Exception:
        pass

    # -------------------------------------------------
    # 3. Duración casi cero (artefacto OMR)
    # -------------------------------------------------
    try:
        if abs(float(el.quarterLength)) < epsilon:
            return True
    except Exception:
        pass

    # -------------------------------------------------
    # 4. Tipo de duración extraño
    # -------------------------------------------------
    try:
        if el.duration.type == 'zero':
            return True
    except Exception:
        pass

    # -------------------------------------------------
    # 5. Duración no ligada al flujo métrico
    # -------------------------------------------------
    try:
        if hasattr(el.duration, "linked") and el.duration.linked is False:
            if el.quarterLength == 0:
                return True
    except Exception:
        pass

    # -------------------------------------------------
    # 6. Grace implícita detectada en XML
    # (music21 guarda flag en duration._components)
    # -------------------------------------------------
    try:
        if hasattr(el.duration, "_components"):
            for comp in el.duration._components:
                if hasattr(comp, "isGrace") and comp.isGrace:
                    return True
    except Exception:
        pass

    return False


def check_anacrusis(score, tol=1e-6):
    """
    Comprueba si el score tiene anacrusa y si está bien formada.

    Returns:
        True  -> hay anacrusa y está bien formada
        False -> hay anacrusa pero está mal formada
        None  -> no hay anacrusa (primer compás completo)
    """

    # --------------------------------------------------
    # 1) Ausencia o mala detección de time signature
    # --------------------------------------------------
    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    if not ts:
        return False

    ts = ts[0]
    try:
        bar_duration = float(ts.barDuration.quarterLength)
    except Exception:
        return False

    # --------------------------------------------------
    # Obtener primer compás real
    # --------------------------------------------------
    measures = list(score.parts[0].getElementsByClass(stream.Measure))
    if not measures:
        return False

    m1 = measures[0]

    # --------------------------------------------------
    # 6) Numeración incoherente
    # (si existe measure 0 pero no es incompleto, o si m1.number
    #  no es 0 o 1, lo consideramos inconsistente)
    # --------------------------------------------------
    if m1.number not in (0, 1):
        return False

    # --------------------------------------------------
    # Calcular duración REAL del primer compás
    # (solo notas y silencios)
    # --------------------------------------------------
    notes_and_rests = list(m1.notesAndRests)
    if not notes_and_rests:
        return False

    total_duration = sum(float(el.quarterLength) for el in notes_and_rests)

    # --------------------------------------------------
    # 4) Suma igual al compás completo → no hay anacrusa
    # --------------------------------------------------
    if abs(total_duration - bar_duration) < tol:
        return None

    # --------------------------------------------------
    # 5) Suma mayor que compás → mal formada
    # (tolerancia para evitar errores con tresillos)
    # --------------------------------------------------
    if total_duration - bar_duration > tol:
        return False

    # Si llegamos aquí, duración < compás → posible anacrusa

    # --------------------------------------------------
    # 1) Silencios a la derecha de las notas
    # --------------------------------------------------
    # Detectamos si después de la última nota aparece un silencio
    last_note_index = None
    for i, el in enumerate(notes_and_rests):
        if not el.isRest:
            last_note_index = i

    if last_note_index is None:
        # solo silencios → no puede ser anacrusa válida
        return False

    for el in notes_and_rests[last_note_index + 1:]:
        if el.isRest:
            return False

    # --------------------------------------------------
    # 2) Si hay silencios, la suma debe coincidir exactamente
    #     con la suma real ya calculada (ya controlado arriba),
    #     pero además no debe rellenar hasta completar compás
    # --------------------------------------------------
    # Si existen silencios y total_duration + resto == compás,
    # ya habría saltado como caso 4.
    # Aquí solo verificamos coherencia interna.
    silence_sum = sum(float(el.quarterLength) for el in notes_and_rests if el.isRest)
    note_sum = sum(float(el.quarterLength) for el in notes_and_rests if not el.isRest)

    if abs((silence_sum + note_sum) - total_duration) > tol:
        return False

    # --------------------------------------------------
    # Si pasa todos los checks → anacrusa válida
    # --------------------------------------------------
    return True
