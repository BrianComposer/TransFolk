# from pathlib import Path
# from statistics import mean
# from music21 import converter, meter, note, stream
#
# def normalize_musicxml_corpus(data_dir, output_dir, allowed_durations, overwrite=False):
#     """
#     Limpieza y normalización rítmica completa de un corpus MusicXML.
#
#     Procesa todos los archivos del directorio `data_dir` y:
#       1. Detecta o corrige el compás (TimeSignature) según la duración media de los compases.
#       2. Corrige compases incompletos, sobrepasados o con anacrusa.
#       3. Ajusta todas las duraciones de notas/silencios a las permitidas en `allowed_durations`.
#       4. Guarda la versión limpia en `output_dir`.
#
#     Parámetros:
#         data_dir : str o Path
#             Carpeta con archivos MusicXML (.xml, .mxl, .musicxml)
#         output_dir : str o Path
#             Carpeta donde se guardarán los archivos corregidos
#         allowed_durations : list[float]
#             Lista de duraciones válidas (en unidades de negra, ej. [0.25, 0.5, 1.0, 2.0])
#         overwrite : bool
#             Si True, sobrescribe archivos existentes
#     """
#
#     def closest_duration(d, allowed):
#         """Devuelve la duración permitida más cercana a 'd'."""
#         return min(allowed, key=lambda x: abs(x - d))
#
#     data_dir = Path(data_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     files = list(data_dir.rglob("*.xml")) + list(data_dir.rglob("*.mxl")) + list(data_dir.rglob("*.musicxml"))
#     print(f"Encontrados {len(files)} archivos MusicXML.\n")
#
#     total_found = len(files)
#     total_processed = 0
#     total_modified = 0
#     total_skipped = 0
#     total_errors = 0
#
#     for xml_path in files:
#         try:
#             score = converter.parse(xml_path)
#         except Exception as e:
#             print(f"⛔ Error leyendo {xml_path.name}: {e}")
#             total_errors += 1
#             continue
#
#         try:
#             first_part = score.parts[0]
#             measures_fp = first_part.getElementsByClass(stream.Measure)
#         except Exception as e:
#             print(f"⚠️ {xml_path.name}: sin partes/compases ({e}). Omitido.")
#             total_skipped += 1
#             continue
#
#         if len(measures_fp) == 0:
#             print(f"⚠️ {xml_path.name}: sin compases. Omitido.")
#             total_skipped += 1
#             continue
#
#         measure_lengths = [sum(el.quarterLength for el in m.notesAndRests)
#                            for m in measures_fp if len(m.notesAndRests) > 0]
#         if not measure_lengths:
#             print(f"⚠️ {xml_path.name}: sin contenido rítmico. Omitido.")
#             total_skipped += 1
#             continue
#
#         avg_len = mean(measure_lengths)
#         total_processed += 1
#
#         # === 1. Leer o corregir el compás ===
#         try:
#             ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
#             current_len = ts.barDuration.quarterLength
#         except IndexError:
#             ts = None
#             current_len = 0.0
#
#         if ts is None or abs(avg_len - current_len) > 0.25:
#             candidates = [(2,4), (3,4), (4,4), (6,8)]
#             num, den = min(candidates, key=lambda s: abs(avg_len - (s[0]*(4/s[1]))))
#             new_ts = meter.TimeSignature(f"{num}/{den}")
#             for p in score.parts:
#                 for old in list(p.recurse().getElementsByClass(meter.TimeSignature)):
#                     old.activeSite.remove(old)
#                 m1 = p.getElementsByClass(stream.Measure)[0]
#                 m1.insert(0.0, meter.TimeSignature(f"{num}/{den}"))
#             ts = new_ts
#             print(f"🧭 {xml_path.name}: compás fijado a {ts.ratioString} (media {avg_len:.2f})")
#             modified_time_sig = True
#         else:
#             modified_time_sig = False
#
#         numerator = ts.numerator
#         denominator = ts.denominator
#         expected_len = numerator * (4 / denominator)
#
#         modified = modified_time_sig
#
#         # === 2. Corrección compás a compás ===
#         for p in score.parts:
#             measures = p.getElementsByClass(stream.Measure)
#             for idx, m in enumerate(measures):
#                 beat = 0.0
#                 last_el = None
#
#                 for el in m.notesAndRests:
#                     # --- 3. Ajuste de duraciones no permitidas ---
#                     old_dur = el.quarterLength
#                     new_dur = closest_duration(old_dur, allowed_durations)
#                     if abs(new_dur - old_dur) > 1e-6:
#                         el.quarterLength = new_dur
#                         modified = True
#                         print(f"🔧 {xml_path.name} c.{m.number}: duración ajustada {old_dur} → {new_dur}")
#
#                     beat += el.quarterLength
#                     last_el = el
#
#                 diff = round(expected_len - beat, 6)
#
#                 # Primer compás anacrúsico
#                 if idx == 0 and diff > 1e-6:
#                     r = note.Rest(quarterLength=diff)
#                     try:
#                         m.insertAndShift(0.0, r)
#                     except Exception:
#                         for el in list(m.notesAndRests):
#                             el.offset += diff
#                         m.insert(0.0, r)
#                     modified = True
#                     print(f"🟢 {xml_path.name}: anacrusa corregida (+{diff:.3f})")
#                     continue
#
#                 # Compás incompleto
#                 if diff > 1e-6 and last_el is not None:
#                     last_el.quarterLength += diff
#                     modified = True
#                     print(f"🟡 {xml_path.name} c.{m.number}: +{diff:.3f}")
#
#                 # Compás sobrepasado
#                 elif diff < -1e-6 and last_el is not None:
#                     exceso = abs(diff)
#                     if last_el.quarterLength - exceso > 1e-3:
#                         last_el.quarterLength -= exceso
#                         modified = True
#                         print(f"🟠 {xml_path.name} c.{m.number}: -{exceso:.3f}")
#                     else:
#                         m.remove(last_el)
#                         modified = True
#                         print(f"🔴 {xml_path.name} c.{m.number}: última figura eliminada")
#
#         # === 4. Guardar resultado ===
#         out_path = output_dir / xml_path.name
#         try:
#             if modified or overwrite:
#                 score.write('musicxml', fp=str(out_path))
#                 total_modified += 1
#             else:
#                 # No modificado, pero no se sobrescribe
#                 pass
#         except Exception as e:
#             print(f"❌ Error guardando {out_path.name}: {e}")
#             total_errors += 1
#
#     print(f"\n✅ Normalización completada.")
#     print(f"  Archivos encontrados: {total_found}")
#     print(f"  Procesados correctamente: {total_processed}")
#     print(f"  Archivos modificados: {total_modified}")
#     print(f"  Archivos omitidos: {total_skipped}")
#     print(f"  Errores: {total_errors}")

from pathlib import Path
from statistics import mean
from fractions import Fraction
from music21 import converter, meter, note, chord, key, stream, harmony, metadata, duration



# def normalize_musicxml_corpus(data_dir, output_dir, allowed_durations, overwrite=False, delete_grace_notes=True, midi_min=57, midi_max=83, create_title=False):
#     """
#     Limpieza y normalización métrica y rítmica avanzada de corpus MusicXML.
#
#     Funcionalidades principales:
#     - Elimina compases iniciales y finales vacíos o formados solo por silencios.
#     - Copia la primera armadura de clave (key signature).
#     - Corrige compases, duraciones, anacrusas y sobrepasos.
#     - Cuantiza duraciones a ALLOWED_DURATIONS.
#     - Normaliza el registro de notas y acordes al rango [57–83] (A3–B5).
#     - Informa detalladamente del proceso por consola.
#     """
#     # Esta funciona OK pero peta con las grace notes semicorcheas
#
#     def list_musicxml_files(root: Path):
#         return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".xml", ".mxl", ".musicxml")]
#
#     def ql_to_frac(x: float) -> Fraction:
#         return Fraction(x).limit_denominator(96)
#
#     allowed_fracs = [ql_to_frac(d) for d in allowed_durations]
#
#     def closest_allowed(frac_dur: Fraction) -> Fraction:
#         return min(allowed_fracs, key=lambda a: abs(a - frac_dur))
#
#     def expected_len_from_ts(ts: meter.TimeSignature) -> Fraction:
#         return ql_to_frac(ts.numerator * (4 / ts.denominator))
#
#
#     def estimate_time_signature(measure_lengths_frac):
#         candidates = [(2, 4), (3, 4), (4, 4), (6, 4), (6, 8), (3, 8)]
#         avg_len = mean([float(x) for x in measure_lengths_frac])
#         def bar_len(n, d): return n * (4 / d)
#         num, den = min(candidates, key=lambda s: abs(avg_len - bar_len(*s)))
#         return meter.TimeSignature(f"{num}/{den}")
#
#     data_dir = Path(data_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     files = list_musicxml_files(data_dir)
#     print(f"\n🎼 Archivos detectados: {len(files)}\n")
#
#     total_read = 0
#     total_processed = 0
#     total_ignored = 0
#     total_errors = 0
#     total_saved = 0
#
#     for src in files:
#         print(f"──────────────────────────────────────────────")
#         print(f"Procesando: {src.name}")
#
#         try:
#             score = converter.parse(src)
#             total_read += 1
#         except Exception as e:
#             print(f"⛔ Error al leer {src.name}: {e}")
#             total_errors += 1
#             continue
#
#         new_score = stream.Score()
#         new_part = stream.Part()
#         new_score.insert(0, new_part)
#
#         # --- Título = nombre del archivo  ---
#         if create_title:
#             if new_score.metadata is None:
#                 new_score.insert(0, metadata.Metadata())
#             # nombre del archivo sin extensión (más bonito)
#             new_score.metadata.title = src.stem
#
#         # --- Copiar armadura ---
#         try:
#             orig_key = score.recurse().getElementsByClass(key.KeySignature)[0]
#             new_part.insert(0.0, key.KeySignature(orig_key.sharps))
#             print(f"🎯 Armadura copiada: {orig_key.sharps} alteraciones.")
#         except IndexError:
#             print("⚠️ Sin armadura detectada (se omitirá).")
#
#         # --- Obtener primera parte ---
#         try:
#             orig_part = score.parts[0]
#         except Exception as e:
#             print(f"⚠️ {src.name}: sin parts ({e}). Omitido.")
#             total_ignored += 1
#             continue
#
#         orig_measures = list(orig_part.getElementsByClass(stream.Measure))
#         if len(orig_measures) == 0:
#             print(f"⚠️ {src.name}: sin compases detectados. Omitido.")
#             total_ignored += 1
#             continue
#
#         # --- Eliminar compases iniciales vacíos o de solo silencios ---
#         while orig_measures:
#             first_measure = orig_measures[0]
#             has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in first_measure.notesAndRests)
#             if not has_note:
#                 # print(f"🗑️ Eliminado compás inicial vacío o solo con silencios.")
#                 orig_measures.pop(0)
#             else:
#                 break
#
#         # --- Eliminar compases finales vacíos o de solo silencios ---
#         while orig_measures:
#             last_measure = orig_measures[-1]
#             has_note = any(isinstance(el, (note.Note, chord.Chord)) for el in last_measure.notesAndRests)
#             if not has_note:
#                 # print(f"🗑️ Eliminado compás final vacío o solo con silencios.")
#                 orig_measures.pop(-1)
#             else:
#                 break
#
#         if not orig_measures:
#             print(f"⚠️ {src.name}: solo compases vacíos, omitido.")
#             total_ignored += 1
#             continue
#
#         # --- Estimar compás ---
#         measure_lengths = []
#         for m in orig_measures:
#             total = sum(el.quarterLength for el in m.notesAndRests)
#             if total > 0:
#                 measure_lengths.append(ql_to_frac(total))
#
#         if not measure_lengths:
#             print(f"⚠️ {src.name}: sin contenido rítmico. Omitido.")
#             total_ignored += 1
#             continue
#
#         estimated_ts = estimate_time_signature(measure_lengths)
#         print(f"⏱️ Compás estimado: {estimated_ts.ratioString}")
#
#         try:
#             read_ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
#             ts_to_use = estimated_ts if abs(float(expected_len_from_ts(read_ts)) -
#                                             float(mean([float(x) for x in measure_lengths]))) > 0.25 else read_ts
#         except IndexError:
#             ts_to_use = estimated_ts
#
#         new_part.insert(0.0, meter.TimeSignature(ts_to_use.ratioString))
#         exp_len = expected_len_from_ts(ts_to_use)
#         print(f"✅ Compás asignado: {ts_to_use.ratioString} (duración esperada {float(exp_len):.2f})")
#
#         modified = False
#
#         # --- Procesar compases ---
#         for mi, m in enumerate(orig_measures, start=1):
#             new_measure = stream.Measure(number=m.number)
#             elems = []
#             grace_idx = 0
#             for el in m.notesAndRests:
#                 isGrace = False
#                 if isinstance(el, harmony.ChordSymbol):
#                     continue
#                 try:
#                     if looks_like_grace(el):
#                         if delete_grace_notes:
#                             isGrace=False
#                             continue
#                         else:
#                             isGrace=True
#                 except Exception:
#                     pass
#
#                 dur_frac = ql_to_frac(float(el.quarterLength))
#                 dur_q = closest_allowed(dur_frac)
#
#                 if abs(dur_q - dur_frac) > Fraction(1, 192):
#                     print(f"🔧 c.{m.number}: dur {float(dur_frac):.3f} ajustada a {float(dur_q):.3f}")
#                     modified = True
#
#                 # --- Rest ---
#                 if isinstance(el, note.Rest):
#                     elems.append(note.Rest(quarterLength=float(dur_q)))
#
#                 # --- Note ---
#                 elif isinstance(el, note.Note):
#                     midi_val = el.pitch.midi
#                     while midi_val < midi_min:
#                         midi_val += 12
#                     while midi_val > midi_max:
#                         midi_val -= 12
#                     n = note.Note(midi_val, quarterLength=float(dur_q))
#                     if isGrace:
#                         n = note.Note(midi_val)
#                         n = n.getGrace()
#                         n.duration.type = 'eighth'
#                     else:
#                         n = note.Note(midi_val, quarterLength=float(dur_q))
#                     # if n.pitch.accidental is not None:
#                     #     n.pitch.accidental.displayStatus = False  # oculta el becuadro innecesario
#                     elems.append(n)
#
#                 # --- Chord ---
#                 elif isinstance(el, chord.Chord):
#                     top_pitch = max(el.pitches, key=lambda p: p.midi)
#                     midi_val = top_pitch.midi
#                     while midi_val < midi_min:
#                         midi_val += 12
#                     while midi_val > midi_max:
#                         midi_val -= 12
#                     elems.append(note.Note(midi_val, quarterLength=float(dur_q)))
#
#             # =====================================================
#             # PRIMER COMPÁS: mover TODOS los silencios al principio
#             # =====================================================
#             if mi == 1 and len(elems) > 1:
#                 rests = [e for e in elems if isinstance(e, note.Rest)]
#                 notes = [e for e in elems if isinstance(e, note.Note)]
#                 if rests and notes:
#                     elems = rests + notes
#                     modified = True
#                     print(f"🟢 c.{m.number}: silencios movidos al inicio del compás.")
#
#             # --- Longitud total ---
#             total_len = Fraction(0, 1)
#             for e in elems:
#                 total_len += ql_to_frac(e.quarterLength)
#
#             # --- Anacrusa ---
#             if mi == 1 and total_len < exp_len:
#                 missing = exp_len - total_len
#                 print(f"🟢 Anacrusa detectada: rellenando +{float(missing):.3f}")
#                 r = note.Rest(quarterLength=float(missing))
#                 new_measure.insert(0.0, r)
#                 offset = float(missing)
#                 for e in elems:
#                     new_measure.insert(offset, e)
#                     offset += float(e.quarterLength)
#                 elems = None
#                 modified = True
#             else:
#                 curr = 0.0
#                 for e in elems:
#                     new_measure.insert(curr, e)
#                     curr += float(e.quarterLength)
#
#             total_len = Fraction(0, 1)
#             for e in new_measure.notesAndRests:
#                 total_len += ql_to_frac(e.quarterLength)
#
#             # --- Compás corto ---
#             if total_len < exp_len and len(new_measure.notesAndRests) > 0:
#                 diff = exp_len - total_len
#                 items = list(new_measure.notesAndRests)
#                 last = items[-1]
#
#                 # crear nueva nota o silencio
#                 if isinstance(last, note.Note):
#                     pitch = last.pitch.midi
#                     new_el = note.Note(pitch, quarterLength=float(diff))
#                 else:
#                     new_el = note.Rest(quarterLength=float(diff))
#
#                 # insertar al final del compás
#                 new_measure.insert(float(total_len), new_el)
#
#                 total_len += diff
#                 modified = True
#                 print(f"🟡 c.{m.number}: añadido elemento para completar {float(diff):.3f}")
#
#
#             # --- Compás largo ---
#             while total_len > exp_len and len(new_measure.notesAndRests) > 0:
#                 exceso = total_len - exp_len
#                 items = list(new_measure.notesAndRests)
#                 last = items[-1]
#                 last_len = ql_to_frac(last.quarterLength)
#
#                 if last_len > exceso + Fraction(1, 192):
#                     last.quarterLength = float(last_len - exceso)
#                     total_len = exp_len
#                     print(f"🟠 c.{m.number}: recortado -{float(exceso):.3f}")
#                     modified = True
#                 else:
#                     new_measure.remove(last)
#                     total_len -= last_len
#                     print(f"🔴 c.{m.number}: eliminada figura excedente {float(last_len):.3f}")
#                     modified = True
#
#             new_part.append(new_measure)
#
#         # --- Guardar ---
#         out_path = output_dir / src.name
#         try:
#             new_score.write('musicxml', fp=str(out_path))
#             print(f"💾 Guardado → {out_path.name}")
#             total_saved += 1
#             total_processed += 1
#         except Exception as e:
#             print(f"❌ Error guardando {out_path.name}: {e}")
#             total_errors += 1
#
#     # --- Informe final ---
#     print("\n═══════════════════════════════════════════════")
#     print("📊 INFORME FINAL")
#     print(f" Archivos encontrados:   {len(files)}")
#     print(f" Leídos correctamente:   {total_read}")
#     print(f" Procesados y guardados: {total_processed}")
#     print(f" Ignorados:              {total_ignored}")
#     print(f" Errores:                {total_errors}")
#     print(f" Archivos escritos:      {total_saved}")
#     print("═══════════════════════════════════════════════")

from pathlib import Path
from fractions import Fraction
from statistics import mean
from music21 import converter, stream, note, chord, meter, key, metadata, harmony
from music21 import pitch as m21pitch
from music21.duration import quarterConversion
from .common import is_grace



#ESTA ES LA ULTIMA QUE FUNCIONA OK
def normalize_musicxml_corpus(
    data_dir,
    output_dir,
    allowed_durations,
    overwrite=False,
    delete_grace_notes=True,
    midi_min=57,
    midi_max=83,
    create_title=False
):
    """
    Limpieza y normalización métrica y rítmica avanzada de corpus MusicXML.

    Fix importante:
    - Las grace notes NO se cuantizan, NO cuentan para el tiempo del compás y NO deben avanzar offsets.
      Si se preservan (delete_grace_notes=False), se insertan como objetos grace puros (duración métrica 0).
    """

    from pathlib import Path
    from fractions import Fraction
    from statistics import mean
    from music21 import converter, stream, note, chord, meter, key, metadata, harmony
    from music21 import pitch as m21pitch

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def list_musicxml_files(root: Path):
        return [p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in (".xml", ".mxl", ".musicxml")]

    def ql_to_frac(x: float) -> Fraction:
        return Fraction(x).limit_denominator(96)

    allowed_fracs = [ql_to_frac(d) for d in allowed_durations]

    def closest_allowed(frac_dur: Fraction) -> Fraction:
        return min(allowed_fracs, key=lambda a: abs(a - frac_dur))

    def expected_len_from_ts(ts: meter.TimeSignature) -> Fraction:
        return ql_to_frac(ts.numerator * (4 / ts.denominator))

    def estimate_time_signature(measure_lengths_frac):
        candidates = [(2, 4), (3, 4), (4, 4), (6, 4), (6, 8), (3, 8)]
        avg_len = mean([float(x) for x in measure_lengths_frac])

        def bar_len(n, d):
            return n * (4 / d)

        num, den = min(candidates, key=lambda s: abs(avg_len - bar_len(*s)))
        return meter.TimeSignature(f"{num}/{den}")

    # Si ya la tienes definida fuera, no pasa nada: esta fallback te salva.
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

    def clamp_midi(midi_val: int) -> int:
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

    # -------------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------------
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        new_score = stream.Score()
        new_part = stream.Part()
        new_score.insert(0, new_part)

        # --- Título = nombre del archivo  ---
        if create_title:
            if new_score.metadata is None:
                new_score.insert(0, metadata.Metadata())
            new_score.metadata.title = src.stem

        # --- Copiar armadura ---
        try:
            orig_key = score.recurse().getElementsByClass(key.KeySignature)[0]
            new_part.insert(0.0, key.KeySignature(orig_key.sharps))
            print(f"🎯 Armadura copiada: {orig_key.sharps} alteraciones.")
        except IndexError:
            print("⚠️ Sin armadura detectada (se omitirá).")
        except Exception as e:
            print(f"⚠️ Error leyendo armadura (se omitirá): {e}")

        # --- Obtener primera parte ---
        try:
            orig_part = score.parts[0]
        except Exception as e:
            print(f"⚠️ {src.name}: sin parts ({e}). Omitido.")
            total_ignored += 1
            continue

        orig_measures = list(orig_part.getElementsByClass(stream.Measure))
        if len(orig_measures) == 0:
            print(f"⚠️ {src.name}: sin compases detectados. Omitido.")
            total_ignored += 1
            continue

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

        if not orig_measures:
            print(f"⚠️ {src.name}: solo compases vacíos, omitido.")
            total_ignored += 1
            continue

        # --- Estimar compás (ignorando grace notes para el cálculo de longitudes) ---
        measure_lengths = []
        for m in orig_measures:
            total = 0.0
            for el in m.notesAndRests:
                if isinstance(el, harmony.ChordSymbol):
                    continue
                if looks_like_grace(el):
                    continue
                try:
                    total += float(el.quarterLength)
                except Exception:
                    pass
            if total > 0:
                measure_lengths.append(ql_to_frac(total))

        if not measure_lengths:
            print(f"⚠️ {src.name}: sin contenido rítmico (tras ignorar graces). Omitido.")
            total_ignored += 1
            continue

        estimated_ts = estimate_time_signature(measure_lengths)
        print(f"⏱️ Compás estimado: {estimated_ts.ratioString}")

        try:
            read_ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
            ts_to_use = estimated_ts if abs(
                float(expected_len_from_ts(read_ts)) - float(mean([float(x) for x in measure_lengths]))
            ) > 0.25 else read_ts
        except IndexError:
            ts_to_use = estimated_ts
        except Exception:
            ts_to_use = estimated_ts

        new_part.insert(0.0, meter.TimeSignature(ts_to_use.ratioString))
        exp_len = expected_len_from_ts(ts_to_use)
        print(f"✅ Compás asignado: {ts_to_use.ratioString} (duración esperada {float(exp_len):.2f})")

        modified = False

        # ---------------------------------------------------------------------
        # Procesar compases
        # ---------------------------------------------------------------------
        for mi, m in enumerate(orig_measures, start=1):
            new_measure = stream.Measure(number=m.number)
            elems = []

            for el in m.notesAndRests:
                if isinstance(el, harmony.ChordSymbol):
                    continue

                # -------------------------
                # GRACE PATH (NO cuantizar)
                # -------------------------
                try:
                    is_grace = looks_like_grace(el)
                except Exception:
                    is_grace = False

                if is_grace:
                    if delete_grace_notes:
                        modified = True
                        continue

                    # Preservar grace: NO asignar duraciones métricas,
                    # NO tocar allowed_durations,
                    # NO forzar duration.type.
                    if isinstance(el, note.Note):
                        midi_val = clamp_midi(int(el.pitch.midi))
                        p = clamp_pitch_octave(el.pitch, midi_min, midi_max)
                        tipo = el.duration.type
                        if tipo =="zero":
                            tipo = '16th'
                        gn = note.Note(p, type=tipo).getGrace()
                        gn.duration.slash = True  # barra diagonal
                        elems.append(gn)
                    elif isinstance(el, chord.Chord):
                        top_pitch = max(el.pitches, key=lambda p: p.midi)
                        midi_val = clamp_midi(int(top_pitch.midi))
                        gn = note.Note(midi_val).getGrace()
                        elems.append(gn)
                    # rests grace raros → ignorar
                    continue

                # -------------------------
                # NORMAL PATH (cuantizar)
                # -------------------------
                dur_frac = ql_to_frac(float(el.quarterLength))
                dur_q = closest_allowed(dur_frac)

                if abs(dur_q - dur_frac) > Fraction(1, 192):
                    print(f"🔧 c.{m.number}: dur {float(dur_frac):.3f} ajustada a {float(dur_q):.3f}")
                    modified = True

                # Rest
                if isinstance(el, note.Rest):
                    elems.append(note.Rest(quarterLength=float(dur_q)))

                # Note
                elif isinstance(el, note.Note):
                    # midi_val = clamp_midi(int(el.pitch.midi))
                    p = clamp_pitch_octave(el.pitch, midi_min, midi_max)
                    elems.append(note.Note(p, quarterLength=float(dur_q)))

                # Chord -> top pitch only
                elif isinstance(el, chord.Chord):
                    top_pitch = max(el.pitches, key=lambda p: p.midi)
                    midi_val = clamp_midi(int(top_pitch.midi))
                    elems.append(note.Note(midi_val, quarterLength=float(dur_q)))

            # =====================================================
            # PRIMER COMPÁS: mover TODOS los silencios al principio
            # (sin tocar graces)
            # =====================================================
            if mi == 1 and len(elems) > 1:
                rests = [e for e in elems if isinstance(e, note.Rest)]
                notes = [e for e in elems if isinstance(e, note.Note)]
                if rests and notes:
                    elems = rests + notes
                    modified = True
                    print(f"🟢 c.{m.number}: silencios movidos al inicio del compás.")

            # --- Longitud total (sin contar graces) ---
            total_len = Fraction(0, 1)
            for e in elems:
                try:
                    if getattr(e.duration, "isGrace", False):
                        continue
                except Exception:
                    pass
                try:
                    total_len += ql_to_frac(e.quarterLength)
                except Exception:
                    pass

            # --- Anacrusa (solo con duración real) ---
            if mi == 1 and total_len < exp_len:
                missing = exp_len - total_len
                print(f"🟢 Anacrusa detectada: rellenando +{float(missing):.3f}")
                r = note.Rest(quarterLength=float(missing))
                new_measure.insert(0.0, r)

                offset = float(missing)
                for e in elems:
                    new_measure.insert(offset, e)
                    # grace no avanza offset
                    try:
                        if getattr(e.duration, "isGrace", False):
                            continue
                    except Exception:
                        pass
                    offset += float(e.quarterLength)

                modified = True
            else:
                curr = 0.0
                for e in elems:
                    new_measure.insert(curr, e)
                    # grace no avanza el tiempo
                    try:
                        if getattr(e.duration, "isGrace", False):
                            continue
                    except Exception:
                        pass
                    curr += float(e.quarterLength)

            # --- Recalcular longitud total del compás (sin graces) ---
            total_len = Fraction(0, 1)
            for e in new_measure.notesAndRests:
                try:
                    if getattr(e.duration, "isGrace", False):
                        continue
                except Exception:
                    pass

                try:
                    total_len += ql_to_frac(e.quarterLength)
                except Exception:
                    pass




            # --- Compás corto: completar al final (solo si hay algo real) ---
            if total_len < exp_len and len([x for x in new_measure.notesAndRests
                                           if not getattr(x.duration, "isGrace", False)]) > 0:
                diff = exp_len - total_len

                # buscar el último elemento NO-grace
                items = list(new_measure.notesAndRests)
                last_real = None
                for it in reversed(items):
                    if not getattr(it.duration, "isGrace", False):
                        last_real = it
                        break

                if last_real is not None:
                    if isinstance(last_real, note.Note):
                        pitch = last_real.pitch.midi
                        new_el = note.Note(pitch, quarterLength=float(diff))
                    else:
                        new_el = note.Rest(quarterLength=float(diff))

                    new_measure.insert(float(total_len), new_el)
                    total_len += diff
                    modified = True
                    print(f"🟡 c.{m.number}: añadido elemento para completar {float(diff):.3f}")

            # --- Compás largo: recortar/eliminar elementos al final (solo reales) ---
            while total_len > exp_len:
                exceso = total_len - exp_len

                # último elemento NO-grace
                items = list(new_measure.notesAndRests)
                last_real = None
                for it in reversed(items):
                    if not getattr(it.duration, "isGrace", False):
                        last_real = it
                        break

                if last_real is None:
                    break

                last_len = ql_to_frac(last_real.quarterLength)

                if last_len > exceso + Fraction(1, 192):
                    last_real.quarterLength = float(last_len - exceso)
                    total_len = exp_len
                    print(f"🟠 c.{m.number}: recortado -{float(exceso):.3f}")
                    modified = True
                else:
                    new_measure.remove(last_real)
                    total_len -= last_len
                    print(f"🔴 c.{m.number}: eliminada figura excedente {float(last_len):.3f}")
                    modified = True

            new_part.append(new_measure)

        # --- Guardar ---
        try:
            new_score.write('musicxml', fp=str(out_path))
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


# def normalize_musicxml_corpus_teimus(
#     data_dir,
#     output_dir,
#     allowed_durations,
#     overwrite=False,
#     delete_grace_notes=True,
#     midi_min=57,
#     midi_max=83,
#     create_title=False,
#     respect_time_signature_changes=False
# ):
#     """
#     Limpieza y normalización métrica y rítmica avanzada de corpus MusicXML.
#
#     Fix importante:
#     - Las grace notes NO se cuantizan, NO cuentan para el tiempo del compás y NO deben avanzar offsets.
#       Si se preservan (delete_grace_notes=False), se insertan como objetos grace puros (duración métrica 0).
#     """
#
#     from pathlib import Path
#     from fractions import Fraction
#     from statistics import mean
#     from music21 import converter, stream, note, chord, meter, key, metadata, harmony
#     from music21 import pitch as m21pitch
#     # -------------------------------------------------------------------------
#     # Helpers
#     # -------------------------------------------------------------------------
#     def list_musicxml_files(root: Path):
#         return [p for p in root.rglob("*")
#                 if p.is_file() and p.suffix.lower() in (".xml", ".mxl", ".musicxml")]
#
#     def ql_to_frac(x: float) -> Fraction:
#         return Fraction(x).limit_denominator(96)
#
#     allowed_fracs = [ql_to_frac(d) for d in allowed_durations]
#
#     def closest_allowed(frac_dur: Fraction) -> Fraction:
#         return min(allowed_fracs, key=lambda a: abs(a - frac_dur))
#
#     def expected_len_from_ts(ts: meter.TimeSignature) -> Fraction:
#         return ql_to_frac(ts.numerator * (4 / ts.denominator))
#
#     def estimate_time_signature(measure_lengths_frac):
#         candidates = [(2, 4), (3, 4), (4, 4), (6, 4), (6, 8), (3, 8)]
#         avg_len = mean([float(x) for x in measure_lengths_frac])
#
#         def bar_len(n, d):
#             return n * (4 / d)
#
#         num, den = min(candidates, key=lambda s: abs(avg_len - bar_len(*s)))
#         return meter.TimeSignature(f"{num}/{den}")
#
#     def looks_like_grace(el) -> bool:
#         try:
#             if hasattr(el, "duration") and getattr(el.duration, "isGrace", False):
#                 return True
#         except Exception:
#             pass
#         try:
#             if hasattr(el, "quarterLength") and float(el.quarterLength) == 0.0:
#                 return True
#         except Exception:
#             pass
#         return False
#
#     def clamp_midi(midi_val: int) -> int:
#         while midi_val < midi_min:
#             midi_val += 12
#         while midi_val > midi_max:
#             midi_val -= 12
#         return midi_val
#
#     def clamp_pitch_octave(p: m21pitch.Pitch, midi_min: int, midi_max: int) -> m21pitch.Pitch:
#         """
#         Devuelve un Pitch con el MISMO nombre (step + accidental),
#         ajustando solo la octava para que su MIDI caiga en [midi_min, midi_max].
#         """
#         # new_p = p.clone()
#         new_p = m21pitch.Pitch(p.nameWithOctave)
#
#         while new_p.midi < midi_min:
#             new_p.octave += 1
#
#         while new_p.midi > midi_max:
#             new_p.octave -= 1
#
#         return new_p
#
#     # -------------------------------------------------------------------------
#     # Main
#     # -------------------------------------------------------------------------
#     data_dir = Path(data_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     files = list_musicxml_files(data_dir)
#     print(f"\n🎼 Archivos detectados: {len(files)}\n")
#
#     total_read = total_processed = total_ignored = total_errors = total_saved = 0
#
#     for src in files:
#         print(f"──────────────────────────────────────────────")
#         print(f"Procesando: {src.name}")
#
#         out_path = output_dir / src.name
#         if out_path.exists() and not overwrite:
#             print(f"⏭️ Ya existe en salida y overwrite=False → omitido: {out_path.name}")
#             total_ignored += 1
#             continue
#
#         try:
#             score = converter.parse(src)
#             total_read += 1
#         except Exception as e:
#             print(f"⛔ Error al leer {src.name}: {e}")
#             total_errors += 1
#             continue
#
#         new_score = stream.Score()
#         new_part = stream.Part()
#         new_score.insert(0, new_part)
#
#         if create_title:
#             if new_score.metadata is None:
#                 new_score.insert(0, metadata.Metadata())
#             new_score.metadata.title = src.stem
#
#         try:
#             orig_key = score.recurse().getElementsByClass(key.KeySignature)[0]
#             new_part.insert(0.0, key.KeySignature(orig_key.sharps))
#             print(f"🎯 Armadura copiada: {orig_key.sharps} alteraciones.")
#         except Exception:
#             print("⚠️ Sin armadura detectada (se omitirá).")
#
#         try:
#             orig_part = score.parts[0]
#         except Exception:
#             total_ignored += 1
#             continue
#
#         orig_measures = list(orig_part.getElementsByClass(stream.Measure))
#         if not orig_measures:
#             total_ignored += 1
#             continue
#
#         # --- mapear TimeSignature por compás ---
#         measure_ts_map = {}
#         current_ts = None
#         for m in orig_measures:
#             ts = m.getTimeSignatures()
#             if ts:
#                 current_ts = ts[0]
#             measure_ts_map[m.number] = current_ts
#
#         # --- limpiar compases vacíos ---
#         while orig_measures and not any(isinstance(el, (note.Note, chord.Chord))
#                                         for el in orig_measures[0].notesAndRests):
#             orig_measures.pop(0)
#
#         while orig_measures and not any(isinstance(el, (note.Note, chord.Chord))
#                                         for el in orig_measures[-1].notesAndRests):
#             orig_measures.pop(-1)
#
#         if not orig_measures:
#             total_ignored += 1
#             continue
#
#         # --- estimar compás global ---
#         measure_lengths = []
#         for m in orig_measures:
#             total = 0.0
#             for el in m.notesAndRests:
#                 if isinstance(el, harmony.ChordSymbol) or looks_like_grace(el):
#                     continue
#                 total += float(el.quarterLength)
#             if total > 0:
#                 measure_lengths.append(ql_to_frac(total))
#
#         if not measure_lengths:
#             total_ignored += 1
#             continue
#
#         estimated_ts = estimate_time_signature(measure_lengths)
#
#         try:
#             read_ts = score.recurse().getElementsByClass(meter.TimeSignature)[0]
#             ts_to_use = estimated_ts if abs(
#                 float(expected_len_from_ts(read_ts)) -
#                 float(mean([float(x) for x in measure_lengths]))
#             ) > 0.25 else read_ts
#         except Exception:
#             ts_to_use = estimated_ts
#
#         if not respect_time_signature_changes:
#             new_part.insert(0.0, meter.TimeSignature(ts_to_use.ratioString))
#             exp_len_global = expected_len_from_ts(ts_to_use)
#         else:
#             exp_len_global = None
#             print("🔀 Respetando cambios de compás originales.")
#
#         # ---------------------------------------------------------------------
#         # Procesar compases
#         # ---------------------------------------------------------------------
#         for mi, m in enumerate(orig_measures, start=1):
#             new_measure = stream.Measure(number=m.number)
#
#             if respect_time_signature_changes:
#                 ts_local = measure_ts_map.get(m.number) or ts_to_use
#                 exp_len = expected_len_from_ts(ts_local)
#                 new_measure.insert(0.0, meter.TimeSignature(ts_local.ratioString))
#             else:
#                 exp_len = exp_len_global
#
#             elems = []
#
#             for el in m.notesAndRests:
#                 if isinstance(el, harmony.ChordSymbol):
#                     continue
#
#                 if looks_like_grace(el):
#                     if delete_grace_notes:
#                         continue
#                     if isinstance(el, note.Note):
#                         # midi_val = clamp_midi(int(el.pitch.midi))
#                         p = clamp_pitch_octave(el.pitch, midi_min, midi_max)
#                         tipo = el.duration.type
#                         if tipo == "zero":
#                             tipo = '16th'
#                         gn = note.Note(p, type=tipo).getGrace()
#                         gn.duration.slash = True
#                         elems.append(gn)
#
#                     continue
#
#                 dur_frac = ql_to_frac(float(el.quarterLength))
#                 dur_q = closest_allowed(dur_frac)
#
#                 if isinstance(el, note.Rest):
#                     elems.append(note.Rest(quarterLength=float(dur_q)))
#                 elif isinstance(el, note.Note):
#                     # midi_val = clamp_midi(int(el.pitch.midi))
#                     # elems.append(note.Note(midi_val, quarterLength=float(dur_q)))
#                     p = clamp_pitch_octave(el.pitch, midi_min, midi_max)
#                     elems.append(note.Note(p, quarterLength=float(dur_q)))
#                 elif isinstance(el, chord.Chord):
#                     top = max(el.pitches, key=lambda p: p.midi)
#                     elems.append(note.Note(clamp_midi(int(top.midi)),
#                                            quarterLength=float(dur_q)))
#
#             curr = 0.0
#             for e in elems:
#                 new_measure.insert(curr, e)
#                 if not getattr(e.duration, "isGrace", False):
#                     curr += float(e.quarterLength)
#
#             new_part.append(new_measure)
#
#         try:
#             new_score.write('musicxml', fp=str(out_path))
#             total_saved += 1
#             total_processed += 1
#         except Exception:
#             total_errors += 1
#
#     print("\n═══════════════════════════════════════════════")
#     print("📊 INFORME FINAL")
#     print(f" Archivos encontrados:   {len(files)}")
#     print(f" Leídos correctamente:   {total_read}")
#     print(f" Procesados y guardados: {total_processed}")
#     print(f" Ignorados:              {total_ignored}")
#     print(f" Errores:                {total_errors}")
#     print(f" Archivos escritos:      {total_saved}")
#     print("═══════════════════════════════════════════════")
#



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
        (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(9,4),(10,4),
        (1,8),(3,8),(5,8),(6,8),(7,8),(9,8),(11,8),(12,8),(13,8),
        (14,8),(15,8),(17,8),(21,8),(23,8)
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
    # Preservar grace: NO asignar duraciones métricas,
    # NO tocar allowed_durations,
    # NO forzar duration.type.
    if isinstance(el, note.Note):
        p = clamp_pitch_octave(el.pitch, midi_min, midi_max)
        # tipo = el.duration.type
        # if tipo == "zero":
        #     tipo = '16th'
        tipo = '16th' #Forzamos la duracion del grace a semicorcheas
        gn = note.Note(p, type=tipo).getGrace()
        gn.duration.slash = True  # barra diagonal
        return gn
    elif isinstance(el, chord.Chord):
        top_pitch = max(el.pitches, key=lambda p: p.midi)
        midi_val = clamp_midi(midi_min, midi_max, int(top_pitch.midi))
        gn = note.Note(midi_val).getGrace()
        return gn
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
        return gn
    # rests grace raros → ignorar

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
        return note.Note(midi_val, quarterLength=float(dur_q))

    elif isinstance(el, note.Unpitched):
        midi_val = el.displayName
        return note.Note(midi_val, quarterLength=float(dur_q))

def fill_anacruse(elems, exp_len):
    rests = [e for e in elems if isinstance(e, note.Rest)]
    notes = [e for e in elems if isinstance(e, note.Note)]

    #Comprueba si a la anacrusa le faltan silencios
    sum = 0
    for rest in rests:
        sum = sum + rest.quarterLength
    for nota in notes:
        sum = sum + nota.quarterLength
    dif = exp_len-sum
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
        new_score.insert(0, new_part)

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
            orig_measures = trim_score2(orig_measures)
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
            print(f"⚠️ {src.name}: Error estimando el compás global ({e}). Omitido.")
            total_ignored += 1
            continue


        modified = False

        # ---------------------------------------------------------------------
        # 7. Procesar todos los compases
        # ---------------------------------------------------------------------
        for mi, m in enumerate(orig_measures, start=1):
            new_measure = stream.Measure(number=m.number)
            elems = []

            #Detectamos cambios de Time signature
            try:
                if len(m.flat.getElementsByClass("TimeSignature"))==0:
                    #Si el compás no tiene definida el time signature, asignamos el estimado
                    ts_mes=time_signatures[mi-1]
                else:
                    ts_mes = m.flat.getElementsByClass("TimeSignature")[0]
                # if ts_to_use.ratioString != ts_mes.ratioString:
                #     ts_to_use=ts_mes
                #     new_measure.insert(0.0, meter.TimeSignature(ts_to_use.ratioString))
                #     # print(f"✅ Cambio de tonalidad detectado: {ts_to_use.ratioString}")
            except Exception as e:
                print(f"Error al acceder a la time signature {e}")
                pass
            new_measure.insert(0.0, meter.TimeSignature(ts_mes.ratioString))
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
            # 9. PRIMER COMPÁS: revisar ANACRUSA
            # =====================================================
            if mi == 1 and len(elems) > 1:
                if fill_anacruse(elems, exp_len):
                    modified = True


            # 10. Calcular la Longitud total del compas (sin contar graces) ---
            total_len = sum_total_len_mesure(elems)


            for e in elems:
                new_measure.append(e)


            # # 11. Comprobar que la anacrusa esté completa (solo con duración real) ---
            # if mi == 1 and total_len < exp_len:
            #     missing = exp_len - total_len
            #     print(f"🟢 Anacrusa incompleta detectada: rellenando +{float(missing):.3f}")
            #     r = note.Rest(quarterLength=float(missing))
            #     new_measure.insert(0.0, r)
            #
            #     offset = float(missing)
            #     for e in elems:
            #         e2 = copy.deepcopy(e)
            #         new_measure.insert(offset, e2)
            #         # grace no avanza offset
            #         try:
            #             if getattr(e.duration, "isGrace", False):
            #                 continue
            #         except Exception:
            #             pass
            #         offset += float(e.quarterLength)

                modified = True
            # else:
            #     curr = 0.0
            #     for e in elems:
            #         new_measure.insert(curr, e)
            #         # grace no avanza el tiempo
            #         try:
            #             if getattr(e.duration, "isGrace", False):
            #                 continue
            #         except Exception:
            #             pass
            #         curr += float(e.quarterLength)
            #
            # # --- Recalcular longitud total del compás (sin graces) ---
            # total_len = Fraction(0, 1)
            # for e in new_measure.notesAndRests:
            #     try:
            #         if getattr(e.duration, "isGrace", False):
            #             continue
            #     except Exception:
            #         pass
            #     total_len += ql_to_frac(e.quarterLength)
            #
            # # --- Compás corto: completar al final (solo si hay algo real) ---
            # if total_len < exp_len and len([x for x in new_measure.notesAndRests
            #                                if not getattr(x.duration, "isGrace", False)]) > 0:
            #     diff = exp_len - total_len
            #
            #     # buscar el último elemento NO-grace
            #     items = list(new_measure.notesAndRests)
            #     last_real = None
            #     for it in reversed(items):
            #         if not getattr(it.duration, "isGrace", False):
            #             last_real = it
            #             break
            #
            #     if last_real is not None:
            #         if isinstance(last_real, note.Note):
            #             pitch = last_real.pitch.midi
            #             new_el = note.Note(pitch, quarterLength=float(diff))
            #         else:
            #             new_el = note.Rest(quarterLength=float(diff))
            #
            #         new_measure.insert(float(total_len), new_el)
            #         total_len += diff
            #         modified = True
            #         print(f"🟡 c.{m.number}: añadido elemento para completar {float(diff):.3f}")
            #
            # # --- Compás largo: recortar/eliminar elementos al final (solo reales) ---
            # while total_len > exp_len:
            #     exceso = total_len - exp_len
            #
            #     # último elemento NO-grace
            #     items = list(new_measure.notesAndRests)
            #     last_real = None
            #     for it in reversed(items):
            #         if not getattr(it.duration, "isGrace", False):
            #             last_real = it
            #             break
            #
            #     if last_real is None:
            #         break
            #
            #     last_len = ql_to_frac(last_real.quarterLength)
            #
            #     if last_len > exceso + Fraction(1, 192):
            #         last_real.quarterLength = float(last_len - exceso)
            #         total_len = exp_len
            #         print(f"🟠 c.{m.number}: recortado -{float(exceso):.3f}")
            #         modified = True
            #     else:
            #         new_measure.remove(last_real)
            #         total_len -= last_len
            #         print(f"🔴 c.{m.number}: eliminada figura excedente {float(last_len):.3f}")
            #         modified = True

            new_part.append(new_measure)

        # --- Guardar ---
        try:
            new_score.write('musicxml', fp=str(out_path))
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




