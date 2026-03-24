from sympy import false

from generar_allowed_durations import get_allowed_durations
from common import is_grace, check_anacrusis

import os
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from music21 import converter, note, stream, meter, chord
#
# def check_corpus(corpus_path, allowed_durations, log_output_path="corpus_check_log.json"):
#     """
#     Chequeo general del corpus MusicXML.
#
#     Parámetros:
#         corpus_path (str): ruta al corpus
#         allowed_durations (set[float]): duraciones permitidas (quarterLength)
#         log_output_path (str): archivo JSON de salida
#
#     Devuelve:
#         dict con métricas globales
#     """
#
#     global_stats = {
#         "total_files": 0,
#         "xml_not_well_formed": 0,
#         "parser_failures": 0,
#         "scores_without_time_signature": 0,
#         "scores_with_multiple_parts": 0,
#         "scores_with_measure_number_gaps": 0,
#         "scores_with_overflow_measures": 0,
#         "scores_with_invalid_durations": 0,
#         "scores_with_empty_measures": 0,
#         "scores_with_multiple_voices": 0,
#         "scores_with_chords": 0,
#         "total_chords": 0,
#         "total_notes_inside_chords": 0,
#         "total_grace_notes": 0,
#         "total_secondary_voice_events": 0,
#         "total_measures_with_multiple_voices": 0,
#         "time_signature_counter": Counter(),
#         "total_measures": 0,
#         "total_notes": 0,
#         "total_rests": 0,
#     }
#
#     problematic_scores = {}
#
#     for root, _, files in os.walk(corpus_path):
#         for file in files:
#             print(f"Analyzing file: {file}")
#             if not file.lower().endswith((".xml", ".musicxml")):
#                 continue
#
#             filepath = os.path.join(root, file)
#             global_stats["total_files"] += 1
#
#             score_report = {
#                 "file": filepath,
#                 "xml_error": False,
#                 "parser_error": False,
#                 "no_time_signature": False,
#                 "measure_number_gap": False,
#                 "overflow_measures": [],
#                 "invalid_durations": [],
#                 "empty_measures": [],
#                 "multiple_parts": False,
#                 "multiple_voices": False,
#                 "has_chords": False,
#                 "total_chords": 0,
#                 "notes_inside_chords": 0,
#                 "grace_notes": 0,
#             }
#
#             # --------------------------------------------------
#             # 1. XML well formed
#             # --------------------------------------------------
#             try:
#                 ET.parse(filepath)
#             except Exception:
#                 global_stats["xml_not_well_formed"] += 1
#                 score_report["xml_error"] = True
#                 problematic_scores[filepath] = score_report
#                 continue
#
#             # --------------------------------------------------
#             # 2. Parse with music21
#             # --------------------------------------------------
#             try:
#                 score = converter.parse(filepath)
#             except Exception:
#                 global_stats["parser_failures"] += 1
#                 score_report["parser_error"] = True
#                 problematic_scores[filepath] = score_report
#                 continue
#
#             parts = score.parts
#
#             if len(parts) > 1:
#                 global_stats["scores_with_multiple_parts"] += 1
#                 score_report["multiple_parts"] = True
#
#             # --------------------------------------------------
#             # Time signatures
#             # --------------------------------------------------
#             time_sigs = score.recurse().getElementsByClass(meter.TimeSignature)
#
#             if not time_sigs:
#                 global_stats["scores_without_time_signature"] += 1
#                 score_report["no_time_signature"] = True
#             else:
#                 for ts in time_sigs:
#                     global_stats["time_signature_counter"][ts.ratioString] += 1
#
#             if not parts:
#                 problematic_scores[filepath] = score_report
#                 continue
#
#             measures = parts[0].getElementsByClass(stream.Measure)
#
#             previous_number = None
#
#             for m in measures:
#                 global_stats["total_measures"] += 1
#
#                 # -----------------------------------
#                 # Gap numeración compás
#                 # -----------------------------------
#                 if previous_number is not None:
#                     if m.number != previous_number + 1:
#                         score_report["measure_number_gap"] = True
#                 previous_number = m.number
#
#                 # -----------------------------------
#                 # Compás vacío
#                 # -----------------------------------
#                 if not any(el for el in m.notesAndRests):
#                     score_report["empty_measures"].append(m.number)
#
#                 # -----------------------------------
#                 # Time signature vigente
#                 # -----------------------------------
#                 ts = m.timeSignature
#                 expected_duration = None
#                 if ts is not None:
#                     expected_duration = ts.barDuration.quarterLength
#
#                 real_duration = 0.0
#
#                 # -----------------------------------
#                 # Elementos del compás
#                 # -----------------------------------
#                 for el in m.notesAndRests:
#
#                     ql = round(float(el.quarterLength), 6)
#                     real_duration += ql
#
#                     if isinstance(el, note.Note):
#                         global_stats["total_notes"] += 1
#
#                     elif isinstance(el, note.Rest):
#                         global_stats["total_rests"] += 1
#
#                     elif isinstance(el, chord.Chord):
#                         score_report["has_chords"] = True
#                         score_report["total_chords"] += 1
#                         score_report["notes_inside_chords"] += len(el.pitches)
#
#                         global_stats["total_chords"] += 1
#                         global_stats["total_notes_inside_chords"] += len(el.pitches)
#
#                     if is_grace(el):
#                         score_report["grace_notes"] += 1
#                         global_stats["total_grace_notes"] += 1
#
#                     if ql not in allowed_durations:
#                         score_report["invalid_durations"].append(
#                             {"measure": m.number, "duration": ql}
#                         )
#
#                 # -----------------------------------
#                 # Overflow métrico
#                 # -----------------------------------
#                 if expected_duration is not None:
#                     if real_duration > expected_duration + 0.01:
#                         score_report["overflow_measures"].append(
#                             {
#                                 "measure": m.number,
#                                 "real_duration": real_duration,
#                                 "expected_duration": expected_duration,
#                             }
#                         )
#
#                 # -----------------------------------
#                 # DETECCIÓN ROBUSTA DE VOCES
#                 # -----------------------------------
#                 voices_in_measure = m.getElementsByClass(stream.Voice)
#
#                 if len(voices_in_measure) > 1:
#                     global_stats["total_measures_with_multiple_voices"] += 1
#
#                     for v in voices_in_measure[1:]:
#                         events = list(v.recurse().notesAndRests)
#
#                         musical_events = [
#                             e for e in events
#                             if isinstance(e, (note.Note, chord.Chord))
#                         ]
#
#                         if musical_events:
#                             score_report["multiple_voices"] = True
#                             global_stats["total_secondary_voice_events"] += len(musical_events)
#
#             # --------------------------------------------------
#             # Agregación por partitura
#             # --------------------------------------------------
#             if score_report["measure_number_gap"]:
#                 global_stats["scores_with_measure_number_gaps"] += 1
#
#             if score_report["overflow_measures"]:
#                 global_stats["scores_with_overflow_measures"] += 1
#
#             if score_report["invalid_durations"]:
#                 global_stats["scores_with_invalid_durations"] += 1
#
#             if score_report["empty_measures"]:
#                 global_stats["scores_with_empty_measures"] += 1
#
#             if score_report["multiple_voices"]:
#                 global_stats["scores_with_multiple_voices"] += 1
#
#             if score_report["has_chords"]:
#                 global_stats["scores_with_chords"] += 1
#
#             if (
#                 score_report["xml_error"]
#                 or score_report["parser_error"]
#                 or score_report["no_time_signature"]
#                 or score_report["measure_number_gap"]
#                 or score_report["overflow_measures"]
#                 or score_report["invalid_durations"]
#                 or score_report["empty_measures"]
#                 or score_report["multiple_voices"]
#                 or score_report["has_chords"]
#                 or score_report["multiple_parts"]
#             ):
#                 problematic_scores[filepath] = score_report
#
#     # --------------------------------------------------
#     # Guardar log
#     # --------------------------------------------------
#     final_log = {
#         "global_stats": {
#             **global_stats,
#             "time_signature_counter": dict(global_stats["time_signature_counter"]),
#         },
#         "problematic_scores": problematic_scores,
#     }
#
#     with open(log_output_path, "w", encoding="utf-8") as f:
#         json.dump(final_log, f, indent=4)
#
#     return final_log


#
# def check_corpus(corpus_path, allowed_durations, log_output_path="corpus_check_log.json"):
#
#     # ----------------------------
#     # CONTADORES NUMÉRICOS
#     # ----------------------------
#     global_counts = {
#         "total_files": 0,
#         "xml_not_well_formed": 0,
#         "parser_failures": 0,
#         "scores_without_time_signature": 0,
#         "scores_with_multiple_parts": 0,
#         "scores_with_measure_number_gaps": 0,
#         "scores_with_overflow_measures": 0,
#         "scores_with_invalid_durations": 0,
#         "scores_with_empty_measures": 0,
#         "scores_with_multiple_voices": 0,
#         "scores_with_chords": 0,
#         "total_chords": 0,
#         "total_notes_inside_chords": 0,
#         "total_grace_notes": 0,
#         "total_secondary_voice_events": 0,
#         "total_measures_with_multiple_voices": 0,
#         "time_signature_counter": Counter(),
#         "total_measures": 0,
#         "total_notes": 0,
#         "total_rests": 0,
#     }
#
#     # ----------------------------
#     # LISTADO DE OBRAS POR CATEGORÍA
#     # ----------------------------
#     global_files_by_category = defaultdict(list)
#
#     problematic_scores = {}
#
#     for root, _, files in os.walk(corpus_path):
#         for file in files:
#
#             if not file.lower().endswith((".xml", ".musicxml")):
#                 continue
#
#             print(f"Analyzing file: {file}")
#
#             filepath = os.path.join(root, file)
#             global_counts["total_files"] += 1
#
#             score_report = {
#                 "file": file,
#                 "xml_error": False,
#                 "parser_error": False,
#                 "no_time_signature": False,
#                 "measure_number_gap": False,
#                 "overflow_measures": [],
#                 "invalid_durations": [],
#                 "empty_measures": [],
#                 "multiple_parts": False,
#                 "multiple_voices": False,
#                 "has_chords": False,
#                 "total_chords": 0,
#                 "notes_inside_chords": 0,
#                 "grace_notes": 0,
#             }
#
#             # --------------------------------------------------
#             # XML well formed
#             # --------------------------------------------------
#             try:
#                 ET.parse(filepath)
#             except Exception:
#                 global_counts["xml_not_well_formed"] += 1
#                 global_files_by_category["xml_not_well_formed"].append(file)
#                 score_report["xml_error"] = True
#                 problematic_scores[file] = score_report
#                 continue
#
#             # --------------------------------------------------
#             # Parse music21
#             # --------------------------------------------------
#             try:
#                 score = converter.parse(filepath)
#             except Exception:
#                 global_counts["parser_failures"] += 1
#                 global_files_by_category["parser_failures"].append(file)
#                 score_report["parser_error"] = True
#                 problematic_scores[file] = score_report
#                 continue
#
#             parts = score.parts
#
#             if len(parts) > 1:
#                 global_counts["scores_with_multiple_parts"] += 1
#                 global_files_by_category["scores_with_multiple_parts"].append(file)
#                 score_report["multiple_parts"] = True
#
#             # --------------------------------------------------
#             # Time signatures
#             # --------------------------------------------------
#             time_sigs = score.recurse().getElementsByClass(meter.TimeSignature)
#
#             if not time_sigs:
#                 global_counts["scores_without_time_signature"] += 1
#                 global_files_by_category["scores_without_time_signature"].append(file)
#                 score_report["no_time_signature"] = True
#             else:
#                 for ts in time_sigs:
#                     global_counts["time_signature_counter"][ts.ratioString] += 1
#
#             if not parts:
#                 problematic_scores[file] = score_report
#                 continue
#
#             measures = parts[0].getElementsByClass(stream.Measure)
#             previous_number = None
#
#             for m in measures:
#
#                 global_counts["total_measures"] += 1
#
#                 # Gap numeración
#                 if previous_number is not None:
#                     if m.number != previous_number + 1:
#                         score_report["measure_number_gap"] = True
#                 previous_number = m.number
#
#                 # Compás vacío
#                 if not any(el for el in m.notesAndRests):
#                     score_report["empty_measures"].append(m.number)
#
#                 # Time signature vigente
#                 ts = m.timeSignature
#                 expected_duration = None
#                 if ts is not None:
#                     expected_duration = ts.barDuration.quarterLength
#
#                 real_duration = 0.0
#
#                 for el in m.notesAndRests:
#
#                     ql = round(float(el.quarterLength), 6)
#                     real_duration += ql
#
#                     if isinstance(el, note.Note):
#                         global_counts["total_notes"] += 1
#
#                     elif isinstance(el, note.Rest):
#                         global_counts["total_rests"] += 1
#
#                     elif isinstance(el, chord.Chord):
#                         score_report["has_chords"] = True
#                         score_report["total_chords"] += 1
#                         score_report["notes_inside_chords"] += len(el.pitches)
#
#                         global_counts["total_chords"] += 1
#                         global_counts["total_notes_inside_chords"] += len(el.pitches)
#
#                     if is_grace(el):
#                         score_report["grace_notes"] += 1
#                         global_counts["total_grace_notes"] += 1
#
#                     if ql not in allowed_durations:
#                         score_report["invalid_durations"].append(
#                             {"measure": m.number, "duration": ql}
#                         )
#
#                 # Overflow
#                 if expected_duration is not None:
#                     if real_duration > expected_duration + 0.01:
#                         score_report["overflow_measures"].append(m.number)
#
#                 # Detección robusta voces
#                 voices_in_measure = m.getElementsByClass(stream.Voice)
#
#                 if len(voices_in_measure) > 1:
#                     global_counts["total_measures_with_multiple_voices"] += 1
#
#                     for v in voices_in_measure[1:]:
#                         musical_events = [
#                             e for e in v.recurse()
#                             if isinstance(e, (note.Note, chord.Chord))
#                         ]
#                         if musical_events:
#                             score_report["multiple_voices"] = True
#                             global_counts["total_secondary_voice_events"] += len(musical_events)
#
#             # --------------------------------------------------
#             # Agregación por partitura
#             # --------------------------------------------------
#
#             if score_report["measure_number_gap"]:
#                 global_counts["scores_with_measure_number_gaps"] += 1
#                 global_files_by_category["scores_with_measure_number_gaps"].append(file)
#
#             if score_report["overflow_measures"]:
#                 global_counts["scores_with_overflow_measures"] += 1
#                 global_files_by_category["scores_with_overflow_measures"].append(file)
#
#             if score_report["invalid_durations"]:
#                 global_counts["scores_with_invalid_durations"] += 1
#                 global_files_by_category["scores_with_invalid_durations"].append(file)
#
#             if score_report["empty_measures"]:
#                 global_counts["scores_with_empty_measures"] += 1
#                 global_files_by_category["scores_with_empty_measures"].append(file)
#
#             if score_report["multiple_voices"]:
#                 global_counts["scores_with_multiple_voices"] += 1
#                 global_files_by_category["scores_with_multiple_voices"].append(file)
#
#             if score_report["has_chords"]:
#                 global_counts["scores_with_chords"] += 1
#                 global_files_by_category["scores_with_chords"].append(file)
#
#             if score_report["multiple_parts"]:
#                 global_files_by_category["scores_with_multiple_parts"].append(file)
#
#             if score_report["no_time_signature"]:
#                 global_files_by_category["scores_without_time_signature"].append(file)
#
#             # Guardar detalle individual si tiene cualquier problema
#             if any([
#                 score_report["xml_error"],
#                 score_report["parser_error"],
#                 score_report["no_time_signature"],
#                 score_report["measure_number_gap"],
#                 score_report["overflow_measures"],
#                 score_report["invalid_durations"],
#                 score_report["empty_measures"],
#                 score_report["multiple_voices"],
#                 score_report["has_chords"],
#                 score_report["multiple_parts"],
#             ]):
#                 problematic_scores[file] = score_report
#
#     # --------------------------------------------------
#     # Log final
#     # --------------------------------------------------
#     final_log = {
#         "global_counts": {
#             **global_counts,
#             "time_signature_counter": dict(global_counts["time_signature_counter"]),
#         },
#         "files_by_category": dict(global_files_by_category),
#         "problematic_scores": problematic_scores,
#     }
#
#     with open(log_output_path, "w", encoding="utf-8") as f:
#         json.dump(final_log, f, indent=4)
#
#     return final_log



#
# def check_corpus(corpus_path, allowed_durations, log_output_path="corpus_check_log.json"):
#
#     global_counts = {
#         "total_files": 0,
#         "xml_not_well_formed": 0,
#         "parser_failures": 0,
#         "scores_without_time_signature": 0,
#         "scores_with_multiple_parts": 0,
#         "scores_with_measure_number_gaps": 0,
#         "scores_with_overflow_measures": 0,
#         "scores_with_invalid_durations": 0,
#         "scores_with_empty_measures": 0,
#         "scores_with_multiple_voices": 0,
#         "scores_with_chords": 0,
#         "scores_with_malformed_anacrusis": 0,
#         "total_chords": 0,
#         "total_notes_inside_chords": 0,
#         "total_grace_notes": 0,
#         "total_secondary_voice_events": 0,
#         "total_measures_with_multiple_voices": 0,
#         "time_signature_counter": Counter(),
#         "total_measures": 0,
#         "total_notes": 0,
#         "total_rests": 0,
#     }
#
#     global_files_by_category = defaultdict(list)
#     problematic_scores = {}
#
#     for root, _, files in os.walk(corpus_path):
#         for file in files:
#
#             if not file.lower().endswith((".xml", ".musicxml")):
#                 continue
#
#             print(f"Analyzing file: {file}")
#
#             filepath = os.path.join(root, file)
#             global_counts["total_files"] += 1
#
#             score_report = {
#                 "file": file,
#                 "xml_error": False,
#                 "parser_error": False,
#                 "no_time_signature": False,
#                 "measure_number_gap": False,
#                 "overflow_measures": [],
#                 "invalid_durations": [],
#                 "empty_measures": [],
#                 "multiple_parts": False,
#                 "multiple_voices": False,
#                 "has_chords": False,
#                 "malformed_anacrusis": False,
#                 "total_chords": 0,
#                 "notes_inside_chords": 0,
#                 "grace_notes": 0,
#             }
#
#             # ---------------- XML check ----------------
#             try:
#                 ET.parse(filepath)
#             except Exception:
#                 global_counts["xml_not_well_formed"] += 1
#                 global_files_by_category["xml_not_well_formed"].append(file)
#                 score_report["xml_error"] = True
#                 problematic_scores[file] = score_report
#                 continue
#
#             # ---------------- Parse ----------------
#             try:
#                 score = converter.parse(filepath)
#             except Exception:
#                 global_counts["parser_failures"] += 1
#                 global_files_by_category["parser_failures"].append(file)
#                 score_report["parser_error"] = True
#                 problematic_scores[file] = score_report
#                 continue
#
#             parts = score.parts
#
#             if len(parts) > 1:
#                 global_counts["scores_with_multiple_parts"] += 1
#                 global_files_by_category["scores_with_multiple_parts"].append(file)
#                 score_report["multiple_parts"] = True
#
#             time_sigs = score.recurse().getElementsByClass(meter.TimeSignature)
#
#             if not time_sigs:
#                 global_counts["scores_without_time_signature"] += 1
#                 global_files_by_category["scores_without_time_signature"].append(file)
#                 score_report["no_time_signature"] = True
#             else:
#                 for ts in time_sigs:
#                     global_counts["time_signature_counter"][ts.ratioString] += 1
#
#             if not parts:
#                 problematic_scores[file] = score_report
#                 continue
#
#             measures = parts[0].getElementsByClass(stream.Measure)
#
#             previous_number = None
#
#             for idx, m in enumerate(measures):
#
#                 global_counts["total_measures"] += 1
#
#                 if previous_number is not None:
#                     if m.number != previous_number + 1:
#                         score_report["measure_number_gap"] = True
#                 previous_number = m.number
#
#                 if not any(el for el in m.notesAndRests):
#                     score_report["empty_measures"].append(m.number)
#
#                 ts = m.timeSignature
#                 expected_duration = None
#                 if ts is not None:
#                     expected_duration = ts.barDuration.quarterLength
#
#                 real_duration = 0.0
#                 elements = list(m.notesAndRests)
#
#                 for el in elements:
#
#                     ql = round(float(el.quarterLength), 6)
#                     real_duration += ql
#
#                     if isinstance(el, note.Note):
#                         global_counts["total_notes"] += 1
#                     elif isinstance(el, note.Rest):
#                         global_counts["total_rests"] += 1
#                     elif isinstance(el, chord.Chord):
#                         score_report["has_chords"] = True
#                         score_report["total_chords"] += 1
#                         score_report["notes_inside_chords"] += len(el.pitches)
#                         global_counts["total_chords"] += 1
#                         global_counts["total_notes_inside_chords"] += len(el.pitches)
#
#                     if is_grace(el):
#                         score_report["grace_notes"] += 1
#                         global_counts["total_grace_notes"] += 1
#
#                     if ql not in allowed_durations:
#                         score_report["invalid_durations"].append(
#                             {"measure": m.number, "duration": ql}
#                         )
#
#                 if expected_duration is not None:
#                     if real_duration > expected_duration + 0.01:
#                         score_report["overflow_measures"].append(m.number)
#
#                 # ---------------- DETECCIÓN ANACRUSA MAL FORMADA ----------------
#                 if idx == 0 and expected_duration is not None:
#
#                     if real_duration < expected_duration:
#
#                         if elements:
#                             first_el = elements[0]
#                             last_el = elements[-1]
#
#                             # Silencio al final (mal formada)
#                             if isinstance(last_el, note.Rest):
#                                 score_report["malformed_anacrusis"] = True
#
#                             # Silencio largo al inicio
#                             if isinstance(first_el, note.Rest):
#                                 if first_el.quarterLength > 0.25:
#                                     score_report["malformed_anacrusis"] = True
#
#                     # Overflow en primer compás
#                     if real_duration > expected_duration + 0.01:
#                         score_report["malformed_anacrusis"] = True
#
#                 # ---------------- DETECCIÓN VOCES ----------------
#                 voices_in_measure = m.getElementsByClass(stream.Voice)
#
#                 if len(voices_in_measure) > 1:
#                     global_counts["total_measures_with_multiple_voices"] += 1
#
#                     for v in voices_in_measure[1:]:
#                         musical_events = [
#                             e for e in v.recurse()
#                             if isinstance(e, (note.Note, chord.Chord))
#                         ]
#                         if musical_events:
#                             score_report["multiple_voices"] = True
#                             global_counts["total_secondary_voice_events"] += len(musical_events)
#
#             # ---------------- AGREGACIÓN ----------------
#
#             if score_report["malformed_anacrusis"]:
#                 global_counts["scores_with_malformed_anacrusis"] += 1
#                 global_files_by_category["scores_with_malformed_anacrusis"].append(file)
#
#             if score_report["measure_number_gap"]:
#                 global_counts["scores_with_measure_number_gaps"] += 1
#                 global_files_by_category["scores_with_measure_number_gaps"].append(file)
#
#             if score_report["overflow_measures"]:
#                 global_counts["scores_with_overflow_measures"] += 1
#                 global_files_by_category["scores_with_overflow_measures"].append(file)
#
#             if score_report["invalid_durations"]:
#                 global_counts["scores_with_invalid_durations"] += 1
#                 global_files_by_category["scores_with_invalid_durations"].append(file)
#
#             if score_report["empty_measures"]:
#                 global_counts["scores_with_empty_measures"] += 1
#                 global_files_by_category["scores_with_empty_measures"].append(file)
#
#             if score_report["multiple_voices"]:
#                 global_counts["scores_with_multiple_voices"] += 1
#                 global_files_by_category["scores_with_multiple_voices"].append(file)
#
#             if score_report["has_chords"]:
#                 global_counts["scores_with_chords"] += 1
#                 global_files_by_category["scores_with_chords"].append(file)
#
#             if any(score_report.values()):
#                 problematic_scores[file] = score_report
#
#     final_log = {
#         "global_counts": {
#             **global_counts,
#             "time_signature_counter": dict(global_counts["time_signature_counter"]),
#         },
#         "files_by_category": dict(global_files_by_category),
#         "problematic_scores": problematic_scores,
#     }
#
#     with open(log_output_path, "w", encoding="utf-8") as f:
#         json.dump(final_log, f, indent=4)
#
#     return final_log
#
#
#



def check_corpus(corpus_path, allowed_durations, log_output_path="corpus_check_log.json"):

    EPS = 0.03  # tolerancia métrica

    global_counts = {
        "total_files": 0,
        "xml_not_well_formed": 0,
        "parser_failures": 0,
        "scores_without_time_signature": 0,
        "scores_with_multiple_parts": 0,
        "scores_with_measure_number_gaps": 0,
        "scores_with_overflow_measures": 0,
        "scores_with_invalid_durations": 0,
        "scores_with_empty_measures": 0,
        "scores_with_multiple_voices": 0,
        "scores_with_chords": 0,
        "scores_with_malformed_anacrusis": 0,
        "total_chords": 0,
        "total_notes_inside_chords": 0,
        "total_grace_notes": 0,
        "total_secondary_voice_events": 0,
        "total_measures_with_multiple_voices": 0,
        "time_signature_counter": Counter(),
        "total_measures": 0,
        "total_notes": 0,
        "total_rests": 0,
    }

    global_files_by_category = defaultdict(list)
    problematic_scores = {}

    for root, _, files in os.walk(corpus_path):
        for file in files:

            if not file.lower().endswith((".xml", ".musicxml")):
                continue

            print(f"Analyzing file: {file}")

            filepath = os.path.join(root, file)
            global_counts["total_files"] += 1

            score_report = {
                "file": file,
                "xml_error": False,
                "parser_error": False,
                "no_time_signature": False,
                "measure_number_gap": False,
                "overflow_measures": [],
                "invalid_durations": [],
                "empty_measures": [],
                "multiple_parts": False,
                "multiple_voices": False,
                "has_chords": False,
                "malformed_anacrusis": False,
                "total_chords": 0,
                "notes_inside_chords": 0,
                "grace_notes": 0,
            }

            # XML check
            try:
                ET.parse(filepath)
            except Exception:
                global_counts["xml_not_well_formed"] += 1
                global_files_by_category["xml_not_well_formed"].append(file)
                score_report["xml_error"] = True
                problematic_scores[file] = score_report
                continue

            # Parse
            try:
                score = converter.parse(filepath)
            except Exception:
                global_counts["parser_failures"] += 1
                global_files_by_category["parser_failures"].append(file)
                score_report["parser_error"] = True
                problematic_scores[file] = score_report
                continue

            parts = score.parts

            if len(parts) > 1:
                global_counts["scores_with_multiple_parts"] += 1
                global_files_by_category["scores_with_multiple_parts"].append(file)
                score_report["multiple_parts"] = True

            time_sigs = score.recurse().getElementsByClass(meter.TimeSignature)

            if not time_sigs:
                global_counts["scores_without_time_signature"] += 1
                global_files_by_category["scores_without_time_signature"].append(file)
                score_report["no_time_signature"] = True
            else:
                for ts in time_sigs:
                    global_counts["time_signature_counter"][ts.ratioString] += 1

            if not parts:
                problematic_scores[file] = score_report
                continue

            measures = parts[0].getElementsByClass(stream.Measure)

            previous_number = None

            for idx, m in enumerate(measures):

                global_counts["total_measures"] += 1

                if previous_number is not None:
                    if m.number != previous_number + 1:
                        score_report["measure_number_gap"] = True
                previous_number = m.number

                if not any(el for el in m.notesAndRests):
                    score_report["empty_measures"].append(m.number)

                ts = m.timeSignature
                expected_duration = None
                if ts is not None:
                    expected_duration = ts.barDuration.quarterLength

                real_duration = 0.0
                elements = list(m.notesAndRests)

                contains_tuplet = False

                for el in elements:

                    ql = round(float(el.quarterLength), 6)
                    real_duration += ql

                    # detectar tuplets reales
                    if el.duration.tuplets:
                        contains_tuplet = True

                    if isinstance(el, note.Note):
                        global_counts["total_notes"] += 1
                    elif isinstance(el, note.Rest):
                        global_counts["total_rests"] += 1
                    elif isinstance(el, chord.Chord):
                        score_report["has_chords"] = True
                        score_report["total_chords"] += 1
                        score_report["notes_inside_chords"] += len(el.pitches)
                        global_counts["total_chords"] += 1
                        global_counts["total_notes_inside_chords"] += len(el.pitches)

                    if is_grace(el):
                        score_report["grace_notes"] += 1
                        global_counts["total_grace_notes"] += 1


                    # if not is_grace(el) and ql not in allowed_durations:
                    #     score_report["invalid_durations"].append(
                    #         {"measure": m.number, "duration": ql}
                    #     )
                    if not is_grace(el):
                        valid = false
                        for dur in allowed_durations:
                            if abs(dur - ql) < EPS*ql:
                                valid=True
                        if not valid:
                            score_report["invalid_durations"].append(
                                {"measure": m.number, "duration": ql}
                            )



                # Overflow
                if expected_duration is not None:
                    if real_duration > expected_duration + EPS:
                        score_report["overflow_measures"].append(m.number)

                # # ---------------- ANACRUSA CORREGIDA ----------------
                # if idx == 0 and expected_duration is not None:
                #
                #     diff = expected_duration - real_duration
                #
                #     # Caso anacrusa incompleta real
                #     if diff > EPS:
                #
                #         # Si hay tuplets, no marcar automáticamente
                #         if not contains_tuplet:
                #
                #             if elements:
                #                 first_el = elements[0]
                #                 last_el = elements[-1]
                #
                #                 # Silencio final = mal formada
                #                 if isinstance(last_el, note.Rest):
                #                     score_report["malformed_anacrusis"] = True
                #
                #                 # Silencio largo inicial
                #                 if isinstance(first_el, note.Rest) and first_el.quarterLength > 0.5:
                #                     score_report["malformed_anacrusis"] = True
                #
                #     # Overflow claro en compás 1
                #     if real_duration > expected_duration + EPS:
                #         score_report["malformed_anacrusis"] = True

                # ---------------- VOCES ----------------
                voices_in_measure = m.getElementsByClass(stream.Voice)

                if len(voices_in_measure) > 1:
                    global_counts["total_measures_with_multiple_voices"] += 1

                    for v in voices_in_measure[1:]:
                        musical_events = [
                            e for e in v.recurse()
                            if isinstance(e, (note.Note, chord.Chord))
                        ]
                        if musical_events:
                            score_report["multiple_voices"] = True
                            global_counts["total_secondary_voice_events"] += len(musical_events)


            if check_anacrusis(score):
                score_report["malformed_anacrusis"] = True

            # ---------------- AGREGACIÓN ----------------

            if score_report["malformed_anacrusis"]:
                global_counts["scores_with_malformed_anacrusis"] += 1
                global_files_by_category["scores_with_malformed_anacrusis"].append(file)

            if score_report["measure_number_gap"]:
                global_counts["scores_with_measure_number_gaps"] += 1
                global_files_by_category["scores_with_measure_number_gaps"].append(file)

            if score_report["overflow_measures"]:
                global_counts["scores_with_overflow_measures"] += 1
                global_files_by_category["scores_with_overflow_measures"].append(file)

            if score_report["invalid_durations"]:
                global_counts["scores_with_invalid_durations"] += 1
                global_files_by_category["scores_with_invalid_durations"].append(file)

            if score_report["empty_measures"]:
                global_counts["scores_with_empty_measures"] += 1
                global_files_by_category["scores_with_empty_measures"].append(file)

            if score_report["multiple_voices"]:
                global_counts["scores_with_multiple_voices"] += 1
                global_files_by_category["scores_with_multiple_voices"].append(file)

            if score_report["has_chords"]:
                global_counts["scores_with_chords"] += 1
                global_files_by_category["scores_with_chords"].append(file)

            if any(score_report.values()):
                problematic_scores[file] = score_report

    # final_log = {
    #     "global_counts": {
    #         **global_counts,
    #         "time_signature_counter": dict(global_counts["time_signature_counter"]),
    #     },
    #     "files_by_category": dict(global_files_by_category),
    #     "problematic_scores": problematic_scores,
    # }

    final_log = {
        "global_counts": {
            **global_counts,
            "time_signature_counter": dict(global_counts["time_signature_counter"]),
        },
        "files_by_category": dict(global_files_by_category)
    }



    with open(log_output_path, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=4)

    return final_log







if __name__ == "__main__":

    duraciones = [-2, -1, 0, 1, 2, 3, 4] #duraciones
    puntillos = [0,1,2] #puntillos
    grupos = [[1,1], [3,2], [5,4]]
    allowed_durs = get_allowed_durations(duraciones, puntillos, grupos)

    corpus= "todos_clean"
    check_corpus(rf"D:\BackUpDrive\Programacion\Python\TransFolk\data\{corpus}",
                 allowed_durs,
                 log_output_path=f"corpus_check_log_{corpus}.json")