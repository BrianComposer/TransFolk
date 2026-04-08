import os
import json
from collections import defaultdict
from music21 import converter, stream


def count_ts_mode_distribution(
    corpus_path: str,
    output_dir: str = None,
    save_json: bool = True
):
    """
    Recorre un corpus MusicXML y calcula:
    1) distribución absoluta de (TS, MODE)
    2) distribución normalizada (probabilidades)

    Además:
    - muestra progreso en tiempo real (%)
    - guarda ambos diccionarios en JSON
    - devuelve el diccionario normalizado

    Parámetros
    ----------
    corpus_path : str
        Ruta al corpus
    output_dir : str
        Carpeta donde guardar JSON (por defecto: corpus_path)
    save_json : bool
        Si guardar o no los JSON

    Retorna
    -------
    dict
        Diccionario normalizado {(ts, mode): prob}
    """

    valid_ext = (".xml", ".musicxml", ".mxl")
    distribution = defaultdict(int)

    if output_dir is None:
        output_dir = corpus_path

    # --------------------------------------------------
    # 1. Recopilar lista de archivos
    # --------------------------------------------------
    all_files = []
    for root, _, files in os.walk(corpus_path):
        for f in files:
            if f.lower().endswith(valid_ext):
                all_files.append(os.path.join(root, f))

    total_files = len(all_files)
    print(f"[INFO] Archivos encontrados: {total_files}")

    if total_files == 0:
        print("[WARNING] No se encontraron archivos XML")
        return {}

    # --------------------------------------------------
    # 2. Procesamiento con progreso
    # --------------------------------------------------
    for idx, file_path in enumerate(all_files):

        progress = (idx + 1) / total_files * 100
        print(f"\rProcesando: {progress:.2f}% ({idx+1}/{total_files})", end="")

        try:
            score = converter.parse(file_path)
        except Exception as e:
            print(f"\n[ERROR] Parseando {file_path}: {e}")
            continue

        try:
            parts = score.parts if score.parts else [score]

            for part in parts:
                measures = part.getElementsByClass(stream.Measure)

                current_ts = None
                current_mode = None

                for m in measures:

                    # TIME SIGNATURE
                    if m.timeSignature is not None:
                        ts = m.timeSignature
                        current_ts = f"{ts.numerator}/{ts.denominator}"

                    # KEY SIGNATURE
                    if m.keySignature is not None:
                        try:
                            key = m.keySignature.asKey()
                            if key.mode in ["major", "minor"]:
                                current_mode = key.mode
                        except Exception:
                            pass

                    # fallback análisis
                    if current_mode is None:
                        try:
                            key = m.analyze("key")
                            if key.mode in ["major", "minor"]:
                                current_mode = key.mode
                        except Exception:
                            pass

                    # conteo
                    if current_ts is not None and current_mode is not None:
                        distribution[(current_ts, current_mode)] += 1

        except Exception as e:
            print(f"\n[ERROR] Procesando {file_path}: {e}")
            continue

    print("\n[INFO] Procesamiento completado")

    # --------------------------------------------------
    # 3. Normalización
    # --------------------------------------------------
    total_counts = sum(distribution.values())

    if total_counts == 0:
        print("[WARNING] No se han contabilizado eventos válidos")
        return {}

    normalized = {
        k: v / total_counts
        for k, v in distribution.items()
    }

    # --------------------------------------------------
    # 4. Preparar formato JSON (claves como string)
    # --------------------------------------------------
    def serialize_keys(d):
        return {f"{k[0]}|{k[1]}": v for k, v in d.items()}

    distribution_json = serialize_keys(distribution)
    normalized_json = serialize_keys(normalized)

    # --------------------------------------------------
    # 5. Guardado
    # --------------------------------------------------
    if save_json:
        os.makedirs(output_dir, exist_ok=True)

        dist_path = os.path.join(output_dir, "ts_mode_distribution.json")
        norm_path = os.path.join(output_dir, "ts_mode_distribution_normalized.json")

        with open(dist_path, "w", encoding="utf-8") as f:
            json.dump(distribution_json, f, indent=4)

        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(normalized_json, f, indent=4)

        print(f"[INFO] Guardado en:\n - {dist_path}\n - {norm_path}")

    return normalized



def load_ts_mode_distribution(json_path: str):
    """
    Carga un JSON de distribución TS-MODE y reconstruye
    las claves como tuplas.

    Retorna
    -------
    dict {(ts, mode): value}
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reconstructed = {}
    for k, v in data.items():
        ts, mode = k.split("|")
        reconstructed[(ts, mode)] = v

    return reconstructed