from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from music21 import converter


def _sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name[:120] if name else "untitled"


def _split_abc_tunes(abc_text: str) -> List[str]:
    """
    Divide el contenido completo de un archivo ABC en tunes individuales.
    Cada tune suele empezar por una línea 'X:'.
    """
    lines = abc_text.splitlines()
    tunes: List[List[str]] = []
    current_tune: List[str] = []

    for line in lines:
        if line.startswith("X:"):
            if current_tune:
                tunes.append(current_tune)
            current_tune = [line]
        else:
            if current_tune:
                current_tune.append(line)

    if current_tune:
        tunes.append(current_tune)

    return [
        "\n".join(tune).strip() + "\n"
        for tune in tunes
        if any(line.strip() for line in tune)
    ]


def _extract_title(abc_tune: str) -> str:
    """
    Extrae el título de la tune a partir de la primera línea 'T:'.
    """
    for line in abc_tune.splitlines():
        if line.startswith("T:"):
            return line[2:].strip()
    return "untitled"


def _convert_abc_tune_to_musicxml(abc_tune: str, output_path: Path) -> Tuple[bool, str]:
    """
    Convierte una tune ABC individual a MusicXML.
    """
    try:
        score = converter.parseData(abc_tune, format="abc")
        score.write("musicxml", fp=str(output_path))
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def convert_abc_folder_to_musicxml(input_dir: str | Path, output_dir: str | Path) -> dict:
    """
    Convierte todos los archivos .abc de una carpeta a MusicXML.

    Cada archivo .abc puede contener múltiples tunes. La función separa
    cada tune interna y la exporta como un fichero .musicxml independiente.

    Parámetros
    ----------
    input_dir : str | Path
        Ruta a la carpeta que contiene archivos .abc
    output_dir : str | Path
        Ruta a la carpeta donde se guardarán los .musicxml

    Retorna
    -------
    dict
        Resumen del proceso con contadores y errores.
    """
    print("\n" + "=" * 70)
    print("INICIO DE CONVERSIÓN ABC -> MUSICXML")
    print("=" * 70)

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    print(f"[INFO] Carpeta de entrada: {input_dir}")
    print(f"[INFO] Carpeta de salida : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Carpeta de salida preparada")

    if not input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"La ruta de entrada no es una carpeta: {input_dir}")

    abc_files = sorted(input_dir.glob("*.abc"))
    if not abc_files:
        raise FileNotFoundError(f"No se encontraron archivos .abc en: {input_dir}")

    print(f"[INFO] Archivos .abc encontrados: {len(abc_files)}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "abc_files_processed": 0,
        "tunes_detected": 0,
        "converted_ok": 0,
        "failed": 0,
        "errors": [],
    }

    for file_idx, abc_file in enumerate(abc_files, start=1):
        print("\n" + "-" * 70)
        print(f"[INFO] Procesando archivo {file_idx}/{len(abc_files)}: {abc_file.name}")
        print("-" * 70)

        summary["abc_files_processed"] += 1

        abc_text = abc_file.read_text(encoding="utf-8", errors="ignore")
        print(f"[OK] Archivo leído correctamente")

        tunes = _split_abc_tunes(abc_text)
        print(f"[INFO] Tunes detectadas en {abc_file.name}: {len(tunes)}")

        if not tunes:
            print(f"[WARN] No se detectaron tunes en este archivo")
            summary["errors"].append({
                "file": abc_file.name,
                "tune_index": None,
                "title": None,
                "error": "No se detectaron tunes en este archivo ABC",
            })
            continue

        subfolder = output_dir / abc_file.stem
        subfolder.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Subcarpeta de salida creada: {subfolder}")

        used_names = {}

        for i, tune in enumerate(tunes, start=1):
            summary["tunes_detected"] += 1

            title = _extract_title(tune)
            safe_title = _sanitize_filename(title)

            count = used_names.get(safe_title, 0) + 1
            used_names[safe_title] = count

            if count > 1:
                filename = f"{i:04d}_{safe_title}_{count}.musicxml"
            else:
                filename = f"{i:04d}_{safe_title}.musicxml"

            output_path = subfolder / filename

            print(f"[TUNE {i:04d}] Título detectado: {title}")
            print(f"[TUNE {i:04d}] Exportando a: {output_path.name}")

            ok, msg = _convert_abc_tune_to_musicxml(tune, output_path)

            if ok:
                summary["converted_ok"] += 1
                print(f"[OK] Conversión correcta")
            else:
                summary["failed"] += 1
                print(f"[FAIL] Error en la conversión: {msg}")
                summary["errors"].append({
                    "file": abc_file.name,
                    "tune_index": i,
                    "title": title,
                    "error": msg,
                })

        print(f"[RESUMEN ARCHIVO] {abc_file.name}")
        print(f"  - Tunes procesadas : {len(tunes)}")
        print(f"  - Convertidas OK   : {summary['converted_ok']}")
        print(f"  - Fallidas totales : {summary['failed']}")

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Carpeta entrada       : {summary['input_dir']}")
    print(f"Carpeta salida        : {summary['output_dir']}")
    print(f"Archivos ABC procesados: {summary['abc_files_processed']}")
    print(f"Tunes detectadas      : {summary['tunes_detected']}")
    print(f"Convertidas OK        : {summary['converted_ok']}")
    print(f"Fallidas              : {summary['failed']}")

    if summary["errors"]:
        print("\n[DETALLE DE ERRORES]")
        for err in summary["errors"]:
            print(
                f"  - Archivo: {err['file']} | "
                f"Tune: {err['tune_index']} | "
                f"Título: {err['title']} | "
                f"Error: {err['error']}"
            )
    else:
        print("\n[OK] No se han producido errores")

    print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    summary = convert_abc_folder_to_musicxml(
        input_dir=r"D:\BackUpDrive\Programacion\Python\TransFolk\data\essenABC",
        output_dir=r"D:\BackUpDrive\Programacion\Python\TransFolk\data\essen"
    )


    print("Archivos ABC procesados:", summary["abc_files_processed"])
    print("Tunes detectadas:", summary["tunes_detected"])
    print("Convertidas OK:", summary["converted_ok"])
    print("Fallidas:", summary["failed"])

    for err in summary["errors"][:10]:
        print(err)
