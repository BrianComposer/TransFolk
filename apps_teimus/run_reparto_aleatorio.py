#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

from transfolk_config import *
from apps.db.config_registry import ConfigRegistry

# =========================
# FUNCIONES
# =========================
def to_path(p) -> Path:
    """Convierte str o Path a Path."""
    return p if isinstance(p, Path) else Path(p)


def list_xml_files(folder) -> List[Path]:
    folder = to_path(folder)
    if not folder.exists():
        raise FileNotFoundError("No existe el directorio: {}".format(folder))
    if not folder.is_dir():
        raise NotADirectoryError("No es un directorio: {}".format(folder))
    return sorted([p for p in folder.glob("*.xml") if p.is_file()])


def safe_person_dirname(name: str) -> str:
    """Normaliza un nombre para usarlo como carpeta."""
    s = re.sub(r"[^\w\-\.]+", "_", name.strip(), flags=re.UNICODE)
    return s or "persona"


def choose_without_replacement(pool: List[Path], k: int, rng: random.Random) -> List[Path]:
    """Elige k elementos del pool sin repetición y los elimina del pool."""
    if len(pool) < k:
        raise ValueError("No hay suficientes archivos para elegir {}. Disponibles: {}".format(k, len(pool)))
    chosen = rng.sample(pool, k)
    chosen_set = set(chosen)
    pool[:] = [x for x in pool if x not in chosen_set]
    return chosen


def generate_ids_for_person(rng: random.Random, total_works: int) -> List[int]:
    """
    Genera IDs únicos por persona en el rango:
      1 .. (2*NUM_PER_PERSON_PER_FOLDER) inclusive
    En la práctica: una permutación aleatoria de 1..total_works.
    """
    # total_works es exactamente 2*NUM_PER_PERSON_PER_FOLDER (10 si 5+5)
    return rng.sample(list(range(1, total_works + 1)), total_works)


def copy_and_rename_for_person(
    person_folder: Path,
    tagged_files: List[Tuple[str, Path]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Copia los XML seleccionados a la carpeta de la persona y los renombra
    a enteros aleatorios en 1..N (N = número total de obras por persona).
    Devuelve entradas para la clave:
      {id_generado, carpeta, archivo_original, archivo_nuevo}
    """
    person_folder.mkdir(parents=True, exist_ok=True)

    total = len(tagged_files)  # 10 si 5 profano + 5 religioso
    ids = generate_ids_for_person(rng, total)

    respuesta = []
    for (kind, src), new_id in zip(tagged_files, ids):
        new_name = "{}.xml".format(new_id)
        dst = person_folder / new_name

        # Si ya existe (por ejecuciones anteriores), lo sobreescribimos
        if dst.exists():
            dst.unlink()

        shutil.copy2(src, dst)

        respuesta.append({
            "id_generado": new_id,
            "carpeta": kind,               # profano / religioso
            "archivo_original": src.name,
            "archivo_nuevo": new_name,
        })

    # Orden opcional para que la clave sea fácil de leer (1..N)
    respuesta.sort(key=lambda x: x["id_generado"])
    return respuesta


# =========================
# MAIN
# =========================

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name("arnaudasteruel")
    corpusA = Corpus(name=corpus.name, subcorpus="profano", id=None)
    corpusB = Corpus(name=corpus.name, subcorpus="religioso", id=None)

    CORPUS_A_DIR = resolver.data_clean(corpusA, corpusA.subcorpus)
    CORPUS_B_DIR = resolver.data_clean(corpusB, corpusB.subcorpus)

    data_dir_profano = to_path(DATA_DIR_PROFANO_CLEAN)
    data_dir_religioso = to_path(DATA_DIR_RELIGIOSO_CLEAN)
    output_dir = to_path(OUTPUT_DIR)

    profanos = list_xml_files(data_dir_profano)
    religiosos = list_xml_files(data_dir_religioso)

    needed = len(PERSONAS) * NUM_PER_PERSON_PER_FOLDER
    if len(profanos) < needed:
        raise SystemExit(
            "ERROR: No hay suficientes profanos. Necesarios {}, disponibles {}".format(needed, len(profanos))
        )
    if len(religiosos) < needed:
        raise SystemExit(
            "ERROR: No hay suficientes religiosos. Necesarios {}, disponibles {}".format(needed, len(religiosos))
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # clave[persona] = [ {id_generado, carpeta, archivo_original, archivo_nuevo}, ... ]
    clave = {}  # type: Dict[str, List[Dict[str, Any]]]

    for person in PERSONAS:
        person_dir = output_dir / safe_person_dirname(person)

        chosen_prof = choose_without_replacement(profanos, NUM_PER_PERSON_PER_FOLDER, rng)
        chosen_rel = choose_without_replacement(religiosos, NUM_PER_PERSON_PER_FOLDER, rng)

        tagged = [("profano", p) for p in chosen_prof] + [("religioso", r) for r in chosen_rel]

        # Copia + renombra (IDs: 1..10 si hay 10 obras)
        clave[person] = copy_and_rename_for_person(person_dir, tagged, rng)

    json_path = output_dir / "clave_respuesta.json"
    txt_path = output_dir / "clave_respuesta.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(clave, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        for person, items in clave.items():
            f.write("{}\n".format(person))
            for it in items:
                # Formato: "<id_generado> <carpeta> <archivo_original>"
                f.write("  - {} {} {}\n".format(it["id_generado"], it["carpeta"], it["archivo_original"]))
            f.write("\n")

    print("OK. Salida en: {}".format(output_dir.resolve()))
    print("Clave JSON: {}".format(json_path.resolve()))
    print("Clave TXT : {}".format(txt_path.resolve()))


# -----------------------------
# RUN WITH "PLAY"
# -----------------------------
if __name__ == "__main__":
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name("arnaudasteruel")
    corpusA = Corpus(name=corpus.name, subcorpus="profano", id=None)
    corpusB = Corpus(name=corpus.name, subcorpus="religioso", id=None)

    CORPUS_A_DIR = resolver.data_clean(corpusA, corpusA.subcorpus)
    CORPUS_B_DIR = resolver.data_clean(corpusB, corpusB.subcorpus)

    DATA_DIR_PROFANO_CLEAN = CORPUS_A_DIR
    DATA_DIR_RELIGIOSO_CLEAN = CORPUS_A_DIR
    OUTPUT_DIR = resolver.charts_dir() / rf"apps_teimus/{corpus.name}/reparto"

    RANDOM_SEED = None  # ej: 1234 # Semilla opcional (para reproducibilidad). Pon None para aleatorio real.
    NUM_PER_PERSON_PER_FOLDER = 5  # 5 profano + 5 religioso = 10 total
    # Lista de personas
    PERSONAS = [
        "Paco",
        "Manuel",
        "Victor",
        "Rafa",
        "Brian",
        "JuanGCastelao",
        "Kike",
        "Cristian",
        "Eulalia",
        "Alejandro"
    ]
    main()
