from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import shutil
import re
from transfolk_config import *
from apps.db.config_registry import ConfigRegistry

def _sanitize_folder_name(name: str) -> str:
    """
    Limpia un nombre para usarlo como carpeta.
    """
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def _collect_musicxml_files(folder: Path, genre: str) -> List[Dict[str, str]]:
    """
    Recoge ficheros MusicXML de una carpeta y los etiqueta con su género.

    Se aceptan extensiones habituales:
    - .musicxml
    - .xml
    - .mxl
    """
    allowed_suffixes = {".musicxml", ".xml", ".mxl"}

    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in allowed_suffixes:
            files.append({
                "path": str(f.resolve()),
                "original_name": f.name,
                "genre": genre
            })
    return files


def repartir_piezas_folk(
    ruta_profano: str,
    ruta_religioso: str,
    personas: List[str],
    n: int,
    carpeta_salida: str,
    semilla: Optional[int] = None
) -> None:
    """
    Selecciona aleatoriamente n piezas por persona a partir de dos carpetas
    (profano y religioso), sin repetir piezas entre personas.

    Parámetros
    ----------
    ruta_profano : str
        Ruta a la carpeta con piezas del género profano.
    ruta_religioso : str
        Ruta a la carpeta con piezas del género religioso.
    personas : List[str]
        Lista de nombres de personas.
    n : int
        Número de piezas que recibirá cada persona.
    carpeta_salida : str
        Carpeta raíz donde se crearán las subcarpetas de cada persona.
    semilla : Optional[int]
        Semilla para reproducibilidad. Si es None, la selección será aleatoria.

    Efectos
    -------
    - Crea una carpeta por persona dentro de `carpeta_salida`.
    - Copia allí n piezas seleccionadas aleatoriamente.
    - Las renombra como 1, 2, ..., n conservando extensión.
    - Genera un fichero `clave_respuestas.txt` con la correspondencia:
      persona -> número -> nombre original -> género.
    """

    if n <= 0:
        raise ValueError("El parámetro n debe ser un entero positivo.")

    if not personas:
        raise ValueError("La lista de personas no puede estar vacía.")

    carpeta_profano = Path(ruta_profano)
    carpeta_religioso = Path(ruta_religioso)
    salida = Path(carpeta_salida)

    if not carpeta_profano.exists() or not carpeta_profano.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de profano: {carpeta_profano}")

    if not carpeta_religioso.exists() or not carpeta_religioso.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de religioso: {carpeta_religioso}")

    # Inicializar generador aleatorio
    rng = random.Random(semilla)

    # Recoger piezas
    piezas_profano = _collect_musicxml_files(carpeta_profano, "profano")
    piezas_religioso = _collect_musicxml_files(carpeta_religioso, "religioso")

    todas_las_piezas = piezas_profano + piezas_religioso

    if not todas_las_piezas:
        raise ValueError("No se han encontrado archivos MusicXML en las carpetas indicadas.")

    total_necesario = len(personas) * n
    total_disponible = len(todas_las_piezas)

    if total_disponible < total_necesario:
        raise ValueError(
            f"No hay suficientes piezas para repartir sin repetición.\n"
            f"Piezas disponibles: {total_disponible}\n"
            f"Piezas necesarias: {total_necesario} "
            f"({len(personas)} personas x {n} piezas)"
        )

    # Mezclar aleatoriamente todas las piezas y repartir bloques consecutivos
    rng.shuffle(todas_las_piezas)
    piezas_seleccionadas = todas_las_piezas[:total_necesario]

    # Crear carpeta de salida
    salida.mkdir(parents=True, exist_ok=True)

    # Estructura para la clave de respuestas
    clave: Dict[str, List[Tuple[int, str, str]]] = {}

    # Reparto sin repetición
    idx = 0
    for persona in personas:
        persona_folder = salida / _sanitize_folder_name(persona)
        persona_folder.mkdir(parents=True, exist_ok=True)

        clave[persona] = []

        lote = piezas_seleccionadas[idx: idx + n]
        idx += n

        for numero, pieza in enumerate(lote, start=1):
            src = Path(pieza["path"])
            ext = src.suffix.lower()
            nuevo_nombre = f"{numero}{ext}"
            dst = persona_folder / nuevo_nombre

            shutil.copy2(src, dst)

            clave[persona].append(
                (numero, pieza["original_name"], pieza["genre"])
            )

    # Generar clave de respuestas
    clave_path = salida / "clave_respuestas.txt"
    with clave_path.open("w", encoding="utf-8") as f:
        f.write("CLAVE DE RESPUESTAS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Carpeta profano   : {carpeta_profano.resolve()}\n")
        f.write(f"Carpeta religioso : {carpeta_religioso.resolve()}\n")
        f.write(f"Número de personas: {len(personas)}\n")
        f.write(f"Piezas por persona: {n}\n")
        f.write(f"Total piezas usadas: {total_necesario}\n\n")

        for persona in personas:
            f.write(f"PERSONA: {persona}\n")
            f.write("-" * 80 + "\n")
            for numero, original_name, genre in clave[persona]:
                f.write(
                    f"{numero}\t{original_name}\t{genre}\n"
                )
            f.write("\n")

    print("Proceso completado correctamente.")
    print(f"Carpeta de salida: {salida.resolve()}")
    print(f"Clave de respuestas: {clave_path.resolve()}")



if __name__ == "__main__":

    #REPARTO ALEATORIO SIN FORZAR 5 y 5
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
    DATA_DIR_RELIGIOSO_CLEAN = CORPUS_B_DIR
    OUTPUT_DIR = str(resolver.charts_dir() / rf"apps_teimus/{corpus.name}/reparto")

    ruta_profano = str(CORPUS_A_DIR)
    ruta_religioso = str(CORPUS_B_DIR)
    personas = [
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
    n = 10

    semilla = 1234  # opcional

    repartir_piezas_folk(
        ruta_profano=ruta_profano,
        ruta_religioso=ruta_religioso,
        personas=personas,
        n=n,
        carpeta_salida=OUTPUT_DIR,
        semilla=semilla
    )
