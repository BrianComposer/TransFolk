from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_classifier.build_corpus_split import build_split_metadata

from pathlib import Path

if __name__ == "__main__":
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name("ataa")
    corpusA = Corpus(name=corpus.name, subcorpus="profano", id=None)
    corpusB = Corpus(name=corpus.name, subcorpus="religioso", id=None)

    CORPUS_P = resolver.data_raw(corpusA, corpusA.subcorpus)
    CORPUS_R = resolver.data_raw(corpusB, corpusB.subcorpus)

    CORPUS_P_CLEAN = Path(resolver.data_clean(corpusA, corpusA.subcorpus))
    CORPUS_R_CLEAN = Path(resolver.data_clean(corpusB, corpusB.subcorpus))

    OUTPUT_DIR = resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}"

    # Crear directorios si no existen
    CORPUS_P_CLEAN.mkdir(parents=True, exist_ok=True)
    CORPUS_R_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Etiquetamos todas las obras y dividimos el CORPUS TRAIN/EVAL
    df_split = build_split_metadata(
        str(CORPUS_P_CLEAN),
        str(CORPUS_R_CLEAN),
        test_size=0.20,
        random_state=44
    )

    output_file = OUTPUT_DIR / "corpus_split.csv"
    df_split.to_csv(output_file, index=False, encoding="utf-8")



# if __name__ == "__main__":
#     settings = Settings()
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#
#     registry = ConfigRegistry()
#     registry.load_all()
#
#     corpus = registry.find_by_name("ataa")
#     corpusA = Corpus(name=corpus.name, subcorpus="profano", id=None)
#     corpusB = Corpus(name=corpus.name, subcorpus="religioso", id=None)
#
#     CORPUS_P = resolver.data_raw(corpusA, corpusA.subcorpus)
#     CORPUS_R = resolver.data_raw(corpusB, corpusB.subcorpus)
#
#     CORPUS_P_CLEAN = str(resolver.data_clean(corpusA, corpusA.subcorpus))
#     CORPUS_R_CLEAN = str(resolver.data_clean(corpusB, corpusB.subcorpus))
#     OUTPUT_DIR = str(resolver.paths.experiments / "teimus" / "classifiers" / f"{corpus.name}" )
#
#     #etiquetamos todas las obras y dividimos el CORPUS TRAIN/EVAL
#     df_split = build_split_metadata(CORPUS_P_CLEAN, CORPUS_R_CLEAN, test_size=0.20, random_state=44)
#     df_split.to_csv(f"{OUTPUT_DIR}\corpus_split.csv", index=False, encoding="utf-8")




