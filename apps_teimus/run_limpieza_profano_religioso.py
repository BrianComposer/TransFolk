from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_preprocesing.dataCleaning import normalize_musicxml_corpus_new


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

    CORPUS_P_CLEAN = resolver.data_clean(corpusA, corpusA.subcorpus)
    CORPUS_R_CLEAN = resolver.data_clean(corpusB, corpusB.subcorpus)

    adc = registry.find_by_name("corpus_cleaning")

    normalize_musicxml_corpus_new(CORPUS_P,
                                  CORPUS_P_CLEAN,
                                  adc.durations,
                                  midi_min=30,
                                  midi_max=110,
                                  overwrite=True,
                                  delete_grace_notes=False,
                                  create_title=True,
                                  respect_time_signature_changes=True,
                                  respect_ties=True)

    normalize_musicxml_corpus_new(CORPUS_R,
                                  CORPUS_R_CLEAN,
                                  adc.durations,
                                  midi_min=30,
                                  midi_max=110,
                                  overwrite=True,
                                  delete_grace_notes=False,
                                  create_title=True,
                                  respect_time_signature_changes=True,
                                  respect_ties=True)


