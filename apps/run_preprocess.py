from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_preprocesing.dataCleaning import normalize_musicxml_corpus_new


if __name__ == "__main__":
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    registry = ConfigRegistry()
    registry.load_all()
    corpus = registry.find_by_name("altoaragon")
    adc=registry.find_by_name("corpus_cleaning")

    data_dir_raw = resolver.data_raw(corpus)
    data_dir_clean = resolver.data_clean(corpus)

    normalize_musicxml_corpus_new(data_dir_raw,
                                  data_dir_clean,
                                  adc.durations,
                                  midi_min=30,
                                  midi_max=110,
                                  overwrite=True,
                                  delete_grace_notes=False,
                                  create_title=True,
                                  respect_time_signature_changes=True,
                                  respect_ties=True)