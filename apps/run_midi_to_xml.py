from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_preprocesing.midi_to_musicxml import midi_folder_to_musicxml


if __name__ == "__main__":
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    registry = ConfigRegistry()
    registry.load_all()
    corpus = registry.find_by_name("altoaragon")
    adc=registry.find_by_name("basic_set")

    data_dir_raw = resolver.data_mid(corpus)
    data_dir_clean = resolver.data_raw(corpus)

    result = midi_folder_to_musicxml(
        input_folder=str(data_dir_raw),
        output_folder=str(data_dir_clean),
        allowed_durations=adc.durations,
        candidate_time_signatures=["2/4", "3/4", "4/4", "6/8"],
        recursive=True,
        verbose=True,
        corpus_name="altoaragon"
    )