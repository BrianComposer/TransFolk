import json

from transfolk_config import *
from apps.db.config_registry import ConfigRegistry

from transfolk_tokenization.tokenizer import *
from transfolk_patterns.rhythmicPatternSearcher import *


if __name__ == "__main__":

    registry = ConfigRegistry()
    registry.load_all()

    arch = registry.find_by_name("kurt001")
    corpus = registry.find_by_name("todos")
    # tk = registry.find_by_name("chm")
    # mc = registry.find_by_name("major_x")
    # adt = registry.find_by_name("corpus_cleaning")
    #exp = Experiment(id=1, name = "prueba", corpus=corpus, tokenizer=tk, music_context=mc, allowed_durations=adt)
    exp = registry.find_by_name("todos_momet_x_x")

    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    data_dir_raw = resolver.data_raw(corpus)
    data_dir_clean = resolver.data_clean(corpus)


    print(f"🔤 TOKENIZATION MODE: {corpus.name}, {exp.music_context.time_signature}, {exp.music_context.tonality}")
    sequences, vocab, errors = process_musicxml_directory(data_dir_clean,
                                                  100000000,
                                                  exp.tokenizer.name,
                                                  exp.music_context.time_signature,
                                                  exp.music_context.tonality,
                                                  exp.allowed_durations.durations)
    # Crear directorios si no existen
    os.makedirs(os.path.dirname(resolver.sequences_file(arch, exp)), exist_ok=True)
    os.makedirs(os.path.dirname(resolver.vocab_file(arch, exp)), exist_ok=True)
    os.makedirs(os.path.dirname(resolver.token_errors_file(arch, exp)), exist_ok=True)

    with open(resolver.sequences_file(arch, exp), "w") as f:
        json.dump(sequences, f)
    with open(resolver.vocab_file(arch, exp), "w") as f:
        json.dump(vocab, f)
    with open(resolver.token_errors_file(arch, exp), "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=4, ensure_ascii=False)

    print("✅ Tokenización completada.")

    #Inluimos la deteccion de patrones tras la tokenizacion en caso de algoritmo "patterns"
    if exp.tokenizer.name== "patterns":
        print(f"🎼 PATTERN DETECTION MODE: {exp.music_context.time_signature}, {exp.music_context.tonality}")
        with open(resolver.sequences_file(arch, exp), "r") as f:
            sequences = json.load(f)
        with open(resolver.vocab_file(arch, exp), "r") as f:
            vocab = json.load(f)

        print("🎶 Detección de patrones rítmicos")
        sequences_new, vocab_new = searchRhythmicPatterns(sequences, vocab, 3, 4, 200)
        # print("🎼 Detección de patrones melódicos")
        # sequences_new2, vocab_new2 = searchMelodicPatterns(sequences_new, vocab_new,3,6, 50)

        with open(resolver.sequences_file(arch, exp), "w") as f:
            json.dump(sequences_new, f)
        with open(resolver.vocab_file(arch, exp), "w") as f:
            json.dump(vocab_new, f)
        print("✅ Detección de patrones finalizada.")
