from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk import main
from transfolk_preprocesing.count_TS_tonality import load_ts_mode_distribution
import sys
import numpy as np


if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
    corpus_name = sys.argv[2] if len(sys.argv) > 2 else "todos"
    tokenizer = sys.argv[3] if len(sys.argv) > 3 else "momet"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "mick001"

    settings = Settings(ruta_base)
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name(corpus_name)
    model = registry.find_by_name(f"{model_name}_{corpus_name}_{tokenizer}_x_x")
    rt = registry.find_by_name("generate_5")

    print(f"--> 📉 STYLE FIDELITY Curve: {corpus}, {tokenizer}, {model_name}")
    print(f"--> 📉 Generation of pieces...")

    # 1. Cargar el diccionario de ocurrencias TS/tonality previamente calculado para cada corpus
    dict_norm = load_ts_mode_distribution(f"{str(resolver.data_token(corpus))}/ts_mode_distribution_normalized.json")

    TEMPERATURES = np.arange(0.8, 2.2, 0.1)
    main.run_generate_for_curves_style(model, rt, TEMPERATURES, dict_norm, 100)


    #

    # TEMPERATURES = np.arange(0.8, 2.2, 0.1)
    # for TEMPERATURE in TEMPERATURES:
    #     rt.temperature = TEMPERATURE
    #     main.run_generate_from_TS_tonality(model, rt, "2/4", "major")
    #



