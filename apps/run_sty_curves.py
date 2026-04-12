from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk import main
from transfolk_config.entities.corpus import Corpus
from transfolk_preprocesing.count_TS_tonality import load_ts_mode_distribution
from transfolk_charts import pca, densityHeatmap, training_curves, histogramsMultimetric, membership

import sys
import numpy as np

if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
    corpus_name = sys.argv[2] if len(sys.argv) > 2 else "todos"
    tokenizer = sys.argv[3] if len(sys.argv) > 3 else "momet"
    model_name = sys.argv[4] if len(sys.argv) > 4 else "mick001"
    num_pieces = sys.argv[5] if len(sys.argv) > 5 else 100

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
    dict_norm = load_ts_mode_distribution(fr"{str(resolver.data_token(corpus))}/ts_mode_distribution_normalized.json")

    # 2. Generamos las piezas para cada temperatura
    TEMPERATURES = np.arange(0.8, 2.2, 0.1)
    #main.run_generate_for_curves_style(model, rt, TEMPERATURES, dict_norm, num_pieces, ruta_base)

    # 3. Generamos la gráfica
    model_dir = resolver.paths.models_classifier  # carpeta para guardar o cargar el modelo
    prod_dir = resolver.production_dir(model, rt).parent.parent.parent

    membership.Membership_Entropy_Scatter_Plot(MODEL_DIR=str(model_dir),
                                               TIME_SIGNATURE="x",
                                               TONALITY="x",
                                               PROD_DIR=prod_dir,
                                               TEMPERATURES=TEMPERATURES,
                                               show=False)
