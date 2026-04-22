from platform import architecture

from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk_charts import pca, densityHeatmap, training_curves, histogramsMultimetric, membership
import sys


if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
    corpus_name = sys.argv[2] if len(sys.argv) > 2 else "todos"
    tokenizer = sys.argv[3] if len(sys.argv) > 3 else "momet"
    # model_name = sys.argv[4] if len(sys.argv) > 4 else "mick001"
    # num_pieces = int(sys.argv[5] if len(sys.argv) > 5 else 100)

    settings = Settings(ruta_base)
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    registry = ConfigRegistry()
    registry.load_all()

    corpus = registry.find_by_name(corpus_name)
    tk = registry.find_by_name(tokenizer)
    # model = registry.find_by_name(f"{model_name}_{corpus_name}_{tokenizer}_x_x")

    print(f"--> 📈 TRAINING CURVES: {corpus.name}, {tk.name}")
    training_curves.plot_training_loss_all_paper(resolver.charts_dir(),
                                           str(paths.models_training),
                                        ["john010", "mick010", "robb010"],
                                           corpus.name,
                                           tk.name,
                                           font_size=24,
                                           axis_size=16,
                                           show_tittle=False,
                                           show_chart=True)

    for name in ["john010", "mick010", "robb010"]:

        print(f"--> 🔸 PCA of Corpus vs. Generated: {corpus}, {algorithm}, {time_signature}, {tonality}, {temperature}")
        pca.visualize_pca_numpy(charts_dir, data_dir_clean, prod_dir, corpus, algorithm, time_signature, tonality, temperature, font_size=18, axis_size=16, show_tittle=True, show_chart=True, by_temperature=False)

        print(f"--> 📊 Comparative Histograms: {corpus}, {algorithm}, {time_signature}, {tonality}, {temperature}")
        histogramsMultimetric.comparative_histograms_multimetric(charts_dir, data_dir_clean, prod_dir, 20, corpus, algorithm, time_signature, tonality, temperature, font_size=18, axis_size=16, legend_size=12, show_tittle=True, show_chart=True)

        print(f"--> 🔥 Kernel-Density heatmap: {corpus}, {algorithm}, {time_signature}, {tonality}, {temperature}")
        densityHeatmap.kernel_density_heatmap(charts_dir, data_dir_clean, prod_dir, 100, corpus, algorithm, time_signature, tonality, temperature, font_size=18, axis_size=16, show_tittle=True, show_chart=True)



#
# from transfolk_config.entities import (
#     Corpus,
#     TokenizerAlgorithm,
#     MusicContext,
#     AllowedDurations,
#     Experiment,
#     TransformerArchitecture,
#     RuntimeTrain,
#     Model, RuntimeGenerate
# )
# from transfolk_config.entities.resolver2 import PathResolver2
#
# from transfolk_config import ProjectPaths, Settings
# from transfolk_charts import pca, densityHeatmap, training_curves, histogramsMultimetric, membership
#
#
#
#
#
# if __name__ == "__main__":
#
#     settings = Settings()
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver2(paths)
#
#     for corpus in ["todos", "valencia", "aragon"]:
#         for algorithm in ["baseline", "standard", "patterns"]:
#             for time_signature in ["2/4", "3/4", "6/8"]:
#                 for tonality in ["major", "minor"]:
#                     for temperature in [1.2]:
#
#                         corpus = Corpus.get_or_create(name="todos")
#                         tk = TokenizerAlgorithm.get_or_create(name=algorithm)
#                         mc = MusicContext.get_or_create(name=f"{tonality}_{time_signature[0]}_{time_signature[2]}",
#                                                         tonality=tonality, time_signature=time_signature)
#                         adt = AllowedDurations.get_or_create(name="tokenization")
#                         exp = Experiment.get_or_create(corpus=corpus, tokenizer=tk, musicContext=mc, allowedWords=adt)
#                         rt = RuntimeGenerate.get_by_name(name="generate_5")
#                         m = Model.get_by_name("kurt001_todos_standard_major_2_4")
#
#                         model_path = resolver.model_file(exp, m)
#                         data_dir_raw = resolver.data_raw(exp)
#                         data_dir_clean = resolver.data_clean(exp)
#                         data_raw = resolver.data_raw(exp)
#                         data_dir = resolver.data_clean(exp)
#                         prod_dir = resolver.production_dir(exp, rt)
#                         train_dir = resolver.train_dir(exp)
#                         charts_dir = resolver.charts_dir()
#                         sequences_file = resolver.sequences_file(exp)
#                         vocab_file = resolver.vocab_file(exp)
#                         model_file = resolver.model_file(exp, m)
#                         loss_log = resolver.loss_log_file(exp, m)
#                         classifier_model_dir = resolver.classifier_dir(exp)
#                         # SEQUENCES_FILE_PROD = rf"{resolver.paths.root}/experiments/productions/{CORPUS}/{ALGORITHM}/{TIME_SIGNATURE[0]}_{TIME_SIGNATURE[2]}/{TONALITY}/{TEMPERATURE}/sequences_{CORPUS}_{TONALITY}_{TIME_SIGNATURE[0]}_{TIME_SIGNATURE[2]}.json"
#                         # VOCAB_FILE_PROD = rf"{resolver.paths.root}/experiments/productions/{CORPUS}/{ALGORITHM}/{TIME_SIGNATURE[0]}_{TIME_SIGNATURE[2]}/{TONALITY}/{TEMPERATURE}/vocab_{CORPUS}_{TONALITY}_{TIME_SIGNATURE[0]}_{TIME_SIGNATURE[2]}.json"
#
#                         print(f"📈 CHART GENERATION: {corpus}, {algorithm}, {time_signature}, {tonality}")
#
#                         print(f"--> 📈 TRAINING CURVES: {corpus}, {algorithm}")
#                         training_curves.plot_training_loss_all(charts_dir, train_dir, corpus, algorithm, font_size=18, axis_size=16, show_tittle=False, show_chart=True)
#
#
#                         print(f"--> 🔸 PCA of Corpus vs. Generated: {corpus}, {algorithm}, {time_signature}, {tonality}, {temperature}")
#                         pca.visualize_pca_numpy(charts_dir, data_dir_clean, prod_dir, corpus, algorithm, time_signature, tonality, temperature, font_size=18, axis_size=16, show_tittle=True, show_chart=True, by_temperature=False)
#
#                         print(f"--> 📊 Comparative Histograms: {corpus}, {algorithm}, {time_signature}, {tonality}, {temperature}")
#                         histogramsMultimetric.comparative_histograms_multimetric(charts_dir, data_dir_clean, prod_dir, 20, corpus, algorithm, time_signature, tonality, temperature, font_size=18, axis_size=16, legend_size=12, show_tittle=True, show_chart=True)
#
#                         print(f"--> 🔥 Kernel-Density heatmap: {corpus}, {algorithm}, {time_signature}, {tonality}, {temperature}")
#                         densityHeatmap.kernel_density_heatmap(charts_dir, data_dir_clean, prod_dir, 100, corpus, algorithm, time_signature, tonality, temperature, font_size=18, axis_size=16, show_tittle=True, show_chart=True)
#
#
#
#
#     # for CORPUS in ["todos", "valencia", "aragon"]:
#     #     for ALGORITHM in ["baseline", "standard", "patterns"]:
#     #         for TIME_SIGNATURE in ["2/4", "3/4", "6/8"]:
#     #             for TONALITY in ["major", "minor"]:
#     #                 print(f"--> 📉 STYLE FIDELITY Curve: {CORPUS}, {ALGORITHM}, {TIME_SIGNATURE}, {TONALITY}")
#     #                 TEMPERATURES = np.arange(0.8, 2.2, 0.1)
#     #                 for TEMPERATURE in TEMPERATURES:
#     #
#     #                     exp = experiment.ExperimentID(
#     #                         corpus=CORPUS,
#     #                         algorithm=ALGORITHM,
#     #                         time_signature=TIME_SIGNATURE,
#     #                         tonality=TONALITY,
#     #                         temperature=TEMPERATURE
#     #                     )
#     #
#     #                     model_path = resolver.model_file(exp)
#     #                     DATA_DIR_RAW = resolver.data_raw(exp)
#     #                     DATA_DIR_CLEAN = resolver.data_clean(exp)
#     #                     DATA_RAW = resolver.data_raw(exp)
#     #                     DATA_DIR = resolver.data_clean(exp)
#     #                     PROD_DIR = resolver.production_dir(exp)
#     #                     TRAIN_DIR = resolver.train_dir(exp)
#     #                     CHARTS_DIR = resolver.charts_dir()
#     #                     SEQUENCES_FILE = resolver.sequences_file(exp)
#     #                     VOCAB_FILE = resolver.vocab_file(exp)
#     #                     MODEL_FILE = resolver.model_file(exp)
#     #                     LOSS_LOG = resolver.loss_log(exp)
#     #                     CLASSIFIER_MODEL_DIR = resolver.classifier_dir(exp)
#     #
#     #                     membership.Style_Fidelity_Curve(DATA_DIR, PROD_DIR, CORPUS, ALGORITHM, TIME_SIGNATURE, TONALITY, TEMPERATURES, font_size=18, axis_size=16, show_tittle=False, show_chart=False)
#     #                     membership.Membership_Entropy_Scatter_Plot(CORPUS, TIME_SIGNATURE, TONALITY, PROD_DIR, TEMPERATURES, False)
#
