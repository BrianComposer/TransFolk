# from transfolk_config import DEFAULT_DOMAIN, Experiment, TransformerModel, RuntimeConfig, ProjectPaths, PathResolver, \
#     Settings, DEFAULT_TRANSFORMER_MODEL
#
# from transfolk.model.transformer import MusicTransformer
#
#
# from transfolk_metrics import tokenLevelEntropy
# from transfolk_metrics import conditionalEntropy
# from transfolk_metrics import modalStability
# from transfolk_metrics import patternRetentionRate
# from transfolk_classifier import corpus_membership_classifier
#
# import json
# import torch
# from torch.utils.data import DataLoader
#
#
#
# if __name__ == "__main__":
#
#     settings = Settings()
#     paths = ProjectPaths(settings.root)
#     resolver = PathResolver(paths)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     for corpus in ["todos", "valencia", "aragon"]:
#         for algorithm in ["baseline", "standard", "patterns"]:
#             for time_signature in ["2/4", "3/4", "6/8"]:
#                 for tonality in ["major", "minor"]:
#                     for temperature in [1.2]:
#
#                         exp = Experiment(
#                             corpus=corpus,
#                             algorithm=algorithm,
#                             time_signature=time_signature,
#                             tonality=tonality
#                         )
#
#                         runtime = RuntimeConfig( mode="metrics", temperature=temperature)
#
#
#                         model_path = resolver.model_file(exp, DEFAULT_TRANSFORMER_MODEL.id)
#                         data_dir_raw = resolver.data_raw(exp)
#                         data_dir_clean = resolver.data_clean(exp)
#                         data_raw = resolver.data_raw(exp)
#                         data_dir = resolver.data_clean(exp)
#                         prod_dir= resolver.production_dir(exp, runtime)
#                         train_dir = resolver.train_dir(exp)
#                         charts_dir = resolver.charts_dir()
#                         sequences_file = resolver.sequences_file(exp)
#                         vocab_file = resolver.vocab_file(exp)
#                         model_file = resolver.model_file(exp, DEFAULT_TRANSFORMER_MODEL.id)
#                         loss_log = resolver.loss_log_file(exp, DEFAULT_TRANSFORMER_MODEL.id)
#                         classifier_model_dir = resolver.classifier_dir(exp)
#                         sequences_file_prod = rf"{resolver.paths.root}\experiments\productions\{corpus}\{algorithm}\{time_signature[0]}_{time_signature[2]}\{tonality}\{temperature}\sequences_{corpus}_{tonality}_{time_signature[0]}_{time_signature[2]}.json"
#                         vocab_file_prod = rf"{resolver.paths.root}\experiments\productions\{corpus}\{algorithm}\{time_signature[0]}_{time_signature[2]}\{tonality}\{temperature}\vocab_{corpus}_{tonality}_{time_signature[0]}_{time_signature[2]}.json"
#
#
#                         # ********** NECESARIO TOKENIZADO DE LAS PRODUCCIONES LA PRIMERA VEZ TRAS GENERAR **************
#                         # print(f"🔤 TOKENIZATION OF PRODUCTIONS: {corpus}, {time_signature}, {tonality}, {temperature}")
#                         # proddirbuf = f"{prod_dir}{time_signature[0]}_{time_signature[2]}/{tonality}/{temperature}/"
#                         # sequences, vocab = process_musicxml_directory(proddirbuf, 100000, algorithm, time_signature, tonality, allowed_durations)
#                         # with open(SEQUENCES_FILE_PROD, "w") as f:
#                         #     json.dump(sequences, f)
#                         # with open(VOCAB_FILE_PROD, "w") as f:
#                         #     json.dump(vocab, f)
#                         # print("✅ Tokenización completada.")
#
#                         # Entropia por token
#                         with open(sequences_file_prod, "r") as f:
#                             sequences = json.load(f)
#                         with open(vocab_file_prod, "r") as f:
#                             vocab = json.load(f)
#                         token_entropy = str(tokenLevelEntropy.token_entropy(sequences, 2)).replace(".", ",")
#                         print(f"Produ. Token Level Entropy\t{corpus}\t{algorithm}\t{time_signature}\t{tonality}\t{temperature}\t{token_entropy}")
#
#                         # Entropia condicional
#                         model = MusicTransformer(vocab_size=len(vocab), num_layers=DEFAULT_TRANSFORMER_MODEL.n_layers).to(device)
#                         model.eval()
#                         mean_Hk, all_Hk = conditionalEntropy.compute_Hk_from_sequences(model, sequences, 8, device)
#
#                         print(f"H_k medio \t{corpus}\t{algorithm}\t{time_signature}\t{tonality}\t{temperature}\t", mean_Hk)
#
#                         # Estabilidad modal
#                         mean_stability, all_values = modalStability.modal_stability_from_folder(prod_dir, window_measures=2)
#
#                         print(f"Estabilidad modal Media \t{corpus}\t{algorithm}\t{time_signature}\t{tonality}\t{temperature}\t", mean_stability)
#
#                         # Classifier Based Style Probability
#                         S_folk, std = corpus_membership_classifier.evaluate_model(prod_dir, classifier_model_dir, False)
#                         print("S_folk =", S_folk)
#                         # print(values[:10])
#
#                         # Pattern Retention Rate
#                         with open(sequences_file_prod, "r") as f:
#                             generated_sequences = json.load(f)
#                         with open(sequences_file_prod, "r") as f:
#                             vocab = json.load(f)
#                         with open(sequences_file, "r") as f:
#                             corpus_sequences = json.load(f)
#
#                         R_rhythm = 0
#                         R_rhythm_i = 0
#                         try:
#
#                             R_rhythm, R_rhythm_i = patternRetentionRate.rhythmic_pattern_retention(
#                                 corpus_sequences,
#                                 generated_sequences,
#                                 vocab,
#                                 n_min=3,
#                                 n_max=6,
#                                 min_count=20
#                             )
#
#                         except Exception as e:
#                             print(f"Errores en el proceso: {e}")
#                             R_rhythm = 0
#                             R_rhythm_i = 0
#
#                         print("Pattern-Retention (Rítmico):", R_rhythm)
#                         print("**** METRICS ENDED ****")
#                         print(
#                             f"{corpus}\t{algorithm}\t{time_signature}\t{tonality}\t{temperature}\t{token_entropy}\t{mean_Hk}\t{mean_stability}\t{S_folk}\t{R_rhythm}".replace(
#                                 ".", ","))
