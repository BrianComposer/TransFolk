# from transfolk.pipeline import TransFolkPipeline
#
# MODEL_FILE = rf"{ROOT_DIR}\models\todos\standard\music_todos_transformer_major_2_4.pt"
# VOCAB_FILE = rf"{ROOT_DIR}\models\todos\standard\vocab_todos_major_2_4.json"
# NUM_LAYERS = 6
# FILE_XML_PROMPT_PATH = rf"{ROOT_DIR}\experiments\prompts\prompt1.xml"
# ALGORITHM = "standard"
# TIME_SIGNATURE = "2/4"
# TONALITY = "major"
#
# miTransformer = TransFolkPipeline(MODEL_FILE, VOCAB_FILE, NUM_LAYERS)
# for i in range(1):
#     score = miTransformer.generate_from_xml(FILE_XML_PROMPT_PATH, ALGORITHM, TONALITY, TIME_SIGNATURE, 1.2, 256)
#     score.show()