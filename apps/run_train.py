from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk import main
import sys


if __name__ == "__main__":
    ruta_base = sys.argv[1] if len(sys.argv) > 1 else None
    model_name = sys.argv[2] if len(sys.argv) > 2 else "todos"
    corpus_name = sys.argv[3] if len(sys.argv) > 3 else "todos"
    token_name = sys.argv[4] if len(sys.argv) > 4 else "momet"
    save_epochs = sys.argv[5] if len(sys.argv) > 5 else False
    registry = ConfigRegistry()
    registry.load_all()

    models = []
    if model_name == "todos":
        models=[ "kurt010", "mick010" ]
    else:
        models=[model_name]

    for modelname in models:
        model = registry.find_by_name(f"{modelname}_{corpus_name}_{token_name}_x_x")
        model = main.run_train(model, save_each_epoch=save_epochs, root_path=ruta_base)
        registry.update_model(model)



    # main.run_test_architecture(model)
    # arch = registry.find_by_name("kurt001")
    # corpus = registry.find_by_name("todos")
    # tk = registry.find_by_name("baseline")
    # mc = registry.find_by_name("major_2_4")
    # adt = registry.find_by_name("basic_set")
    # exp = registry.find_by_name("todos_baseline_major_2_4")
    # rt = registry.find_by_name("train_2")


    # for name in ["kurt", "mick", "robb", "stvy", "john"]:
    #     model = registry.find_by_name(f"{name}001_todos_momet_x_x")
    #     model = main.run_train(model)
    #     registry.update_model(model)



