from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk import main

if __name__ == "__main__":
    registry = ConfigRegistry()
    registry.load_all()

    #arch = registry.find_by_name("kurt001")
    #corpus = registry.find_by_name("todos")
    #tk = registry.find_by_name("baseline")
    #mc = registry.find_by_name("major_2_4")
    #adt = registry.find_by_name("basic_set")
    #exp = registry.find_by_name("todos_baseline_major_2_4")
    # rt = registry.find_by_name("train_2")
    # model = registry.find_by_name("john001_todos_momet_x_x")
    # model = main.run_train(model)
    # registry.update_model(model) #guardamos en BD datetimes y vocab_file


    for name in ["kurt", "mick", "robb", "stvy", "john"]:
        model = registry.find_by_name(f"{name}001_todos_momet_x_x")
        model = main.run_train(model)
        registry.update_model(model)



