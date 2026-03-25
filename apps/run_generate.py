from transfolk_config import *
from apps.db.config_registry import ConfigRegistry
from transfolk import main

if __name__ == "__main__":
    registry = ConfigRegistry()
    registry.load_all()

    # arch = registry.find_by_name("kurt001")
    # corpus = registry.find_by_name("todos")
    # tk = registry.find_by_name("baseline")
    # mc = registry.find_by_name("major_2_4")
    # adt = registry.find_by_name("basic_set")
    # exp = registry.find_by_name("todos_baseline_major_2_4")
    rt = registry.find_by_name("generate_5")
    model = registry.find_by_name("kurt001_todos_momet_x_x")

    main.run_generate(model, rt)
    # main.run_generate_from_musicxml_prompt(model, rt,
    # r"G:\Mi unidad\Programacion\Python\TransFolk\experiments\prompts\prompt1.xml")
    #main.run_generate_from_TS_tonality(model, rt, "2/4", "major")
