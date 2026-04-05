import os
from pathlib import Path


#os.environ["TRANSFOLK"] = r"D:\BackUpDrive\Programacion\Python\TransFolk"


class Settings:

    def __init__(self, root=None):
        self.root=""
        if root:
            if root == ".":
                self.root = Path(Path.cwd()).resolve() #Truco para que se pueda ejecutar en runpod con python -m apps.run_train "."
                print("RUNPOD SETTINGS ROOT =", self.root)
            else:
                self.root = Path(root).resolve()
        elif os.environ.get("TRANSFOLK"):
            self.root = Path(os.environ.get("TRANSFOLK")).resolve()
        else:
            self.root = Path(Path.cwd()).resolve().parent





# class Settings:
#
#     def __init__(self):
#         self.root = Path(
#             os.environ.get("TRANSFOLK", Path.cwd())
#         ).resolve()
#
