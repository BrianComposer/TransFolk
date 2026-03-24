import os
from apps.db.db_connection import get_connection

from transfolk_config import Corpus
from transfolk_config import TokenizerAlgorithm
from transfolk_config import MusicContext
from transfolk_config import AllowedDurations
from transfolk_config import Experiment
from transfolk_config import TransformerArchitecture
from transfolk_config import RuntimeTrain
from transfolk_config import RuntimeGenerate
from transfolk_config import Model



class ConfigRegistry(object):

    def __init__(self, db_path=None):

        if db_path is None:
            # Ruta absoluta al directorio apps/db
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construimos ruta absoluta a la BD
            db_path = os.path.join(current_dir, "transfolk_config.db")

        self.conn = get_connection(db_path)

        self.corpus = {}
        self.tokenizers = {}
        self.music_contexts = {}
        self.allowed_durations = {}
        self.experiments = {}
        self.architectures = {}
        self.runtime_trains = {}
        self.runtime_generates = {}
        self.models = {}

    def load_all(self):
        self._load_simple()
        self._load_experiments()
        self._load_models()

    def _load_simple(self):
        for row in self.conn.execute("SELECT * FROM corpus"):
            self.corpus[row["id"]] = Corpus(**row)

        for row in self.conn.execute("SELECT * FROM tokenizer_algorithm"):
            self.tokenizers[row["id"]] = TokenizerAlgorithm(**row)

        for row in self.conn.execute("SELECT * FROM music_context"):
            self.music_contexts[row["id"]] = MusicContext(**row)

        for row in self.conn.execute("SELECT * FROM allowed_durations"):
            self.allowed_durations[row["id"]] = AllowedDurations.from_db(row)

        for row in self.conn.execute("SELECT * FROM transformer_architecture"):
            self.architectures[row["id"]] = TransformerArchitecture(**row)

        for row in self.conn.execute("SELECT * FROM runtime_train"):
            self.runtime_trains[row["id"]] = RuntimeTrain(**row)

        for row in self.conn.execute("SELECT * FROM runtime_generate"):
            self.runtime_generates[row["id"]] = RuntimeGenerate(**row)

    def _load_experiments(self):
        for row in self.conn.execute("SELECT * FROM experiment"):
            self.experiments[row["id"]] = Experiment(
                id=row["id"],
                name=row["name"],
                descripcion=row["descripcion"],
                corpus=self.corpus[row["id_corpus"]],
                tokenizer=self.tokenizers[row["id_tk"]],
                music_context=self.music_contexts[row["id_mc"]],
                allowed_durations=self.allowed_durations[row["id_ad"]],
            )

    def _load_models(self):
        for row in self.conn.execute("SELECT * FROM model"):
            self.models[row["id"]] = Model(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                architecture=self.architectures[row["id_ta"]],
                experiment=self.experiments[row["id_exp"]],
                runtime_train=self.runtime_trains[row["id_rt"]],
                train_start_time=row["train_start_time"],
                train_end_time=row["train_end_time"],
                train_total_time=row["train_total_time"],
                train_date=row["train_date"],
                vocab_file=row["vocab_file"],
            )

    def find_by_name(self, name):
        all_objs = (
            list(self.corpus.values())
            + list(self.tokenizers.values())
            + list(self.music_contexts.values())
            + list(self.allowed_durations.values())
            + list(self.experiments.values())
            + list(self.architectures.values())
            + list(self.runtime_trains.values())
            + list(self.runtime_generates.values())
            + list(self.models.values())
        )
        return [o for o in all_objs if getattr(o, "name", None) == name][0]

    def update_model(self, model: Model):
        query = """
        UPDATE model
        SET
            name = ?,
            description = ?,
            train_start_time = ?,
            train_end_time = ?,
            train_total_time = ?,
            train_date = ?,
            vocab_file = ?
        WHERE id = ?
        """

        self.conn.execute(
            query,
            (
                model.name,
                model.description,
                model.train_start_time,
                model.train_end_time,
                model.train_total_time,
                model.train_date,
                model.vocab_file,
                model.id,
            ),
        )

        self.conn.commit()

def prueba1():
    registry = ConfigRegistry("transfolk_config.db")
    registry.load_all()

    print("Corpus loaded:", len(registry.corpus))
    print("Tokenizers loaded:", len(registry.tokenizers))
    print("Music Contexts loaded:", len(registry.music_contexts))
    print("Allowed durations:", len(registry.allowed_durations))
    print("Transformers architectures loaded:", len(registry.architectures))
    print("Experiments loaded:", len(registry.experiments))
    print("Train runtimes loaded:", len(registry.runtime_trains))
    print("Generate runtimes loaded:", len(registry.runtime_generates))
    print("Models loaded:", len(registry.models))

    # Ejemplo: buscar por nombre
    results = registry.find_by_name("kurt001_todos_baseline_major_2_4")
    print("Search results:", results)
    results = registry.find_by_name("todos_standard_major_2_4")
    print("Search results:", results)
    results = registry.find_by_name("todos")
    print("Search results:", results)
    results = registry.find_by_name("major_6_8")
    print("Search results:", results)


if __name__ == "__main__":
    try:
        prueba1()

    except Exception as e:
        print("Error during execution:")
        print(str(e))

