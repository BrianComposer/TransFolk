from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# Asumiendo que este paquete se colocará en apps/db_admin
DEFAULT_DB_PATH = (APP_DIR.parent / "db" / "transfolk_config.db").resolve()

TABLE_CONFIG = {
    "corpus": {
        "title": "Corpus",
        "columns": ["id", "name", "subcorpus", "descripcion"],
        "pk": "id",
        "display_column": "name",
    },
    "tokenizer_algorithm": {
        "title": "Tokenizer Algorithm",
        "columns": ["id", "name", "description"],
        "pk": "id",
        "display_column": "name",
    },
    "allowed_durations": {
        "title": "Allowed Durations",
        "columns": ["id", "name", "durations", "description"],
        "pk": "id",
        "display_column": "name",
    },
    "music_context": {
        "title": "Music Context",
        "columns": ["id", "name", "tonality", "time_signature"],
        "pk": "id",
        "display_column": "name",
    },
    "experiment": {
        "title": "Experiment",
        "columns": [
            "id",
            "id_corpus",
            "id_tk",
            "id_mc",
            "id_ad",
            "name",
            "descripcion",
        ],
        "pk": "id",
        "display_column": "name",
        "foreign_keys": {
            "id_corpus": {"table": "corpus", "display": "name"},
            "id_tk": {"table": "tokenizer_algorithm", "display": "name"},
            "id_mc": {"table": "music_context", "display": "name"},
            "id_ad": {"table": "allowed_durations", "display": "name"},
        },
    },
    "runtime_train": {
        "title": "Runtime Train",
        "columns": [
            "id", "name", "epochs", "batch_size", "learning_rate",
            "weight_decay", "gradient_clip", "scheduler", "warmup_steps",
            "accumulation_steps", "early_stopping", "patience", "save_every",
            "optimizer", "loss"
        ],
        "pk": "id",
        "display_column": "name",
    },
    "transformer_architecture": {
        "title": "Transformer Architecture",
        "columns": [
            "id", "name", "description", "type", "d_model", "n_heads", "n_layers", "d_ff",
            "dropout", "max_seq_len", "attention_type", "activation",
            "positional_encoding", "layer_norm_eps", "bias", "weight_tying",
            "embedding_dropout", "residual_dropout", "attention_dropout",
            "initializer", "rotary_dim", "encoder_layers", "decoder_layers"
        ],
        "pk": "id",
        "display_column": "name",
    },
    "runtime_generate": {
        "title": "Runtime Generate",
        "columns": [
            "id", "name", "temperature", "max_len", "num_productions", "top_k",
            "top_p", "repetition_penalty", "greedy", "seed", "device",
            "mixed_precision", "num_workers", "deterministic"
        ],
        "pk": "id",
        "display_column": "name",
    },
    "model": {
        "title": "Model",
        "columns": [
            "id", "name", "id_ta", "id_exp", "id_rt", "description",
            "vocab_file", "train_start_time", "train_end_time",
            "train_total_time", "train_date"
        ],
        "pk": "id",
        "display_column": "name",
        "foreign_keys": {
            "id_ta": {"table": "transformer_architecture", "display": "name"},
            "id_exp": {"table": "experiment", "display": "name"},
            "id_rt": {"table": "runtime_train", "display": "name"},
        },
    },
    "snapshot_generation": {
        "title": "Snapshot Generation",
        "columns": ["id", "id_rg", "id_model", "snapshot_datetime"],
        "pk": "id",
        "display_column": "snapshot_datetime",
        "foreign_keys": {
            "id_rg": {"table": "runtime_generate", "display": "name"},
            "id_model": {"table": "model", "display": "name"},
        },
    },
}
