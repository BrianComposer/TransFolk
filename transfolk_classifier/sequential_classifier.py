# -*- coding: utf-8 -*-
"""
sequential_classifier.py

Clasificador secuencial para TEIMUS Profano/Religioso basado en tokens MoMeT.

Flujo nuevo, coexistente con el pipeline tabular existente:
    MusicXML -> tokenizacion simbolica MoMeT -> ids -> Embedding -> CNN1D -> pooling enmascarado -> logit

Convencion de clases del proyecto:
    Religioso = 0
    Profano   = 1

El modulo esta aislado para que futuros modelos secuenciales puedan incorporarse
sin modificar la logica de features. La API publica replica train/eval del
classifier.py principal y genera el mismo CSV de predicciones usado por las
comparativas y curvas del pipeline.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"
CLS_TOKEN = "[CLS]"
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2


DEFAULT_ALLOWED_DURATIONS = [
    0.25, 0.333333, 0.5, 0.666667, 0.75,
    1.0, 1.333333, 1.5, 2.0, 3.0, 4.0,
]


@dataclass(frozen=True)
class SequenceClassifierConfig:
    """Configuracion moderada para reducir sobreajuste en corpus pequeno."""

    model_type: str = "cnn1d"
    embedding_dim: int = 128
    conv_channels: int = 128
    hidden_dim: int = 64
    kernel_size_1: int = 3
    kernel_size_2: int = 5
    dropout: float = 0.40
    batch_size: int = 16
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    validation_size: float = 0.15
    pooling: str = "masked_max"  # "masked_max" o "masked_mean"
    max_length: Optional[int] = None  # None conserva la longitud completa por batch.
    num_workers: int = 0
    mark_beats: bool = True
    mark_bars: bool = True
    mark_ts_changes: bool = True
    mark_grace_notes: bool = False
    strict_measure_rejection: bool = False
    strict_modality_rejection: bool = False
    allowed_durations: Tuple[float, ...] = tuple(DEFAULT_ALLOWED_DURATIONS)


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_momet_tokenizer_function():
    """Carga el tokenizador MoMeT sin imponer una ruta unica de paquete.

    En distintas versiones del proyecto ha vivido en paquetes con nombres
    ligeramente distintos. Se intenta primero la ruta indicada para TransFolk y
    despues alias historicos, sin modificar el tokenizador original.
    """

    candidates = [
        "transkfolk_tokenization.tokenizer",  # nombre indicado en el prompt del pipeline actual
        "transfolk_tokenization.tokenizer",
        "tokenization.tokenizer",
        "transfolk_tokenizer.tokenizer",
    ]
    last_error = None
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["extract_tokens_with_meter_modulation_full_measures"])
            return module.extract_tokens_with_meter_modulation_full_measures
        except Exception as exc:  # pragma: no cover - depende del layout local del proyecto
            last_error = exc
    raise ImportError(
        "No se pudo importar extract_tokens_with_meter_modulation_full_measures desde el tokenizador MoMeT. "
        "Comprueba que transkfolk_tokenization.tokenizer este accesible. Ultimo error: %r" % (last_error,)
    )


def tokenize_musicxml_momet(path: str, config: SequenceClassifierConfig) -> Optional[List[str]]:
    """Tokeniza una pieza con MoMeT y antepone [CLS] sin tocar el tokenizador."""

    tokenizer_fn = _load_momet_tokenizer_function()
    errors: Dict[str, list] = {}
    try:
        tokens = tokenizer_fn(
            xml_path=path,
            time_signature="",          # no filtra porque strict_measure_rejection=False
            tonality="",                # no filtra porque strict_modality_rejection=False
            allowed_durations=list(config.allowed_durations),
            errors=errors,
            mark_bars=config.mark_bars,
            strict_measure_rejection=config.strict_measure_rejection,
            strict_modality_rejection=config.strict_modality_rejection,
            mark_ts_changes=config.mark_ts_changes,
            mark_grace_notes=config.mark_grace_notes,
            mark_beats=config.mark_beats,
        )
    except TypeError:
        # Fallback para firmas antiguas del tokenizador.
        tokens = tokenizer_fn(path, "", "", list(config.allowed_durations), errors)
    except Exception as exc:
        print("[WARN] Error tokenizando con MoMeT:", os.path.basename(path), "->", repr(exc))
        return None

    if not tokens:
        return None
    return [CLS_TOKEN] + [str(t) for t in tokens]


def build_vocab_from_training_sequences(token_sequences: Iterable[Sequence[str]]) -> Dict[str, int]:
    """Construye el vocabulario solo con training para evitar data leakage."""

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, CLS_TOKEN: CLS_ID}
    next_id = 3
    for seq in token_sequences:
        for tok in seq:
            tok = str(tok)
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
    return vocab


def tokens_to_ids(tokens: Sequence[str], vocab: Dict[str, int], max_length: Optional[int] = None) -> List[int]:
    ids = [int(vocab.get(str(tok), UNK_ID)) for tok in tokens]
    if max_length is not None and len(ids) > int(max_length):
        ids = ids[: int(max_length)]
        ids[0] = CLS_ID
    return ids


class MusicSequenceDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[List[int], int, str, str]]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ids, y, path, fname = self.items[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(float(y), dtype=torch.float32),
            "path": path,
            "file": fname,
        }


def collate_variable_length(batch: Sequence[dict]) -> dict:
    """Padding dinamico por batch y mascara binaria 1=token real, 0=PAD."""

    max_len = max(int(x["input_ids"].numel()) for x in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    labels = torch.zeros((len(batch),), dtype=torch.float32)
    paths, files = [], []

    for i, item in enumerate(batch):
        seq = item["input_ids"]
        n = int(seq.numel())
        input_ids[i, :n] = seq
        mask[i, :n] = 1.0
        labels[i] = item["label"]
        paths.append(item["path"])
        files.append(item["file"])

    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels, "paths": paths, "files": files}


class CNN1DSequenceClassifier(nn.Module):
    """Embedding -> Conv1D -> ReLU -> Dropout -> Conv1D -> ReLU -> masked pooling -> dense.

    La entrada del embedding es [B, L] y produce [B, L, E]. Para Conv1D se
    transpone a [B, E, L]. La mascara evita que el pooling use posiciones PAD.
    """

    def __init__(self, vocab_size: int, config: SequenceClassifierConfig, pad_id: int = PAD_ID):
        super().__init__()
        self.config = config
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=self.pad_id)
        self.conv1 = nn.Conv1d(
            in_channels=config.embedding_dim,
            out_channels=config.conv_channels,
            kernel_size=config.kernel_size_1,
            padding=config.kernel_size_1 // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.conv_channels,
            out_channels=config.conv_channels,
            kernel_size=config.kernel_size_2,
            padding=config.kernel_size_2 // 2,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.conv_channels, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def _masked_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], mask: [B, L]
        mask = mask.unsqueeze(1).to(dtype=x.dtype)
        if self.config.pooling == "masked_mean":
            x = x * mask
            denom = mask.sum(dim=2).clamp_min(1.0)
            return x.sum(dim=2) / denom

        # masked max pooling. Las posiciones PAD no pueden convertirse en maximos.
        x = x.masked_fill(mask <= 0, -1e9)
        pooled = x.max(dim=2).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)      # [B, L, E]
        x = x.transpose(1, 2)              # [B, E, L]
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.activation(self.conv2(x)))
        x = self._masked_pool(x, attention_mask)
        return self.classifier(x).squeeze(-1)  # logits [B]


# Punto de extension: se pueden registrar aqui BiLSTM/Transformer con la misma firma.
SEQUENCE_MODEL_REGISTRY = {
    "cnn1d": CNN1DSequenceClassifier,
}


def build_sequence_model(vocab_size: int, config: SequenceClassifierConfig) -> nn.Module:
    if config.model_type not in SEQUENCE_MODEL_REGISTRY:
        raise ValueError(f"Modelo secuencial no soportado: {config.model_type}. Disponibles: {sorted(SEQUENCE_MODEL_REGISTRY)}")
    return SEQUENCE_MODEL_REGISTRY[config.model_type](vocab_size=vocab_size, config=config, pad_id=PAD_ID)


def _prepare_sequence_items(
    df_meta: pd.DataFrame,
    config: SequenceClassifierConfig,
    vocab: Optional[Dict[str, int]] = None,
    build_vocab: bool = False,
) -> Tuple[List[Tuple[List[int], int, str, str]], Optional[Dict[str, int]], pd.DataFrame]:
    token_rows = []
    sequences = []

    for _, r in df_meta.iterrows():
        path = str(r["__path"])
        y = int(r["label_id"])
        fname = os.path.basename(path)
        tokens = tokenize_musicxml_momet(path, config)
        if not tokens:
            print("[WARN] Sin tokens MoMeT:", fname)
            continue
        token_rows.append({"file": fname, "path": path, "label_id": y, "n_tokens": len(tokens)})
        sequences.append((tokens, y, path, fname))

    if not sequences:
        raise RuntimeError("No se pudieron tokenizar piezas MusicXML con MoMeT.")

    if build_vocab:
        vocab = build_vocab_from_training_sequences(seq for seq, _, _, _ in sequences)
    if vocab is None:
        raise ValueError("vocab no puede ser None si build_vocab=False.")

    items = []
    for tokens, y, path, fname in sequences:
        ids = tokens_to_ids(tokens, vocab, max_length=config.max_length)
        if len(ids) == 0:
            continue
        items.append((ids, y, path, fname))

    return items, vocab, pd.DataFrame(token_rows)


def _make_loader(items, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        MusicSequenceDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
    )


def _run_epoch(model, loader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    losses, all_y, all_pred = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(input_ids=input_ids, attention_mask=mask)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        all_pred.extend((probs >= 0.5).astype(int).tolist())
        all_y.extend(labels.detach().cpu().numpy().astype(int).tolist())

    bacc = balanced_accuracy_score(all_y, all_pred) if len(set(all_y)) > 1 else 0.0
    return float(np.mean(losses)), float(bacc)


def _predict(model, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
            labels = batch["labels"].detach().cpu().numpy().astype(int)
            preds = (probs >= 0.5).astype(int)
            for fname, path, y, yp, score in zip(batch["files"], batch["paths"], labels, preds, probs):
                rows.append({"file": fname, "path": path, "y_true": int(y), "y_pred": int(yp), "score_profano": float(score)})
    return pd.DataFrame(rows)


def train_and_save_sequence_model(
    df_split: pd.DataFrame,
    algorithm_id: int,
    out_dir: str,
    algorithm_name: str = "cnn1d_momet",
    label_prof: str = "Profano",
    label_reli: str = "Religioso",
    random_state: int = 42,
    config: Optional[SequenceClassifierConfig] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    config = config or SequenceClassifierConfig()
    _set_reproducibility(random_state)
    dev = _device()

    df_train_meta = df_split[df_split["split"] == "train"].copy()
    all_items, vocab, token_table = _prepare_sequence_items(df_train_meta, config, build_vocab=True)

    y_all = np.asarray([y for _, y, _, _ in all_items], dtype=int)
    indices = np.arange(len(all_items))
    if len(np.unique(y_all)) == 2 and len(all_items) >= 10:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=config.validation_size,
            random_state=random_state,
            stratify=y_all,
        )
    else:
        train_idx, val_idx = indices, indices

    train_items = [all_items[i] for i in train_idx]
    val_items = [all_items[i] for i in val_idx]

    train_loader = _make_loader(train_items, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = _make_loader(val_items, config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_sequence_model(vocab_size=len(vocab), config=config).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []

    print("[INFO] Entrenando clasificador secuencial MoMeT CNN1D en", dev)
    print(f"[INFO] n_train={len(train_items)} n_val={len(val_items)} vocab={len(vocab)}")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_bacc = _run_epoch(model, train_loader, criterion, optimizer, dev)
        val_loss, val_bacc = _run_epoch(model, val_loader, criterion, None, dev)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_balanced_accuracy": train_bacc, "val_loss": val_loss, "val_balanced_accuracy": val_bacc})
        print(f"[INFO] epoch={epoch:03d} train_loss={train_loss:.4f} train_bacc={train_bacc:.4f} val_loss={val_loss:.4f} val_bacc={val_bacc:.4f}")

        if val_bacc > best_val + 1e-6:
            best_val = val_bacc
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"[INFO] Early stopping en epoch={epoch}. Mejor epoch={best_epoch}, val_bacc={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = Path(out_dir) / "model.pt"
    vocab_path = Path(out_dir) / "vocab.json"
    config_path = Path(out_dir) / "sequence_config.json"
    summary_path = Path(out_dir) / "train_summary.json"
    history_path = Path(out_dir) / "training_history.csv"
    token_cache_path = Path(out_dir) / "train_tokenization_summary.csv"

    torch.save({"model_state_dict": model.state_dict(), "vocab_size": len(vocab)}, model_path)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")
    token_table.to_csv(token_cache_path, index=False, encoding="utf-8")

    summary = {
        "algorithm_id": int(algorithm_id),
        "algorithm_name": algorithm_name,
        "model_family": "sequential_momet",
        "label_prof": label_prof,
        "label_reli": label_reli,
        "n_train_tokenized": int(len(all_items)),
        "n_train_used": int(len(train_items)),
        "n_validation_used": int(len(val_items)),
        "n_prof": int((y_all == 1).sum()),
        "n_reli": int((y_all == 0).sum()),
        "vocab_size": int(len(vocab)),
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_val),
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "cls_id": CLS_ID,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Modelo secuencial guardado en:", model_path)
    return out_dir


def load_sequence_model_and_evaluate(
    model_dir: str,
    df_split: pd.DataFrame,
    out_csv_path: str = "eval_predictions_cnn1d_momet.csv",
) -> pd.DataFrame:
    model_dir_p = Path(model_dir)
    model_path = model_dir_p / "model.pt"
    vocab_path = model_dir_p / "vocab.json"
    config_path = model_dir_p / "sequence_config.json"
    summary_path = model_dir_p / "train_summary.json"

    if not model_path.exists() or not vocab_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"No encuentro model.pt, vocab.json o sequence_config.json en: {model_dir}")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(raw_cfg.get("allowed_durations"), list):
        raw_cfg["allowed_durations"] = tuple(raw_cfg["allowed_durations"])
    config = SequenceClassifierConfig(**raw_cfg)

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    algorithm_id = summary.get("algorithm_id", None)
    algorithm_name = summary.get("algorithm_name", model_dir_p.name)
    label_prof = summary.get("label_prof", "Profano")
    label_reli = summary.get("label_reli", "Religioso")

    dev = _device()
    checkpoint = torch.load(model_path, map_location=dev)
    model = build_sequence_model(vocab_size=len(vocab), config=config).to(dev)
    model.load_state_dict(checkpoint["model_state_dict"])

    df_eval_meta = df_split[df_split["split"].isin(["test", "eval"])].copy()
    eval_items, _, token_table = _prepare_sequence_items(df_eval_meta, config, vocab=vocab, build_vocab=False)
    eval_loader = _make_loader(eval_items, config.batch_size, shuffle=False, num_workers=config.num_workers)

    print("[INFO] Evaluando clasificador secuencial MoMeT CNN1D...")
    df_out = _predict(model, eval_loader, dev)
    if df_out.empty:
        raise RuntimeError("No se generaron predicciones secuenciales.")

    df_out["score_religioso"] = 1.0 - pd.to_numeric(df_out["score_profano"], errors="coerce")
    df_out["label_pred"] = [label_prof if int(yp) == 1 else label_reli for yp in df_out["y_pred"].values]
    df_out["algorithm_id"] = algorithm_id
    df_out["algorithm_name"] = algorithm_name
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8")
    token_table.to_csv(model_dir_p / "eval_tokenization_summary.csv", index=False, encoding="utf-8")

    print("[OK] CSV guardado:", out_csv_path)

    y_true = df_out["y_true"].astype(int).values
    y_pred = df_out["y_pred"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_true, df_out["score_profano"].values)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred)

    print("\n==============================")
    print("RESUMEN EVALUACION SECUENCIAL")
    print("==============================")
    print("Modelo:", algorithm_name)
    print("Accuracy:          %.4f" % acc)
    print("Balanced accuracy: %.4f" % bacc)
    print("F1 weighted:       %.4f" % f1w)
    if np.isfinite(auc):
        print("ROC-AUC:           %.4f" % auc)
    print("\nMatriz de confusion")
    print(cm)
    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=[label_reli, label_prof]))

    return df_out
