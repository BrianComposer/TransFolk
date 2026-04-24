# -*- coding: utf-8 -*-
"""
ablation_study.py

Ablation study para el clasificador Profano vs Religioso.

Contraste experimental:
    - baseline_generic: rasgos simbólicos generales, sin rasgos musicológicos
      idiomáticos específicos.
    - full_musical: todas las features disponibles, incluyendo las features
      musicales diseñadas para el problema religioso/profano.

El módulo se integra con el pipeline existente porque recibe el mismo df_split
que build_split_metadata(), reutiliza make_model() de classifier.py y guarda
los resultados en la carpeta del experimento con seed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from transfolk_classifier.classifier import ALGORITHM_NAME, make_model


# ---------------------------------------------------------------------
# 1) Feature sets del ablation
# ---------------------------------------------------------------------
# Baseline: descriptors generales de MIR simbólico. Estos rasgos describen
# ritmo, intervalos, contorno, rango y tonalidad de forma genérica, pero no
# codifican explícitamente las hipótesis musicológicas del estudio.
GENERIC_BASELINE_FEATURES: List[str] = [
    "note_density", "mean_dur", "cv_dur", "short_note_ratio",
    "rhythmic_entropy", "npvi_rhythmic", "rhythmic_energy",
    "strong_beat_note_ratio", "syncopation_index",
    "reciting_pitch_ratio", "pitch_entropy",
    "final_tonic_match", "last_note_duration_ratio", "last_interval_abs",
    "last_interval_is_step",
    "diatonic_ratio", "best_corr", "key_clarity", "best_key_pc",
    "best_mode_minor",
    "mean_abs_semitones", "interval_std", "max_leap", "step_ratio",
    "unison_ratio", "interval_2nd_ratio", "interval_3rd_ratio",
    "interval_4th_ratio", "interval_5th_ratio", "interval_6th_ratio",
    "interval_7th_ratio", "interval_octave_plus_ratio", "tritone_ratio",
    "consonant_interval_ratio",
    "range_semitones", "range_p95", "range_relative",
    "proximity_step_le2", "proximity_inv_mean",
    "up_ratio", "down_ratio", "stay_ratio", "num_direction_changes",
    "climax_pos", "direction_balance", "pitch_time_slope", "mean_ioi",
]

# Features musicológicas/idiomáticas. Son las que se añaden en full_musical
# y cuya contribución se quiere demostrar frente al baseline.
MUSICAL_IDIOMATIC_FEATURES: List[str] = [
    "grace_note_ratio", "short_ornament_window_ratio", "turn_like_ratio",
    "appoggiatura_like_ratio",
    "triplet_in_binary_ratio", "dotted_rhythm_ratio",
    "initial_interval_is_6th", "final_leading_tone_appoggiatura",
    "strong_weak_semitone_resolution_ratio",
    "chromatic_usage_ratio", "retardo_la_solsharp_ratio",
    "retardo_si_la_ratio", "minor_leading_to_mediant_ratio",
]

ABLATION_CONFIG = {
    "baseline_generic": GENERIC_BASELINE_FEATURES,
    "full_musical": None,  # None = todas las features extraídas
}


# ---------------------------------------------------------------------
# 2) Estilo gráfico de paper
# ---------------------------------------------------------------------
def _setup_paper_style(style: str = "tableau-colorblind10", font_size: int = 13) -> None:
    import matplotlib.pyplot as plt

    if style in plt.style.available:
        plt.style.use(style)
    elif "tableau-colorblind10" in plt.style.available:
        plt.style.use("tableau-colorblind10")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
    })


def _save_figure(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".eps")), format="eps", bbox_inches="tight")


# ---------------------------------------------------------------------
# 3) Extracción/cache de features
# ---------------------------------------------------------------------
def _extract_feature_table(
    df_split: pd.DataFrame,
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    cache_csv_path: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Extrae features una sola vez y las cachea para no repetir parseo MusicXML."""
    if cache_csv_path.exists() and not force_recompute:
        print("[INFO] Cargando cache de features:", cache_csv_path)
        return pd.read_csv(cache_csv_path)

    rows = []
    print("[INFO] Extrayendo features para ablation study...")

    for i, r in df_split.reset_index(drop=True).iterrows():
        p = r["__path"]
        try:
            feats = extract_features_musicxml(p)
            if not feats:
                print("[WARN] Sin features:", os.path.basename(p))
                continue
            row = dict(feats)
            row["__path"] = p
            row["__file"] = os.path.basename(p)
            row["__y"] = int(r["label_id"])
            row["__split"] = str(r["split"])
            rows.append(row)
        except Exception as exc:
            print("[WARN] Error extrayendo features:", os.path.basename(p), "->", repr(exc))

        if (i + 1) % 50 == 0:
            print(f"[INFO] Procesadas {i + 1}/{len(df_split)} obras")

    if not rows:
        raise RuntimeError("No se pudieron extraer features para el ablation study.")

    df_features = pd.DataFrame(rows)
    meta = {"__path", "__file", "__y", "__split"}
    for c in df_features.columns:
        if c not in meta:
            df_features[c] = pd.to_numeric(df_features[c], errors="coerce")

    cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(cache_csv_path, index=False, encoding="utf-8")
    print("[OK] Cache guardada:", cache_csv_path)
    return df_features


def _feature_columns(df_features: pd.DataFrame) -> List[str]:
    meta = {"__path", "__file", "__y", "__split"}
    return [c for c in df_features.columns if c not in meta]


def _resolve_subset(df_features: pd.DataFrame, requested: Optional[Sequence[str]], name: str) -> List[str]:
    available = _feature_columns(df_features)
    if requested is None:
        return available
    selected = [f for f in requested if f in available]
    missing = [f for f in requested if f not in available]
    if missing:
        print(f"[WARN] {name}: features no encontradas y omitidas:", missing)
    if not selected:
        raise RuntimeError(f"La condición {name} no tiene features válidas.")
    return selected


# ---------------------------------------------------------------------
# 4) Entrenamiento/evaluación
# ---------------------------------------------------------------------
def _predict_scores(model, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve predicción binaria y score de la clase positiva Profano (=1)."""
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        return (scores >= 0.5).astype(int), scores

    if hasattr(model, "decision_function"):
        dec = np.asarray(model.decision_function(X), dtype=float)
        lo, hi = float(np.min(dec)), float(np.max(dec))
        scores = np.ones_like(dec) * 0.5 if np.isclose(lo, hi) else (dec - lo) / (hi - lo)
        return (scores >= 0.5).astype(int), scores

    pred = np.asarray(model.predict(X), dtype=int)
    return pred, pred.astype(float)


def _safe(fn, default=np.nan):
    try:
        return fn()
    except Exception:
        return default


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(_safe(lambda: roc_auc_score(y_true, y_score))),
        "average_precision": float(_safe(lambda: average_precision_score(y_true, y_score))),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _run_condition(
    df_features: pd.DataFrame,
    feature_cols: List[str],
    algorithm_id: int,
    condition: str,
    out_dir: Path,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Entrena y evalúa una condición del ablation."""
    train = df_features[df_features["__split"].astype(str).eq("train")].copy()
    test = df_features[df_features["__split"].astype(str).isin(["test", "eval"])].copy()
    if train.empty or test.empty:
        raise RuntimeError("El ablation necesita train y test/eval no vacíos.")

    X_train = train[feature_cols]
    y_train = train["__y"].astype(int).values
    X_test = test[feature_cols]
    y_test = test["__y"].astype(int).values

    model = make_model(algorithm_id=algorithm_id, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred, y_score = _predict_scores(model, X_test)
    met = _metrics(y_test, y_pred, y_score)
    met.update({
        "condition": condition,
        "algorithm_id": int(algorithm_id),
        "algorithm_name": ALGORITHM_NAME[algorithm_id],
        "n_train": int(len(train)),
        "n_eval": int(len(test)),
        "n_features": int(len(feature_cols)),
    })

    cond_dir = out_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cond_dir / "model.joblib")
    (cond_dir / "features.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    (cond_dir / "metrics.json").write_text(json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")

    df_pred = pd.DataFrame({
        "file": test["__file"].values,
        "path": test["__path"].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "score_profano": y_score,
        "condition": condition,
        "algorithm_id": algorithm_id,
        "algorithm_name": ALGORITHM_NAME[algorithm_id],
    })
    df_pred.to_csv(cond_dir / "predictions.csv", index=False, encoding="utf-8")
    return df_pred, met


# ---------------------------------------------------------------------
# 5) Gráficas
# ---------------------------------------------------------------------
def _plot_metrics(df: pd.DataFrame, out_dir: Path, algorithm_name: str, style: str) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average precision"),
    ]
    conditions = ["baseline_generic", "full_musical"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, cond in enumerate(conditions):
        row = df[df["condition"] == cond]
        if row.empty:
            continue
        vals = [float(row.iloc[0][m]) for m, _ in metrics]
        label = "Without musical features" if cond == "baseline_generic" else "With musical features"
        bars = ax.bar(x + (-width / 2 if i == 0 else width / 2), vals, width, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_title(f"Ablation study — {algorithm_name.replace('_', ' ').title()}")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="lower right")
    _save_figure(fig, out_dir / f"ablation_metrics_{algorithm_name}")
    plt.close(fig)


def _plot_delta(df: pd.DataFrame, out_dir: Path, algorithm_name: str, style: str) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    base = df[df["condition"] == "baseline_generic"]
    full = df[df["condition"] == "full_musical"]
    if base.empty or full.empty:
        return
    base, full = base.iloc[0], full.iloc[0]
    metrics = [
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average\nprecision"),
    ]
    deltas = [float(full[m]) - float(base[m]) for m, _ in metrics]
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    bars = ax.bar(x, deltas, width=0.55)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_title(f"Effect of musical features — {algorithm_name.replace('_', ' ').title()}")
    ax.set_ylabel("Δ metric value")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6)
    ax.grid(False, axis="x")
    max_abs = max(0.05, float(np.nanmax(np.abs(deltas))))
    ax.set_ylim(-max_abs * 1.45, max_abs * 1.55)
    for b, v in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
    _save_figure(fig, out_dir / f"ablation_delta_{algorithm_name}")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6) Interpretación automática
# ---------------------------------------------------------------------
def generate_ablation_interpretation(df: pd.DataFrame, out_path: Path, main_metric: str = "balanced_accuracy") -> str:
    base = df[df["condition"] == "baseline_generic"]
    full = df[df["condition"] == "full_musical"]
    if base.empty or full.empty:
        text = "# Ablation study interpretation\n\nNo se han encontrado las dos condiciones necesarias.\n"
        out_path.write_text(text, encoding="utf-8")
        return text

    base, full = base.iloc[0], full.iloc[0]
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    deltas = {m: float(full[m]) - float(base[m]) for m in metrics if m in df.columns and pd.notna(full[m]) and pd.notna(base[m])}
    main_delta = deltas.get(main_metric, np.nan)

    if pd.isna(main_delta):
        verdict = "No se puede evaluar la métrica principal porque no está disponible."
    elif main_delta > 0.02:
        verdict = "La mejora es clara: las features musicales aportan información estilística adicional."
    elif main_delta > 0.005:
        verdict = "La mejora es positiva pero moderada; conviene reforzarla con medias multi-seed."
    elif main_delta >= -0.005:
        verdict = "La diferencia es muy pequeña; la evidencia de contribución en esta seed es débil."
    else:
        verdict = "La condición completa empeora al baseline en esta seed; revisa estabilidad multi-seed o selección de features."

    def fmt(x):
        return "nan" if pd.isna(x) else f"{float(x):.4f}"

    lines = [
        "# Ablation study interpretation", "",
        f"**Model:** `{full['algorithm_name']}`", "",
        "## Experimental contrast", "",
        f"The ablation compares a generic baseline with `{int(base['n_features'])}` features against the full musical feature set with `{int(full['n_features'])}` features.",
        "", "## Results", "",
        "| Metric | Without musical features | With musical features | Δ |",
        "|---|---:|---:|---:|",
    ]
    for m in metrics:
        if m in df.columns:
            lines.append(f"| {m} | {fmt(base[m])} | {fmt(full[m])} | {deltas.get(m, np.nan):+.4f} |")
    lines += [
        "", "## Automatic interpretation", "", verdict, "",
        f"The main metric `{main_metric}` changes from {fmt(base[main_metric])} to {fmt(full[main_metric])} ({main_delta:+.4f}).",
        "", "## Suggested paper sentence", "",
    ]
    if not pd.isna(main_delta) and main_delta > 0:
        lines.append("The ablation study shows that the proposed musicologically motivated features improve the classifier over a generic symbolic baseline, suggesting that the observed performance is driven not only by low-level descriptors but also by stylistically informed musical cues.")
    else:
        lines.append("The ablation study does not show a robust improvement for the full feature set in this configuration, so the contribution should be checked through multi-seed aggregation and possible feature selection.")

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------
# 7) Función pública
# ---------------------------------------------------------------------
def run_ablation_study(
    df_split: pd.DataFrame,
    algorithm_id: int,
    output_dir: str,
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    random_state: int = 42,
    style: str = "tableau-colorblind10",
    force_recompute_features: bool = False,
    main_metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Ejecuta el ablation study completo y guarda CSV, modelos, gráficas e interpretación."""
    if algorithm_id not in ALGORITHM_NAME:
        raise ValueError(f"algorithm_id inválido: {algorithm_id}. Debe estar en {sorted(ALGORITHM_NAME)}.")

    output_path = Path(output_dir)
    algorithm_name = ALGORITHM_NAME[algorithm_id]
    ablation_dir = output_path / "ablation_study" / algorithm_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    df_features = _extract_feature_table(
        df_split=df_split,
        extract_features_musicxml=extract_features_musicxml,
        cache_csv_path=output_path / "ablation_features_cache.csv",
        force_recompute=force_recompute_features,
    )

    metrics_rows = []
    pred_rows = []
    for condition, requested in ABLATION_CONFIG.items():
        feature_cols = _resolve_subset(df_features, requested, condition)
        print("\n==============================")
        print("ABLATION CONDITION")
        print("==============================")
        print("Modelo:", algorithm_name)
        print("Condición:", condition)
        print("N features:", len(feature_cols))

        df_pred, met = _run_condition(
            df_features=df_features,
            feature_cols=feature_cols,
            algorithm_id=algorithm_id,
            condition=condition,
            out_dir=ablation_dir,
            random_state=random_state,
        )
        metrics_rows.append(met)
        pred_rows.append(df_pred)

    df_metrics = pd.DataFrame(metrics_rows).sort_values("condition").reset_index(drop=True)
    df_metrics.to_csv(ablation_dir / f"ablation_metrics_{algorithm_name}.csv", index=False, encoding="utf-8")
    pd.concat(pred_rows, ignore_index=True).to_csv(ablation_dir / f"ablation_predictions_{algorithm_name}.csv", index=False, encoding="utf-8")

    _plot_metrics(df_metrics, ablation_dir, algorithm_name, style)
    _plot_delta(df_metrics, ablation_dir, algorithm_name, style)
    interpretation = generate_ablation_interpretation(
        df_metrics,
        ablation_dir / f"ablation_interpretation_{algorithm_name}.md",
        main_metric=main_metric,
    )

    print("\n==============================")
    print("ABLATION STUDY RESULTS")
    print("==============================")
    print(df_metrics.to_string(index=False))
    print("\n" + interpretation)
    return df_metrics


# ---------------------------------------------------------------------
# 8) Resumen multi-seed para el paper
# ---------------------------------------------------------------------
def _collect_ablation_metric_files(
    base_results_dir: str | os.PathLike,
    seeds: Iterable[int],
    algorithm_names: Iterable[str],
) -> pd.DataFrame:
    """Lee los CSV por seed/algoritmo generados por run_ablation_study()."""
    base_results_dir = Path(base_results_dir)
    rows = []

    for seed in seeds:
        for algorithm_name in algorithm_names:
            csv_path = (
                base_results_dir
                / str(seed)
                / "ablation_study"
                / algorithm_name
                / f"ablation_metrics_{algorithm_name}.csv"
            )
            if not csv_path.exists():
                print("[WARN] No existe ablation metrics:", csv_path)
                continue

            df = pd.read_csv(csv_path)
            df["seed"] = int(seed)
            df["source_csv"] = str(csv_path)
            rows.append(df)

    if not rows:
        raise RuntimeError("No se encontraron CSV de ablation study para agregar.")

    return pd.concat(rows, ignore_index=True)


def _build_delta_rows(df_all: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Construye filas delta = full_musical - baseline_generic por seed y algoritmo."""
    rows = []
    group_cols = ["algorithm_id", "algorithm_name", "seed"]

    for keys, g in df_all.groupby(group_cols, dropna=False):
        algorithm_id, algorithm_name, seed = keys
        base = g[g["condition"] == "baseline_generic"]
        full = g[g["condition"] == "full_musical"]
        if base.empty or full.empty:
            continue
        base = base.iloc[0]
        full = full.iloc[0]
        row = {
            "algorithm_id": algorithm_id,
            "algorithm_name": algorithm_name,
            "seed": int(seed),
            "baseline_n_features": int(base.get("n_features", -1)),
            "full_n_features": int(full.get("n_features", -1)),
        }
        for m in metrics:
            if m in g.columns:
                row[f"baseline_{m}"] = float(base[m])
                row[f"full_{m}"] = float(full[m])
                row[f"delta_{m}"] = float(full[m]) - float(base[m])
        rows.append(row)

    if not rows:
        raise RuntimeError("No se pudieron construir deltas de ablation.")

    return pd.DataFrame(rows)


def _plot_multiseed_summary(
    df_delta: pd.DataFrame,
    out_dir: Path,
    algorithm_name: str,
    metric: str,
    style: str,
) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style, font_size=12)
    df_alg = df_delta[df_delta["algorithm_name"] == algorithm_name].copy()
    if df_alg.empty:
        return

    values = [df_alg[f"baseline_{metric}"].astype(float), df_alg[f"full_{metric}"].astype(float)]
    means = [float(v.mean()) for v in values]
    stds = [float(v.std(ddof=1)) if len(v) > 1 else 0.0 for v in values]
    labels = ["Without musical\nfeatures", "With musical\nfeatures"]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    bars = ax.bar(np.arange(2), means, yerr=stds, capsize=5, width=0.55)
    for b, mean, std in zip(bars, means, stds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            mean + std + 0.012,
            f"{mean:.3f} ± {std:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    delta_mean = float(df_alg[f"delta_{metric}"].mean())
    delta_std = float(df_alg[f"delta_{metric}"].std(ddof=1)) if len(df_alg) > 1 else 0.0
    ax.set_title(f"Ablation study — {algorithm_name.replace('_', ' ').title()}")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, min(1.10, max(1.0, max(means) + max(stds) + 0.12)))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6)
    ax.grid(False, axis="x")
    ax.text(
        0.5,
        0.04,
        f"Δ = {delta_mean:+.3f} ± {delta_std:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )
    _save_figure(fig, out_dir / f"ablation_multiseed_{metric}_{algorithm_name}")
    plt.close(fig)


def _write_multiseed_interpretation(
    df_delta: pd.DataFrame,
    out_path: Path,
    main_metric: str,
) -> str:
    lines = [
        "# Multi-seed ablation study interpretation",
        "",
        "The study compares a generic symbolic baseline without the musicological idiomatic features against the complete feature set.",
        "",
        "| Algorithm | Without musical features | With musical features | Δ | Seeds |",
        "|---|---:|---:|---:|---:|",
    ]

    for algorithm_name, g in df_delta.groupby("algorithm_name"):
        base = g[f"baseline_{main_metric}"].astype(float)
        full = g[f"full_{main_metric}"].astype(float)
        delta = g[f"delta_{main_metric}"].astype(float)
        lines.append(
            f"| {algorithm_name} | {base.mean():.4f} ± {base.std(ddof=1):.4f} | "
            f"{full.mean():.4f} ± {full.std(ddof=1):.4f} | "
            f"{delta.mean():+.4f} ± {delta.std(ddof=1):.4f} | {len(g)} |"
        )

    best = (
        df_delta.groupby("algorithm_name")[f"delta_{main_metric}"]
        .mean()
        .sort_values(ascending=False)
    )
    best_name = str(best.index[0])
    best_delta = float(best.iloc[0])

    lines += ["", "## Automatic interpretation", ""]
    if best_delta > 0.02:
        lines.append(
            f"The strongest average effect is obtained by `{best_name}`. The full feature set improves `{main_metric}` by {best_delta:+.4f}, which supports the claim that the musicological features add discriminative information beyond generic symbolic descriptors."
        )
    elif best_delta > 0.005:
        lines.append(
            f"The strongest average effect is obtained by `{best_name}`. The gain in `{main_metric}` is {best_delta:+.4f}; this is positive but should be presented as a moderate improvement."
        )
    else:
        lines.append(
            f"The best average gain is {best_delta:+.4f} for `{best_name}`. This does not provide strong evidence that the full feature set improves over the generic baseline; feature selection or a larger evaluation split may be needed."
        )

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def aggregate_ablation_results(
    base_results_dir: str,
    seeds: Iterable[int],
    algorithm_names: Iterable[str],
    style: str = "tableau-colorblind10",
    main_metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """
    Agrega los resultados de ablation de todas las seeds y genera material directo para paper.

    Outputs en <base_results_dir>/ablation_study_summary/:
        - ablation_all_seed_metrics.csv
        - ablation_delta_by_seed.csv
        - ablation_summary_mean_std.csv
        - ablation_multiseed_<metric>_<algorithm>.png/.eps
        - ablation_multiseed_interpretation.md
    """
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    base_results_dir = Path(base_results_dir)
    out_dir = base_results_dir / "ablation_study_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = _collect_ablation_metric_files(base_results_dir, seeds, algorithm_names)
    df_all.to_csv(out_dir / "ablation_all_seed_metrics.csv", index=False, encoding="utf-8")

    df_delta = _build_delta_rows(df_all, metrics)
    df_delta.to_csv(out_dir / "ablation_delta_by_seed.csv", index=False, encoding="utf-8")

    summary_rows = []
    for algorithm_name, g in df_delta.groupby("algorithm_name"):
        row = {"algorithm_name": algorithm_name, "n_seeds": int(len(g))}
        for m in metrics:
            for prefix in ["baseline", "full", "delta"]:
                col = f"{prefix}_{m}"
                if col in g.columns:
                    vals = g[col].astype(float)
                    row[f"{col}_mean"] = float(vals.mean())
                    row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values(f"delta_{main_metric}_mean", ascending=False)
    df_summary.to_csv(out_dir / "ablation_summary_mean_std.csv", index=False, encoding="utf-8")

    for algorithm_name in df_delta["algorithm_name"].dropna().unique():
        _plot_multiseed_summary(df_delta, out_dir, str(algorithm_name), main_metric, style)

    interpretation = _write_multiseed_interpretation(
        df_delta=df_delta,
        out_path=out_dir / "ablation_multiseed_interpretation.md",
        main_metric=main_metric,
    )

    print("\n==============================")
    print("MULTI-SEED ABLATION SUMMARY")
    print("==============================")
    print(df_summary.to_string(index=False))
    print("\n" + interpretation)
    print("[OK] Resumen multi-seed guardado en:", out_dir)
    return df_summary
