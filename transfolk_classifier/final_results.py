# -*- coding: utf-8 -*-
"""
final_results.py

Agregación final multi-seed para el pipeline Profano/Religioso de TEIMUS.

Este módulo NO recalcula curvas ni predicciones. Lee los CSV generados en cada
experimento/seed y produce figuras finales con media ± desviación estándar en:

    <base_results_dir>/final_results/

Entradas esperadas por seed y algoritmo:
    - roc_curve_<algorithm>.csv
    - precision_recall_curve_<algorithm>.csv
    - threshold_analysis_<algorithm>.csv
    - model_comparison.csv
    - feature_importance_<algorithm>.csv
    - ablation_study/<algorithm>/ablation_metrics_<algorithm>.csv
    - musical_interpretability/<algorithm>/musical_interpretability_rank_<algorithm>.csv
    - musical_interpretability/<algorithm>/musical_interpretability_families_<algorithm>.csv

Si falta cualquier CSV, se imprime un warning y el agregado continúa con las
seeds disponibles.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Estilo gráfico compatible con classifier_curves.py
# ---------------------------------------------------------------------
_TABLEAU_COLORBLIND10 = [
    "#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1",
    "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF",
]


# ---------------------------------------------------------------------
# Incertidumbre en histogramas horizontales
# ---------------------------------------------------------------------
# Opciones:
#   - "line": formato clásico con barras de error horizontales.
#   - "band": formato recomendado para paper; muestra ±1 SD como una banda
#             semitransparente detrás de la barra de la media.
HORIZONTAL_UNCERTAINTY_STYLE = "band"
HORIZONTAL_UNCERTAINTY_BAND_ALPHA = 0.18
HORIZONTAL_BAR_ALPHA = 0.96


def _horizontal_mean_sd_barh(
    ax,
    y,
    means,
    stds,
    color=None,
    bar_height: float = 0.58,
    band_height: float = 0.78,
    add_legend: bool = True,
):
    """Dibuja barras horizontales de media ± SD con estilo configurable.

    La variable global HORIZONTAL_UNCERTAINTY_STYLE permite alternar entre
    el formato clásico de líneas de error y el formato de banda suave.
    """
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    y = np.asarray(y)
    means = pd.to_numeric(pd.Series(means), errors="coerce").to_numpy(float)
    stds = pd.to_numeric(pd.Series(stds), errors="coerce").fillna(0.0).to_numpy(float)
    stds = np.maximum(stds, 0.0)

    if color is None:
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", _TABLEAU_COLORBLIND10)
        color = colors[0] if colors else _TABLEAU_COLORBLIND10[0]

    mode = str(HORIZONTAL_UNCERTAINTY_STYLE).strip().lower()

    if mode == "line":
        bars = ax.barh(
            y,
            means,
            xerr=stds,
            capsize=3,
            height=bar_height,
            color=color,
            alpha=HORIZONTAL_BAR_ALPHA,
            error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
            zorder=2,
        )
        if add_legend:
            mean_proxy = mpatches.Patch(facecolor=color, alpha=HORIZONTAL_BAR_ALPHA, label="Mean")
            sd_proxy = mlines.Line2D([], [], color="black", linewidth=1.0, marker="|", markersize=8, label="±1 SD")
            ax.legend(handles=[mean_proxy, sd_proxy], frameon=False, loc="lower right")
        return bars

    if mode != "band":
        print(f"[WARN] HORIZONTAL_UNCERTAINTY_STYLE={HORIZONTAL_UNCERTAINTY_STYLE!r} no reconocido. Uso 'band'.")

    left = means - stds
    width = 2.0 * stds

    # Banda de incertidumbre: queda detrás de la barra para evitar el aspecto
    # agresivo de las líneas negras y mantener legible el ranking.
    ax.barh(
        y,
        width,
        left=left,
        height=band_height,
        color=color,
        alpha=HORIZONTAL_UNCERTAINTY_BAND_ALPHA,
        edgecolor="none",
        zorder=1,
    )
    bars = ax.barh(
        y,
        means,
        height=bar_height,
        color=color,
        alpha=HORIZONTAL_BAR_ALPHA,
        edgecolor="none",
        zorder=2,
    )

    if add_legend:
        mean_proxy = mpatches.Patch(facecolor=color, alpha=HORIZONTAL_BAR_ALPHA, label="Mean")
        sd_proxy = mpatches.Patch(facecolor=color, alpha=HORIZONTAL_UNCERTAINTY_BAND_ALPHA, label="±1 SD across seeds")
        ax.legend(handles=[mean_proxy, sd_proxy], frameon=False, loc="lower right")

    return bars


def _setup_paper_style(style: str = "tableau-colorblind10", font_size: int = 11) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("default")
    try:
        if style in plt.style.available:
            plt.style.use(style)
            colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", _TABLEAU_COLORBLIND10)
        else:
            colors = _TABLEAU_COLORBLIND10
    except Exception:
        colors = _TABLEAU_COLORBLIND10

    mpl.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size + 1,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=colors),
    })


def _save_figure(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), format="png")
    fig.savefig(str(out_base.with_suffix(".eps")), format="eps")


def _warn_missing(path: Path) -> None:
    print(f"[WARN] No existe el CSV requerido para agregación final: {path}")


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        _warn_missing(path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] No se pudo leer {path}: {repr(exc)}")
        return None


def _normalise_algorithm_name(name: str) -> str:
    return str(name).replace("_", " ").title()


# ---------------------------------------------------------------------
# 1) ROC curve: media de TPR sobre una rejilla común de FPR
# ---------------------------------------------------------------------
CLASS_SUFFIX_TO_DISPLAY = {"profano": "Profano", "religioso": "Religioso"}


def _class_suffix(class_suffix: str) -> str:
    class_suffix = str(class_suffix).lower().strip()
    if class_suffix not in CLASS_SUFFIX_TO_DISPLAY:
        raise ValueError("class_suffix debe ser 'profano' o 'religioso'.")
    return class_suffix


def _positive_display(class_suffix: str) -> str:
    return CLASS_SUFFIX_TO_DISPLAY[_class_suffix(class_suffix)]


def _collect_roc(base: Path, seeds: Iterable[int], algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
    class_suffix = _class_suffix(class_suffix)
    rows = []
    for seed in seeds:
        p = base / str(seed) / f"roc_curve_{algorithm_name}_{class_suffix}.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        required = {"fpr", "tpr"}
        if not required.issubset(df.columns):
            print(f"[WARN] CSV ROC inválido, faltan columnas {sorted(required.difference(df.columns))}: {p}")
            continue
        df = df.copy()
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        df["algorithm_name"] = algorithm_name
        df["positive_class_suffix"] = class_suffix
        df["positive_class"] = _positive_display(class_suffix)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _aggregate_roc(df_all: pd.DataFrame, out_dir: Path, algorithm_name: str, class_suffix: str = "profano", grid_size: int = 201) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    class_suffix = _class_suffix(class_suffix)
    positive_class = _positive_display(class_suffix)
    if df_all.empty:
        return pd.DataFrame()

    grid = np.linspace(0.0, 1.0, grid_size)
    interp_rows = []
    auc_rows = []
    for seed, g in df_all.groupby("seed"):
        g = g.sort_values("fpr")
        fpr = pd.to_numeric(g["fpr"], errors="coerce").to_numpy(dtype=float)
        tpr = pd.to_numeric(g["tpr"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(fpr) & np.isfinite(tpr)
        if mask.sum() < 2:
            continue
        fpr, tpr = fpr[mask], tpr[mask]
        uniq_fpr, idx = np.unique(fpr, return_index=True)
        uniq_tpr = tpr[idx]
        interp = np.interp(grid, uniq_fpr, uniq_tpr)
        interp[0] = 0.0
        interp[-1] = 1.0
        interp_rows.append(pd.DataFrame({"seed": int(seed), "fpr": grid, "tpr": interp}))
        if "roc_auc" in g.columns:
            auc_rows.append(float(pd.to_numeric(g["roc_auc"], errors="coerce").dropna().iloc[0]))

    if not interp_rows:
        return pd.DataFrame()

    df_interp = pd.concat(interp_rows, ignore_index=True)
    df_summary = (
        df_interp.groupby("fpr", as_index=False)
        .agg(tpr_mean=("tpr", "mean"), tpr_std=("tpr", "std"), n_seeds=("seed", "nunique"))
    )
    df_summary["tpr_std"] = df_summary["tpr_std"].fillna(0.0)
    df_summary["algorithm_name"] = algorithm_name
    df_summary["positive_class"] = positive_class
    df_summary["positive_class_suffix"] = class_suffix
    df_summary.to_csv(out_dir / f"final_roc_curve_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
    df_interp.to_csv(out_dir / f"final_roc_curve_by_seed_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")

    x = df_summary["fpr"].to_numpy(float)
    y = df_summary["tpr_mean"].to_numpy(float)
    s = df_summary["tpr_std"].to_numpy(float)
    auc_mean = float(np.nanmean(auc_rows)) if auc_rows else float("nan")
    auc_std = float(np.nanstd(auc_rows, ddof=1)) if len(auc_rows) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(x, y, linewidth=2.2, label=f"Mean AUC = {auc_mean:.3f} ± {auc_std:.3f}")
    ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.18, linewidth=0)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.45", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"False positive rate ({positive_class} positive)")
    ax.set_ylabel(f"True positive rate ({positive_class} positive)")
    ax.set_title(f"Mean ROC curve — {_normalise_algorithm_name(algorithm_name)} — {positive_class}")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="lower right", frameon=False)
    _save_figure(fig, out_dir / f"final_roc_curve_{algorithm_name}_{class_suffix}")
    plt.close(fig)
    return df_summary


# ---------------------------------------------------------------------
# 2) Precision-Recall: media de precisión sobre rejilla común de recall
# ---------------------------------------------------------------------
def _collect_pr(base: Path, seeds: Iterable[int], algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
    class_suffix = _class_suffix(class_suffix)
    rows = []
    for seed in seeds:
        p = base / str(seed) / f"precision_recall_curve_{algorithm_name}_{class_suffix}.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        required = {"recall", "precision"}
        if not required.issubset(df.columns):
            print(f"[WARN] CSV PR inválido, faltan columnas {sorted(required.difference(df.columns))}: {p}")
            continue
        df = df.copy()
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        df["algorithm_name"] = algorithm_name
        df["positive_class_suffix"] = class_suffix
        df["positive_class"] = _positive_display(class_suffix)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _aggregate_pr(df_all: pd.DataFrame, out_dir: Path, algorithm_name: str, class_suffix: str = "profano", grid_size: int = 201) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    class_suffix = _class_suffix(class_suffix)
    positive_class = _positive_display(class_suffix)
    if df_all.empty:
        return pd.DataFrame()

    grid = np.linspace(0.0, 1.0, grid_size)
    interp_rows = []
    ap_rows = []
    prevalence_rows = []
    for seed, g in df_all.groupby("seed"):
        recall = pd.to_numeric(g["recall"], errors="coerce").to_numpy(dtype=float)
        precision = pd.to_numeric(g["precision"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(recall) & np.isfinite(precision)
        if mask.sum() < 2:
            continue
        recall, precision = recall[mask], precision[mask]
        order = np.argsort(recall)
        recall, precision = recall[order], precision[order]
        uniq_recall, idx = np.unique(recall, return_index=True)
        uniq_precision = precision[idx]
        interp = np.interp(grid, uniq_recall, uniq_precision)
        interp_rows.append(pd.DataFrame({"seed": int(seed), "recall": grid, "precision": interp}))
        if "average_precision" in g.columns:
            ap_rows.append(float(pd.to_numeric(g["average_precision"], errors="coerce").dropna().iloc[0]))
        if "prevalence" in g.columns:
            prevalence_rows.append(float(pd.to_numeric(g["prevalence"], errors="coerce").dropna().iloc[0]))

    if not interp_rows:
        return pd.DataFrame()

    df_interp = pd.concat(interp_rows, ignore_index=True)
    df_summary = (
        df_interp.groupby("recall", as_index=False)
        .agg(precision_mean=("precision", "mean"), precision_std=("precision", "std"), n_seeds=("seed", "nunique"))
    )
    df_summary["precision_std"] = df_summary["precision_std"].fillna(0.0)
    df_summary["algorithm_name"] = algorithm_name
    df_summary["positive_class"] = positive_class
    df_summary["positive_class_suffix"] = class_suffix
    df_summary.to_csv(out_dir / f"final_precision_recall_curve_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
    df_interp.to_csv(out_dir / f"final_precision_recall_curve_by_seed_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")

    x = df_summary["recall"].to_numpy(float)
    y = df_summary["precision_mean"].to_numpy(float)
    s = df_summary["precision_std"].to_numpy(float)
    ap_mean = float(np.nanmean(ap_rows)) if ap_rows else float("nan")
    ap_std = float(np.nanstd(ap_rows, ddof=1)) if len(ap_rows) > 1 else 0.0
    prevalence = float(np.nanmean(prevalence_rows)) if prevalence_rows else float("nan")

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(x, y, linewidth=2.2, label=f"Mean AP = {ap_mean:.3f} ± {ap_std:.3f}")
    ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.18, linewidth=0)
    if np.isfinite(prevalence):
        ax.axhline(prevalence, linestyle="--", linewidth=1.0, color="0.45", label=f"Mean prevalence = {prevalence:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Recall ({positive_class})")
    ax.set_ylabel(f"Precision ({positive_class})")
    ax.set_title(f"Mean Precision–Recall — {_normalise_algorithm_name(algorithm_name)} — {positive_class}")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="lower left", frameon=False)
    _save_figure(fig, out_dir / f"final_precision_recall_curve_{algorithm_name}_{class_suffix}")
    plt.close(fig)
    return df_summary


# ---------------------------------------------------------------------
# 3) Threshold analysis: media ± std por threshold
# ---------------------------------------------------------------------
def _collect_threshold(base: Path, seeds: Iterable[int], algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
    class_suffix = _class_suffix(class_suffix)
    rows = []
    for seed in seeds:
        p = base / str(seed) / f"threshold_analysis_{algorithm_name}_{class_suffix}.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        required = {"threshold", "precision", "recall", "f1", "balanced_accuracy"}
        if not required.issubset(df.columns):
            print(f"[WARN] CSV threshold inválido, faltan columnas {sorted(required.difference(df.columns))}: {p}")
            continue
        df = df.copy()
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        df["algorithm_name"] = algorithm_name
        df["positive_class_suffix"] = class_suffix
        df["positive_class"] = _positive_display(class_suffix)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _aggregate_threshold(df_all: pd.DataFrame, out_dir: Path, algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
    import matplotlib.pyplot as plt

    class_suffix = _class_suffix(class_suffix)
    positive_class = _positive_display(class_suffix)
    if df_all.empty:
        return pd.DataFrame()

    metric_cols = ["precision", "recall", "f1", "balanced_accuracy"]
    work = df_all.copy()
    work["threshold"] = pd.to_numeric(work["threshold"], errors="coerce").round(6)
    for c in metric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_parts = []
    for c in metric_cols:
        g = work.groupby("threshold", as_index=False).agg(
            **{f"{c}_mean": (c, "mean"), f"{c}_std": (c, "std")},
            n_seeds=("seed", "nunique"),
        )
        g[f"{c}_std"] = g[f"{c}_std"].fillna(0.0)
        agg_parts.append(g)

    df_summary = agg_parts[0]
    for g in agg_parts[1:]:
        df_summary = df_summary.merge(g.drop(columns=["n_seeds"]), on="threshold", how="outer")
    df_summary["algorithm_name"] = algorithm_name
    df_summary["positive_class"] = positive_class
    df_summary["positive_class_suffix"] = class_suffix
    df_summary = df_summary.sort_values("threshold")
    df_summary.to_csv(out_dir / f"final_threshold_analysis_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
    work.to_csv(out_dir / f"final_threshold_analysis_by_seed_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = df_summary["threshold"].to_numpy(float)
    for c, label, lw in [
        ("precision", "Precision", 1.9),
        ("recall", "Recall", 1.9),
        ("f1", "F1", 2.2),
        ("balanced_accuracy", "Balanced accuracy", 2.2),
    ]:
        y = df_summary[f"{c}_mean"].to_numpy(float)
        s = df_summary[f"{c}_std"].to_numpy(float)
        ax.plot(x, y, linewidth=lw, label=label)
        ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.10, linewidth=0)

    best_f1 = df_summary.loc[df_summary["f1_mean"].idxmax()]
    best_bacc = df_summary.loc[df_summary["balanced_accuracy_mean"].idxmax()]
    ax.axvline(best_f1["threshold"], linestyle="--", linewidth=1.0, color="0.35")
    ax.axvline(best_bacc["threshold"], linestyle=":", linewidth=1.2, color="0.20")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"Decision threshold for {positive_class} class")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Mean threshold analysis — {_normalise_algorithm_name(algorithm_name)} — {positive_class}")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)
    _save_figure(fig, out_dir / f"final_threshold_analysis_{algorithm_name}_{class_suffix}")
    plt.close(fig)
    return df_summary


# ---------------------------------------------------------------------
# 4) Model comparison: tabla final media ± std por algoritmo y clase positiva
# ---------------------------------------------------------------------
def _aggregate_model_comparison(base: Path, out_dir: Path, seeds: Iterable[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        p = base / str(seed) / "model_comparison.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        df = df.copy()
        if "positive_class_suffix" not in df.columns:
            # Compatibilidad con CSV antiguos: eran métricas de Profano como clase positiva.
            df["positive_class_suffix"] = "profano"
            df["positive_class"] = "Profano"
            df["positive_label_id"] = 1
            df["score_column"] = "score_profano"
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(out_dir / "final_model_comparison_all_seeds.csv", index=False, encoding="utf-8")
    for class_suffix in sorted(df_all["positive_class_suffix"].dropna().unique()):
        df_all[df_all["positive_class_suffix"] == class_suffix].to_csv(
            out_dir / f"final_model_comparison_all_seeds_{class_suffix}.csv",
            index=False,
            encoding="utf-8",
        )

    metrics = [
        "accuracy", "balanced_accuracy", "precision", "recall", "f1",
        "precision_macro", "recall_macro", "f1_macro", "roc_auc",
    ]
    available = [m for m in metrics if m in df_all.columns]
    summary_rows = []
    group_cols = ["algorithm_name", "positive_class_suffix"]
    for (algorithm_name, class_suffix), g in df_all.groupby(group_cols):
        row = {
            "algorithm_name": algorithm_name,
            "positive_class_suffix": class_suffix,
            "positive_class": g["positive_class"].dropna().iloc[0] if "positive_class" in g.columns and len(g["positive_class"].dropna()) else _positive_display(class_suffix),
            "n_seeds": int(g["seed"].nunique()),
        }
        if "algorithm_id" in g.columns:
            row["algorithm_id"] = g["algorithm_id"].dropna().iloc[0] if len(g["algorithm_id"].dropna()) else np.nan
        for m in available:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    sort_col = "balanced_accuracy_mean" if "balanced_accuracy_mean" in df_summary.columns else df_summary.columns[0]
    df_summary = df_summary.sort_values(["positive_class_suffix", sort_col], ascending=[True, False]).reset_index(drop=True)
    df_summary.to_csv(out_dir / "final_model_comparison_mean_std.csv", index=False, encoding="utf-8")
    for class_suffix in sorted(df_summary["positive_class_suffix"].dropna().unique()):
        df_summary[df_summary["positive_class_suffix"] == class_suffix].to_csv(
            out_dir / f"final_model_comparison_mean_std_{class_suffix}.csv",
            index=False,
            encoding="utf-8",
        )
    return df_summary


# ---------------------------------------------------------------------
# 5) Feature importance: top estable por algoritmo
# ---------------------------------------------------------------------
def _aggregate_feature_importance(base: Path, out_dir: Path, seeds: Iterable[int], algorithm_name: str, top_n: int, style: str) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows = []
    for seed in seeds:
        p = base / str(seed) / f"feature_importance_{algorithm_name}.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        required = {"feature", "importance"}
        if not required.issubset(df.columns):
            print(f"[WARN] CSV feature importance inválido: {p}")
            continue
        df = df.copy()
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        df["algorithm_name"] = algorithm_name
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(out_dir / f"final_feature_importance_all_seeds_{algorithm_name}.csv", index=False, encoding="utf-8")
    df_all["importance"] = pd.to_numeric(df_all["importance"], errors="coerce")
    if "importance_std" in df_all.columns:
        df_all["importance_std"] = pd.to_numeric(df_all["importance_std"], errors="coerce")

    df_summary = (
        df_all.groupby(["algorithm_name", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std_across_seeds=("importance", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_summary["importance_std_across_seeds"] = df_summary["importance_std_across_seeds"].fillna(0.0)
    df_summary.to_csv(out_dir / f"final_feature_importance_mean_std_{algorithm_name}.csv", index=False, encoding="utf-8")

    dfp = df_summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(4.4, 0.33 * len(dfp) + 1.2)))
    _horizontal_mean_sd_barh(
        ax,
        np.arange(len(dfp)),
        dfp["importance_mean"],
        dfp["importance_std_across_seeds"],
    )
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in dfp["feature"]])
    ax.set_xlabel("Permutation importance, mean ± SD across seeds")
    ax.set_title(f"Mean feature importance — {_normalise_algorithm_name(algorithm_name)}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"final_feature_importance_{algorithm_name}")
    plt.close(fig)
    return df_summary


# ---------------------------------------------------------------------
# 6) Ablation e interpretability reutilizando sus CSV ya generados
# ---------------------------------------------------------------------
def _aggregate_ablation(base: Path, out_dir: Path, seeds: Iterable[int], algorithm_name: str, main_metric: str) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows = []
    for seed in seeds:
        p = base / str(seed) / "ablation_study" / algorithm_name / f"ablation_metrics_{algorithm_name}.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        df = df.copy()
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(out_dir / f"final_ablation_all_seeds_{algorithm_name}.csv", index=False, encoding="utf-8")

    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    available = [m for m in metrics if m in df_all.columns]
    summary = []
    for condition, g in df_all.groupby("condition"):
        row = {"algorithm_name": algorithm_name, "condition": condition, "n_seeds": int(g["seed"].nunique())}
        if "n_features" in g.columns:
            row["n_features_mean"] = float(pd.to_numeric(g["n_features"], errors="coerce").mean())
        for m in available:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        summary.append(row)
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(out_dir / f"final_ablation_mean_std_{algorithm_name}.csv", index=False, encoding="utf-8")

    if main_metric in available:
        order = [c for c in ["baseline_generic", "full_musical"] if c in set(df_summary["condition"])]
        dfp = df_summary.set_index("condition").loc[order].reset_index()
        labels = ["Without musical\nfeatures" if c == "baseline_generic" else "With musical\nfeatures" for c in dfp["condition"]]
        means = dfp[f"{main_metric}_mean"].to_numpy(float)
        stds = dfp[f"{main_metric}_std"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        bars = ax.bar(np.arange(len(dfp)), means, yerr=stds, capsize=5, width=0.55)
        for b, mean, std in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, mean + std + 0.012, f"{mean:.3f} ± {std:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(np.arange(len(dfp)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(main_metric.replace("_", " ").title())
        ax.set_ylim(0, min(1.10, max(1.0, float(np.nanmax(means + stds)) + 0.12)))
        ax.set_title(f"Mean ablation study — {_normalise_algorithm_name(algorithm_name)}")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(False, axis="x")
        _save_figure(fig, out_dir / f"final_ablation_{main_metric}_{algorithm_name}")
        plt.close(fig)
    return df_summary


def _aggregate_interpretability(base: Path, out_dir: Path, seeds: Iterable[int], algorithm_name: str, top_n: int) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows = []
    for seed in seeds:
        p = base / str(seed) / "musical_interpretability" / algorithm_name / f"musical_interpretability_rank_{algorithm_name}.csv"
        df = _read_csv_if_exists(p)
        if df is None:
            continue
        df = df.copy()
        df["seed"] = int(seed)
        df["source_csv"] = str(p)
        rows.append(df)
    if not rows:
        return pd.DataFrame()

    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(out_dir / f"final_musical_interpretability_all_seeds_{algorithm_name}.csv", index=False, encoding="utf-8")

    def mode_or_first(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return ""
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    rows_out = []
    for feature, g in df_all.groupby("feature"):
        row = {"algorithm_name": algorithm_name, "feature": feature, "n_seeds": int(g["seed"].nunique())}
        for c in ["discriminative_score", "permutation_importance", "cohens_d_profano_minus_religioso", "abs_cohens_d"]:
            if c in g.columns:
                vals = pd.to_numeric(g[c], errors="coerce")
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        for c in ["enriched_in", "feature_family", "hypothesis_alignment", "description"]:
            if c in g.columns:
                row[f"{c}_mode"] = mode_or_first(g[c])
        if "is_idiomatic_musicological" in g.columns:
            row["is_idiomatic_musicological"] = bool(g["is_idiomatic_musicological"].astype(bool).any())
        rows_out.append(row)

    sort_col = "discriminative_score_mean"
    df_summary = pd.DataFrame(rows_out).sort_values(sort_col, ascending=False).reset_index(drop=True)
    df_summary.to_csv(out_dir / f"final_musical_interpretability_mean_std_{algorithm_name}.csv", index=False, encoding="utf-8")

    dfp = df_summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4.4, 0.34 * len(dfp) + 1.2)))
    _horizontal_mean_sd_barh(
        ax,
        np.arange(len(dfp)),
        dfp["discriminative_score_mean"],
        dfp["discriminative_score_std"],
    )
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in dfp["feature"]])
    ax.set_xlabel("Discriminative score, mean ± SD across seeds")
    ax.set_title(f"Stable discriminative features — {_normalise_algorithm_name(algorithm_name)}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"final_musical_interpretability_{algorithm_name}")
    plt.close(fig)
    return df_summary



# ---------------------------------------------------------------------
# 7) Comparativa global de modelos: LaTeX, detección del mejor modelo y figuras
# ---------------------------------------------------------------------
def _latex_escape(text: object) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _metric_label(metric: str) -> str:
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "precision_macro": "Macro precision",
        "recall_macro": "Macro recall",
        "f1_macro": "Macro F1",
        "roc_auc": "ROC-AUC",
    }
    return labels.get(metric, str(metric).replace("_", " ").title())


def _metric_tex(metric: str) -> str:
    labels = {
        "accuracy": "Acc.",
        "balanced_accuracy": "BAcc.",
        "precision": "Prec.",
        "recall": "Rec.",
        "f1": "F1",
        "precision_macro": "Macro Prec.",
        "recall_macro": "Macro Rec.",
        "f1_macro": "Macro F1",
        "roc_auc": "ROC--AUC",
    }
    return labels.get(metric, _latex_escape(_metric_label(metric)))


def _mean_std_cell(row: pd.Series, metric: str, decimals: int = 3) -> str:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in row.index:
        return "--"
    mean = pd.to_numeric(pd.Series([row.get(mean_col)]), errors="coerce").iloc[0]
    std = pd.to_numeric(pd.Series([row.get(std_col, 0.0)]), errors="coerce").iloc[0]
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _ordered_metric_columns(df_summary: pd.DataFrame) -> List[str]:
    preferred = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]
    return [m for m in preferred if f"{m}_mean" in df_summary.columns]


def _comparison_ranking_frame(df_summary: pd.DataFrame, main_metric: str) -> pd.DataFrame:
    """Devuelve una fila por modelo para ranking global, preferiblemente Profano.

    Balanced accuracy y accuracy son invariantes a la elección de clase positiva.
    Para evitar duplicar filas en la detección del ganador, se toma la vista
    Profano cuando existe; si no, se usa la primera vista disponible por modelo.
    """
    if df_summary.empty:
        return pd.DataFrame()
    work = df_summary.copy()
    if "positive_class_suffix" in work.columns:
        work["_class_priority"] = np.where(work["positive_class_suffix"].astype(str) == "profano", 0, 1)
    else:
        work["_class_priority"] = 0
    mean_col = f"{main_metric}_mean"
    if mean_col not in work.columns:
        mean_col = "balanced_accuracy_mean" if "balanced_accuracy_mean" in work.columns else None
    if mean_col is not None:
        work = work.sort_values(["algorithm_name", "_class_priority", mean_col], ascending=[True, True, False])
    else:
        work = work.sort_values(["algorithm_name", "_class_priority"], ascending=[True, True])
    return work.groupby("algorithm_name", as_index=False).head(1).drop(columns=["_class_priority"], errors="ignore")


def _detect_best_model(df_summary: pd.DataFrame, main_metric: str) -> Dict[str, object]:
    rank_df = _comparison_ranking_frame(df_summary, main_metric)
    if rank_df.empty:
        return {}
    primary = f"{main_metric}_mean"
    if primary not in rank_df.columns:
        primary = "balanced_accuracy_mean" if "balanced_accuracy_mean" in rank_df.columns else None
    if primary is None:
        return {}
    tie_breakers = [primary]
    for c in ["roc_auc_mean", "f1_macro_mean", "accuracy_mean"]:
        if c in rank_df.columns and c not in tie_breakers:
            tie_breakers.append(c)
    rank_df = rank_df.sort_values(tie_breakers, ascending=[False] * len(tie_breakers)).reset_index(drop=True)
    best = rank_df.iloc[0]
    second = rank_df.iloc[1] if len(rank_df) > 1 else None
    primary_metric = primary[:-5] if primary.endswith("_mean") else primary
    margin = np.nan
    if second is not None:
        margin = float(best[primary] - second[primary])
    return {
        "algorithm_name": best["algorithm_name"],
        "display_name": _normalise_algorithm_name(best["algorithm_name"]),
        "primary_metric": primary_metric,
        "primary_metric_mean": float(best[primary]),
        "primary_metric_std": float(best.get(f"{primary_metric}_std", 0.0)) if f"{primary_metric}_std" in best.index else 0.0,
        "positive_class_suffix": best.get("positive_class_suffix", ""),
        "positive_class": best.get("positive_class", ""),
        "n_seeds": int(best.get("n_seeds", 0)) if pd.notna(best.get("n_seeds", np.nan)) else 0,
        "runner_up_algorithm_name": None if second is None else second["algorithm_name"],
        "runner_up_display_name": None if second is None else _normalise_algorithm_name(second["algorithm_name"]),
        "margin_over_runner_up": margin,
        "ranking": rank_df,
    }


def _write_model_comparison_latex_tables(df_summary: pd.DataFrame, out_dir: Path, main_metric: str) -> Dict[str, Path]:
    if df_summary.empty:
        return {}
    metrics = _ordered_metric_columns(df_summary)
    if not metrics:
        return {}

    def make_table(df: pd.DataFrame, caption: str, label: str, include_class: bool) -> str:
        align = "l" + ("l" if include_class else "") + "c" + "c" * len(metrics)
        header = ["Model"]
        if include_class:
            header.append("Positive class")
        header.append("Seeds")
        header.extend(_metric_tex(m) for m in metrics)
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{" + align + r"}",
            r"\hline",
            " & ".join(header) + r" \\",
            r"\hline",
        ]
        sort_col = f"{main_metric}_mean" if f"{main_metric}_mean" in df.columns else "balanced_accuracy_mean"
        sort_by = ["positive_class_suffix", sort_col] if include_class and "positive_class_suffix" in df.columns else [sort_col]
        asc = [True, False] if len(sort_by) == 2 else [False]
        for _, row in df.sort_values(sort_by, ascending=asc).iterrows():
            cells = [_latex_escape(_normalise_algorithm_name(row["algorithm_name"]))]
            if include_class:
                cells.append(_latex_escape(row.get("positive_class", row.get("positive_class_suffix", ""))))
            cells.append(str(int(row.get("n_seeds", 0))))
            cells.extend(_mean_std_cell(row, m) for m in metrics)
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\end{table*}",
            "",
        ])
        return "\n".join(lines)

    outputs: Dict[str, Path] = {}
    combined = make_table(
        df_summary,
        r"Multi-seed classifier comparison for the Profano/Religioso task. Values are reported as mean $\pm$ standard deviation across random splits.",
        "tab:teimus_classifier_comparison_all",
        include_class=True,
    )
    path = out_dir / "final_model_comparison_mean_std_table.tex"
    path.write_text(combined, encoding="utf-8")
    outputs["all"] = path

    if "positive_class_suffix" in df_summary.columns:
        for class_suffix in sorted(df_summary["positive_class_suffix"].dropna().unique()):
            df_class = df_summary[df_summary["positive_class_suffix"] == class_suffix].copy()
            display = _positive_display(class_suffix)
            tex = make_table(
                df_class,
                rf"Multi-seed classifier comparison with {display} treated as the positive class. Values are mean $\pm$ standard deviation across random splits.",
                f"tab:teimus_classifier_comparison_{class_suffix}",
                include_class=False,
            )
            p = out_dir / f"final_model_comparison_mean_std_table_{class_suffix}.tex"
            p.write_text(tex, encoding="utf-8")
            outputs[class_suffix] = p
    return outputs


def _write_best_model_latex_report(df_summary: pd.DataFrame, out_dir: Path, main_metric: str) -> Dict[str, Path]:
    best = _detect_best_model(df_summary, main_metric)
    if not best:
        return {}
    ranking = best["ranking"].copy()
    ranking_path = out_dir / "final_model_ranking_for_best_model_detection.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8")

    metric = str(best["primary_metric"])
    metric_label = _metric_label(metric)
    best_name = str(best["display_name"])
    best_mean = float(best["primary_metric_mean"])
    best_std = float(best["primary_metric_std"])
    n_seeds = int(best["n_seeds"])
    runner = best.get("runner_up_display_name")
    margin = best.get("margin_over_runner_up", np.nan)

    additional = []
    row = ranking.iloc[0]
    for m in ["accuracy", "f1_macro", "roc_auc", "precision", "recall", "f1"]:
        if f"{m}_mean" in row.index:
            additional.append(f"{_metric_label(m)} = {float(row[f'{m}_mean']):.3f} $\\pm$ {float(row.get(f'{m}_std', 0.0)):.3f}")
    additional_text = "; ".join(additional)

    if runner is not None and np.isfinite(margin):
        comparison_sentence = (
            f"It outperformed the second-ranked model, {runner}, by {margin:.3f} absolute {metric_label.lower()} points."
        )
    else:
        comparison_sentence = "No runner-up model was available in the aggregated comparison."

    paragraph = (
        f"Across {n_seeds} random stratified train/test splits, {best_name} obtained the best mean "
        f"{metric_label.lower()} ({best_mean:.3f} $\\pm$ {best_std:.3f}) among the evaluated classifiers. "
        f"{comparison_sentence} The same model obtained {additional_text}. "
        "These results indicate that the selected architecture provides the most stable aggregate discrimination "
        "between Profano and Religioso melodies under the current feature/token representation and evaluation protocol."
    )

    tex_lines = [
        r"\paragraph{Best-performing classifier.}",
        paragraph,
        "",
    ]
    tex_path = out_dir / "final_best_model_paper_comment.tex"
    txt_path = out_dir / "final_best_model_paper_comment.txt"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    txt_path.write_text(paragraph.replace("$\\pm$", "±"), encoding="utf-8")
    return {"tex": tex_path, "txt": txt_path, "ranking": ranking_path}


def _plot_model_metric_ranking(df_summary: pd.DataFrame, out_dir: Path, main_metric: str) -> None:
    import matplotlib.pyplot as plt

    rank_df = _comparison_ranking_frame(df_summary, main_metric)
    if rank_df.empty:
        return
    metric = main_metric if f"{main_metric}_mean" in rank_df.columns else "balanced_accuracy"
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    if mean_col not in rank_df.columns:
        return
    dfp = rank_df.sort_values(mean_col, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.36 * len(dfp) + 1.2)))
    _horizontal_mean_sd_barh(
        ax,
        np.arange(len(dfp)),
        dfp[mean_col],
        dfp[std_col] if std_col in dfp.columns else np.zeros(len(dfp)),
        add_legend=True,
    )
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels([_normalise_algorithm_name(x) for x in dfp["algorithm_name"]])
    ax.set_xlabel(f"{_metric_label(metric)}, mean ± SD across seeds")
    ax.set_title("Final model ranking")
    ax.set_xlim(0, min(1.05, max(1.0, float(pd.to_numeric(dfp[mean_col], errors="coerce").max()) + 0.08)))
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"final_model_comparison_ranking_{metric}")
    plt.close(fig)


def _plot_model_metric_grouped(df_summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rank_df = _comparison_ranking_frame(df_summary, "balanced_accuracy")
    metrics = [m for m in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"] if f"{m}_mean" in rank_df.columns]
    if rank_df.empty or not metrics:
        return
    rank_df = rank_df.sort_values("balanced_accuracy_mean" if "balanced_accuracy_mean" in rank_df.columns else f"{metrics[0]}_mean", ascending=False)
    labels = [_normalise_algorithm_name(x) for x in rank_df["algorithm_name"]]
    x = np.arange(len(rank_df))
    width = 0.80 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(7.2, 0.62 * len(labels) + 2.6), 4.4))
    for i, m in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2.0) * width
        vals = pd.to_numeric(rank_df[f"{m}_mean"], errors="coerce").to_numpy(float)
        ax.bar(x + offset, vals, width=width, label=_metric_label(m))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean score across seeds")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparative model performance")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, ncol=min(len(metrics), 4), loc="lower center", bbox_to_anchor=(0.5, -0.40))
    _save_figure(fig, out_dir / "final_model_comparison_grouped_metrics")
    plt.close(fig)


def _plot_model_metric_heatmap(df_summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rank_df = _comparison_ranking_frame(df_summary, "balanced_accuracy")
    metrics = [m for m in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"] if f"{m}_mean" in rank_df.columns]
    if rank_df.empty or not metrics:
        return
    sort_col = "balanced_accuracy_mean" if "balanced_accuracy_mean" in rank_df.columns else f"{metrics[0]}_mean"
    rank_df = rank_df.sort_values(sort_col, ascending=False)
    data = np.vstack([pd.to_numeric(rank_df[f"{m}_mean"], errors="coerce").to_numpy(float) for m in metrics]).T
    fig, ax = plt.subplots(figsize=(max(6.6, 0.60 * len(metrics) + 3.0), max(4.2, 0.36 * len(rank_df) + 1.6)))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([_metric_tex(m) for m in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rank_df)))
    ax.set_yticklabels([_normalise_algorithm_name(x) for x in rank_df["algorithm_name"]])
    ax.set_title("Mean metric matrix by model")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean score")
    _save_figure(fig, out_dir / "final_model_comparison_metric_heatmap")
    plt.close(fig)


def _plot_model_stability(df_summary: pd.DataFrame, out_dir: Path, main_metric: str) -> None:
    import matplotlib.pyplot as plt

    rank_df = _comparison_ranking_frame(df_summary, main_metric)
    metric = main_metric if f"{main_metric}_mean" in rank_df.columns else "balanced_accuracy"
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    if rank_df.empty or mean_col not in rank_df.columns:
        return
    x = pd.to_numeric(rank_df[mean_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(rank_df[std_col], errors="coerce").fillna(0.0).to_numpy(float) if std_col in rank_df.columns else np.zeros(len(rank_df))
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.scatter(x, y, s=42)
    for xi, yi, name in zip(x, y, rank_df["algorithm_name"]):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(_normalise_algorithm_name(name), (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel(f"Mean {_metric_label(metric)}")
    ax.set_ylabel("SD across seeds")
    ax.set_title("Performance vs. split stability")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    _save_figure(fig, out_dir / f"final_model_comparison_stability_{metric}")
    plt.close(fig)


def _generate_model_comparison_outputs(df_summary: pd.DataFrame, out_dir: Path, main_metric: str) -> Dict[str, object]:
    if df_summary.empty:
        print("[WARN] No hay tabla final de comparación de modelos; se omiten LaTeX, informe y figuras comparativas.")
        return {}
    outputs: Dict[str, object] = {}
    outputs["latex_tables"] = _write_model_comparison_latex_tables(df_summary, out_dir, main_metric)
    outputs["best_model_report"] = _write_best_model_latex_report(df_summary, out_dir, main_metric)
    _plot_model_metric_ranking(df_summary, out_dir, main_metric)
    _plot_model_metric_grouped(df_summary, out_dir)
    _plot_model_metric_heatmap(df_summary, out_dir)
    _plot_model_stability(df_summary, out_dir, main_metric)
    return outputs

# ---------------------------------------------------------------------
# 7) API pública
# ---------------------------------------------------------------------
def aggregate_final_results(
    base_results_dir: str | os.PathLike,
    seeds: Iterable[int],
    algorithm_names: Iterable[str],
    style: str = "tableau-colorblind10",
    main_metric: str = "balanced_accuracy",
    top_n: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Genera el último paso del pipeline: resultados finales multi-seed.

    Todo se guarda en <base_results_dir>/final_results. La función solo lee CSVs
    generados previamente; si falta alguno, avisa y continúa.
    """
    base = Path(base_results_dir)
    out_dir = base / "final_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_paper_style(style)

    outputs: Dict[str, pd.DataFrame] = {}
    outputs["model_comparison"] = _aggregate_model_comparison(base, out_dir, seeds)
    outputs["model_comparison_outputs"] = _generate_model_comparison_outputs(
        outputs["model_comparison"], out_dir, main_metric=main_metric
    )

    for algorithm_name in algorithm_names:
        print("\n==============================")
        print("FINAL MULTI-SEED RESULTS")
        print("==============================")
        print("Modelo:", algorithm_name)

        for class_suffix in ("profano", "religioso"):
            print("Clase positiva:", _positive_display(class_suffix))
            roc_all = _collect_roc(base, seeds, algorithm_name, class_suffix=class_suffix)
            outputs[f"roc_{algorithm_name}_{class_suffix}"] = _aggregate_roc(
                roc_all, out_dir, algorithm_name, class_suffix=class_suffix
            )

            pr_all = _collect_pr(base, seeds, algorithm_name, class_suffix=class_suffix)
            outputs[f"precision_recall_{algorithm_name}_{class_suffix}"] = _aggregate_pr(
                pr_all, out_dir, algorithm_name, class_suffix=class_suffix
            )

            thr_all = _collect_threshold(base, seeds, algorithm_name, class_suffix=class_suffix)
            outputs[f"threshold_{algorithm_name}_{class_suffix}"] = _aggregate_threshold(
                thr_all, out_dir, algorithm_name, class_suffix=class_suffix
            )

        outputs[f"feature_importance_{algorithm_name}"] = _aggregate_feature_importance(
            base, out_dir, seeds, algorithm_name, top_n=top_n, style=style
        )

        outputs[f"ablation_{algorithm_name}"] = _aggregate_ablation(
            base, out_dir, seeds, algorithm_name, main_metric=main_metric
        )
        outputs[f"interpretability_{algorithm_name}"] = _aggregate_interpretability(
            base, out_dir, seeds, algorithm_name, top_n=top_n
        )

    _write_index(out_dir, algorithm_names, seeds)
    print("[OK] Resultados finales guardados en:", out_dir)
    return outputs


def _write_index(out_dir: Path, algorithm_names: Iterable[str], seeds: Iterable[int]) -> None:
    lines = [
        "# Final multi-seed results",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Algorithms: {', '.join(str(a) for a in algorithm_names)}",
        "",
        "This folder contains the final aggregated CSV and PNG/EPS figures computed only from previously saved per-seed CSV files.",
        "Missing files are reported as warnings and do not interrupt the aggregation.",
        "",
        "Main outputs per algorithm:",
        "- final_roc_curve_<algorithm>_profano.csv/png/eps",
        "- final_roc_curve_<algorithm>_religioso.csv/png/eps",
        "- final_precision_recall_curve_<algorithm>_profano.csv/png/eps",
        "- final_precision_recall_curve_<algorithm>_religioso.csv/png/eps",
        "- final_threshold_analysis_<algorithm>_profano.csv/png/eps",
        "- final_threshold_analysis_<algorithm>_religioso.csv/png/eps",
        "- final_feature_importance_<algorithm>.csv/png/eps",
        "- final_ablation_<metric>_<algorithm>.csv/png/eps",
        "- final_musical_interpretability_<algorithm>.csv/png/eps",
        "",
        "Global model-comparison outputs:",
        "- final_model_comparison_mean_std_table.tex",
        "- final_model_comparison_mean_std_table_profano.tex",
        "- final_model_comparison_mean_std_table_religioso.tex",
        "- final_best_model_paper_comment.tex",
        "- final_best_model_paper_comment.txt",
        "- final_model_ranking_for_best_model_detection.csv",
        "- final_model_comparison_ranking_<metric>.png/eps",
        "- final_model_comparison_grouped_metrics.png/eps",
        "- final_model_comparison_metric_heatmap.png/eps",
        "- final_model_comparison_stability_<metric>.png/eps",
        "",
    ]
    (out_dir / "README_final_results.md").write_text("\n".join(lines), encoding="utf-8")







#antes del cambio de la comparativa final. funciona OK
# # -*- coding: utf-8 -*-
# """
# final_results.py
#
# Agregación final multi-seed para el pipeline Profano/Religioso de TEIMUS.
#
# Este módulo NO recalcula curvas ni predicciones. Lee los CSV generados en cada
# experimento/seed y produce figuras finales con media ± desviación estándar en:
#
#     <base_results_dir>/final_results/
#
# Entradas esperadas por seed y algoritmo:
#     - roc_curve_<algorithm>.csv
#     - precision_recall_curve_<algorithm>.csv
#     - threshold_analysis_<algorithm>.csv
#     - model_comparison.csv
#     - feature_importance_<algorithm>.csv
#     - ablation_study/<algorithm>/ablation_metrics_<algorithm>.csv
#     - musical_interpretability/<algorithm>/musical_interpretability_rank_<algorithm>.csv
#     - musical_interpretability/<algorithm>/musical_interpretability_families_<algorithm>.csv
#
# Si falta cualquier CSV, se imprime un warning y el agregado continúa con las
# seeds disponibles.
# """
#
# from __future__ import annotations
#
# import os
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple
#
# import numpy as np
# import pandas as pd
#
#
# # ---------------------------------------------------------------------
# # Estilo gráfico compatible con classifier_curves.py
# # ---------------------------------------------------------------------
# _TABLEAU_COLORBLIND10 = [
#     "#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1",
#     "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF",
# ]
#
#
# # ---------------------------------------------------------------------
# # Incertidumbre en histogramas horizontales
# # ---------------------------------------------------------------------
# # Opciones:
# #   - "line": formato clásico con barras de error horizontales.
# #   - "band": formato recomendado para paper; muestra ±1 SD como una banda
# #             semitransparente detrás de la barra de la media.
# HORIZONTAL_UNCERTAINTY_STYLE = "band"
# HORIZONTAL_UNCERTAINTY_BAND_ALPHA = 0.18
# HORIZONTAL_BAR_ALPHA = 0.96
#
#
# def _horizontal_mean_sd_barh(
#     ax,
#     y,
#     means,
#     stds,
#     color=None,
#     bar_height: float = 0.58,
#     band_height: float = 0.78,
#     add_legend: bool = True,
# ):
#     """Dibuja barras horizontales de media ± SD con estilo configurable.
#
#     La variable global HORIZONTAL_UNCERTAINTY_STYLE permite alternar entre
#     el formato clásico de líneas de error y el formato de banda suave.
#     """
#     import matplotlib as mpl
#     import matplotlib.lines as mlines
#     import matplotlib.patches as mpatches
#
#     y = np.asarray(y)
#     means = pd.to_numeric(pd.Series(means), errors="coerce").to_numpy(float)
#     stds = pd.to_numeric(pd.Series(stds), errors="coerce").fillna(0.0).to_numpy(float)
#     stds = np.maximum(stds, 0.0)
#
#     if color is None:
#         colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", _TABLEAU_COLORBLIND10)
#         color = colors[0] if colors else _TABLEAU_COLORBLIND10[0]
#
#     mode = str(HORIZONTAL_UNCERTAINTY_STYLE).strip().lower()
#
#     if mode == "line":
#         bars = ax.barh(
#             y,
#             means,
#             xerr=stds,
#             capsize=3,
#             height=bar_height,
#             color=color,
#             alpha=HORIZONTAL_BAR_ALPHA,
#             error_kw={"ecolor": "black", "elinewidth": 1.0, "capthick": 1.0},
#             zorder=2,
#         )
#         if add_legend:
#             mean_proxy = mpatches.Patch(facecolor=color, alpha=HORIZONTAL_BAR_ALPHA, label="Mean")
#             sd_proxy = mlines.Line2D([], [], color="black", linewidth=1.0, marker="|", markersize=8, label="±1 SD")
#             ax.legend(handles=[mean_proxy, sd_proxy], frameon=False, loc="lower right")
#         return bars
#
#     if mode != "band":
#         print(f"[WARN] HORIZONTAL_UNCERTAINTY_STYLE={HORIZONTAL_UNCERTAINTY_STYLE!r} no reconocido. Uso 'band'.")
#
#     left = means - stds
#     width = 2.0 * stds
#
#     # Banda de incertidumbre: queda detrás de la barra para evitar el aspecto
#     # agresivo de las líneas negras y mantener legible el ranking.
#     ax.barh(
#         y,
#         width,
#         left=left,
#         height=band_height,
#         color=color,
#         alpha=HORIZONTAL_UNCERTAINTY_BAND_ALPHA,
#         edgecolor="none",
#         zorder=1,
#     )
#     bars = ax.barh(
#         y,
#         means,
#         height=bar_height,
#         color=color,
#         alpha=HORIZONTAL_BAR_ALPHA,
#         edgecolor="none",
#         zorder=2,
#     )
#
#     if add_legend:
#         mean_proxy = mpatches.Patch(facecolor=color, alpha=HORIZONTAL_BAR_ALPHA, label="Mean")
#         sd_proxy = mpatches.Patch(facecolor=color, alpha=HORIZONTAL_UNCERTAINTY_BAND_ALPHA, label="±1 SD across seeds")
#         ax.legend(handles=[mean_proxy, sd_proxy], frameon=False, loc="lower right")
#
#     return bars
#
#
# def _setup_paper_style(style: str = "tableau-colorblind10", font_size: int = 11) -> None:
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#
#     plt.style.use("default")
#     try:
#         if style in plt.style.available:
#             plt.style.use(style)
#             colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", _TABLEAU_COLORBLIND10)
#         else:
#             colors = _TABLEAU_COLORBLIND10
#     except Exception:
#         colors = _TABLEAU_COLORBLIND10
#
#     mpl.rcParams.update({
#         "font.family": "DejaVu Serif",
#         "font.size": font_size,
#         "axes.titlesize": font_size + 1,
#         "axes.labelsize": font_size + 1,
#         "legend.fontsize": font_size,
#         "xtick.labelsize": font_size,
#         "ytick.labelsize": font_size,
#         "axes.linewidth": 0.9,
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "figure.dpi": 400,
#         "savefig.dpi": 400,
#         "savefig.bbox": "tight",
#         "savefig.pad_inches": 0.04,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#         "axes.prop_cycle": mpl.cycler(color=colors),
#     })
#
#
# def _save_figure(fig, out_base: Path) -> None:
#     out_base.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(str(out_base.with_suffix(".png")), format="png")
#     fig.savefig(str(out_base.with_suffix(".eps")), format="eps")
#
#
# def _warn_missing(path: Path) -> None:
#     print(f"[WARN] No existe el CSV requerido para agregación final: {path}")
#
#
# def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
#     if not path.exists():
#         _warn_missing(path)
#         return None
#     try:
#         return pd.read_csv(path)
#     except Exception as exc:
#         print(f"[WARN] No se pudo leer {path}: {repr(exc)}")
#         return None
#
#
# def _normalise_algorithm_name(name: str) -> str:
#     return str(name).replace("_", " ").title()
#
#
# # ---------------------------------------------------------------------
# # 1) ROC curve: media de TPR sobre una rejilla común de FPR
# # ---------------------------------------------------------------------
# CLASS_SUFFIX_TO_DISPLAY = {"profano": "Profano", "religioso": "Religioso"}
#
#
# def _class_suffix(class_suffix: str) -> str:
#     class_suffix = str(class_suffix).lower().strip()
#     if class_suffix not in CLASS_SUFFIX_TO_DISPLAY:
#         raise ValueError("class_suffix debe ser 'profano' o 'religioso'.")
#     return class_suffix
#
#
# def _positive_display(class_suffix: str) -> str:
#     return CLASS_SUFFIX_TO_DISPLAY[_class_suffix(class_suffix)]
#
#
# def _collect_roc(base: Path, seeds: Iterable[int], algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
#     class_suffix = _class_suffix(class_suffix)
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / f"roc_curve_{algorithm_name}_{class_suffix}.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         required = {"fpr", "tpr"}
#         if not required.issubset(df.columns):
#             print(f"[WARN] CSV ROC inválido, faltan columnas {sorted(required.difference(df.columns))}: {p}")
#             continue
#         df = df.copy()
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         df["algorithm_name"] = algorithm_name
#         df["positive_class_suffix"] = class_suffix
#         df["positive_class"] = _positive_display(class_suffix)
#         rows.append(df)
#     return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
#
#
# def _aggregate_roc(df_all: pd.DataFrame, out_dir: Path, algorithm_name: str, class_suffix: str = "profano", grid_size: int = 201) -> pd.DataFrame:
#     import matplotlib.pyplot as plt
#
#     class_suffix = _class_suffix(class_suffix)
#     positive_class = _positive_display(class_suffix)
#     if df_all.empty:
#         return pd.DataFrame()
#
#     grid = np.linspace(0.0, 1.0, grid_size)
#     interp_rows = []
#     auc_rows = []
#     for seed, g in df_all.groupby("seed"):
#         g = g.sort_values("fpr")
#         fpr = pd.to_numeric(g["fpr"], errors="coerce").to_numpy(dtype=float)
#         tpr = pd.to_numeric(g["tpr"], errors="coerce").to_numpy(dtype=float)
#         mask = np.isfinite(fpr) & np.isfinite(tpr)
#         if mask.sum() < 2:
#             continue
#         fpr, tpr = fpr[mask], tpr[mask]
#         uniq_fpr, idx = np.unique(fpr, return_index=True)
#         uniq_tpr = tpr[idx]
#         interp = np.interp(grid, uniq_fpr, uniq_tpr)
#         interp[0] = 0.0
#         interp[-1] = 1.0
#         interp_rows.append(pd.DataFrame({"seed": int(seed), "fpr": grid, "tpr": interp}))
#         if "roc_auc" in g.columns:
#             auc_rows.append(float(pd.to_numeric(g["roc_auc"], errors="coerce").dropna().iloc[0]))
#
#     if not interp_rows:
#         return pd.DataFrame()
#
#     df_interp = pd.concat(interp_rows, ignore_index=True)
#     df_summary = (
#         df_interp.groupby("fpr", as_index=False)
#         .agg(tpr_mean=("tpr", "mean"), tpr_std=("tpr", "std"), n_seeds=("seed", "nunique"))
#     )
#     df_summary["tpr_std"] = df_summary["tpr_std"].fillna(0.0)
#     df_summary["algorithm_name"] = algorithm_name
#     df_summary["positive_class"] = positive_class
#     df_summary["positive_class_suffix"] = class_suffix
#     df_summary.to_csv(out_dir / f"final_roc_curve_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
#     df_interp.to_csv(out_dir / f"final_roc_curve_by_seed_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
#
#     x = df_summary["fpr"].to_numpy(float)
#     y = df_summary["tpr_mean"].to_numpy(float)
#     s = df_summary["tpr_std"].to_numpy(float)
#     auc_mean = float(np.nanmean(auc_rows)) if auc_rows else float("nan")
#     auc_std = float(np.nanstd(auc_rows, ddof=1)) if len(auc_rows) > 1 else 0.0
#
#     fig, ax = plt.subplots(figsize=(4.8, 4.0))
#     ax.plot(x, y, linewidth=2.2, label=f"Mean AUC = {auc_mean:.3f} ± {auc_std:.3f}")
#     ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.18, linewidth=0)
#     ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.45", label="Chance")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(f"False positive rate ({positive_class} positive)")
#     ax.set_ylabel(f"True positive rate ({positive_class} positive)")
#     ax.set_title(f"Mean ROC curve — {_normalise_algorithm_name(algorithm_name)} — {positive_class}")
#     ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
#     ax.legend(loc="lower right", frameon=False)
#     _save_figure(fig, out_dir / f"final_roc_curve_{algorithm_name}_{class_suffix}")
#     plt.close(fig)
#     return df_summary
#
#
# # ---------------------------------------------------------------------
# # 2) Precision-Recall: media de precisión sobre rejilla común de recall
# # ---------------------------------------------------------------------
# def _collect_pr(base: Path, seeds: Iterable[int], algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
#     class_suffix = _class_suffix(class_suffix)
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / f"precision_recall_curve_{algorithm_name}_{class_suffix}.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         required = {"recall", "precision"}
#         if not required.issubset(df.columns):
#             print(f"[WARN] CSV PR inválido, faltan columnas {sorted(required.difference(df.columns))}: {p}")
#             continue
#         df = df.copy()
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         df["algorithm_name"] = algorithm_name
#         df["positive_class_suffix"] = class_suffix
#         df["positive_class"] = _positive_display(class_suffix)
#         rows.append(df)
#     return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
#
#
# def _aggregate_pr(df_all: pd.DataFrame, out_dir: Path, algorithm_name: str, class_suffix: str = "profano", grid_size: int = 201) -> pd.DataFrame:
#     import matplotlib.pyplot as plt
#
#     class_suffix = _class_suffix(class_suffix)
#     positive_class = _positive_display(class_suffix)
#     if df_all.empty:
#         return pd.DataFrame()
#
#     grid = np.linspace(0.0, 1.0, grid_size)
#     interp_rows = []
#     ap_rows = []
#     prevalence_rows = []
#     for seed, g in df_all.groupby("seed"):
#         recall = pd.to_numeric(g["recall"], errors="coerce").to_numpy(dtype=float)
#         precision = pd.to_numeric(g["precision"], errors="coerce").to_numpy(dtype=float)
#         mask = np.isfinite(recall) & np.isfinite(precision)
#         if mask.sum() < 2:
#             continue
#         recall, precision = recall[mask], precision[mask]
#         order = np.argsort(recall)
#         recall, precision = recall[order], precision[order]
#         uniq_recall, idx = np.unique(recall, return_index=True)
#         uniq_precision = precision[idx]
#         interp = np.interp(grid, uniq_recall, uniq_precision)
#         interp_rows.append(pd.DataFrame({"seed": int(seed), "recall": grid, "precision": interp}))
#         if "average_precision" in g.columns:
#             ap_rows.append(float(pd.to_numeric(g["average_precision"], errors="coerce").dropna().iloc[0]))
#         if "prevalence" in g.columns:
#             prevalence_rows.append(float(pd.to_numeric(g["prevalence"], errors="coerce").dropna().iloc[0]))
#
#     if not interp_rows:
#         return pd.DataFrame()
#
#     df_interp = pd.concat(interp_rows, ignore_index=True)
#     df_summary = (
#         df_interp.groupby("recall", as_index=False)
#         .agg(precision_mean=("precision", "mean"), precision_std=("precision", "std"), n_seeds=("seed", "nunique"))
#     )
#     df_summary["precision_std"] = df_summary["precision_std"].fillna(0.0)
#     df_summary["algorithm_name"] = algorithm_name
#     df_summary["positive_class"] = positive_class
#     df_summary["positive_class_suffix"] = class_suffix
#     df_summary.to_csv(out_dir / f"final_precision_recall_curve_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
#     df_interp.to_csv(out_dir / f"final_precision_recall_curve_by_seed_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
#
#     x = df_summary["recall"].to_numpy(float)
#     y = df_summary["precision_mean"].to_numpy(float)
#     s = df_summary["precision_std"].to_numpy(float)
#     ap_mean = float(np.nanmean(ap_rows)) if ap_rows else float("nan")
#     ap_std = float(np.nanstd(ap_rows, ddof=1)) if len(ap_rows) > 1 else 0.0
#     prevalence = float(np.nanmean(prevalence_rows)) if prevalence_rows else float("nan")
#
#     fig, ax = plt.subplots(figsize=(4.8, 4.0))
#     ax.plot(x, y, linewidth=2.2, label=f"Mean AP = {ap_mean:.3f} ± {ap_std:.3f}")
#     ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.18, linewidth=0)
#     if np.isfinite(prevalence):
#         ax.axhline(prevalence, linestyle="--", linewidth=1.0, color="0.45", label=f"Mean prevalence = {prevalence:.3f}")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(f"Recall ({positive_class})")
#     ax.set_ylabel(f"Precision ({positive_class})")
#     ax.set_title(f"Mean Precision–Recall — {_normalise_algorithm_name(algorithm_name)} — {positive_class}")
#     ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
#     ax.legend(loc="lower left", frameon=False)
#     _save_figure(fig, out_dir / f"final_precision_recall_curve_{algorithm_name}_{class_suffix}")
#     plt.close(fig)
#     return df_summary
#
#
# # ---------------------------------------------------------------------
# # 3) Threshold analysis: media ± std por threshold
# # ---------------------------------------------------------------------
# def _collect_threshold(base: Path, seeds: Iterable[int], algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
#     class_suffix = _class_suffix(class_suffix)
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / f"threshold_analysis_{algorithm_name}_{class_suffix}.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         required = {"threshold", "precision", "recall", "f1", "balanced_accuracy"}
#         if not required.issubset(df.columns):
#             print(f"[WARN] CSV threshold inválido, faltan columnas {sorted(required.difference(df.columns))}: {p}")
#             continue
#         df = df.copy()
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         df["algorithm_name"] = algorithm_name
#         df["positive_class_suffix"] = class_suffix
#         df["positive_class"] = _positive_display(class_suffix)
#         rows.append(df)
#     return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
#
#
# def _aggregate_threshold(df_all: pd.DataFrame, out_dir: Path, algorithm_name: str, class_suffix: str = "profano") -> pd.DataFrame:
#     import matplotlib.pyplot as plt
#
#     class_suffix = _class_suffix(class_suffix)
#     positive_class = _positive_display(class_suffix)
#     if df_all.empty:
#         return pd.DataFrame()
#
#     metric_cols = ["precision", "recall", "f1", "balanced_accuracy"]
#     work = df_all.copy()
#     work["threshold"] = pd.to_numeric(work["threshold"], errors="coerce").round(6)
#     for c in metric_cols:
#         work[c] = pd.to_numeric(work[c], errors="coerce")
#
#     agg_parts = []
#     for c in metric_cols:
#         g = work.groupby("threshold", as_index=False).agg(
#             **{f"{c}_mean": (c, "mean"), f"{c}_std": (c, "std")},
#             n_seeds=("seed", "nunique"),
#         )
#         g[f"{c}_std"] = g[f"{c}_std"].fillna(0.0)
#         agg_parts.append(g)
#
#     df_summary = agg_parts[0]
#     for g in agg_parts[1:]:
#         df_summary = df_summary.merge(g.drop(columns=["n_seeds"]), on="threshold", how="outer")
#     df_summary["algorithm_name"] = algorithm_name
#     df_summary["positive_class"] = positive_class
#     df_summary["positive_class_suffix"] = class_suffix
#     df_summary = df_summary.sort_values("threshold")
#     df_summary.to_csv(out_dir / f"final_threshold_analysis_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
#     work.to_csv(out_dir / f"final_threshold_analysis_by_seed_{algorithm_name}_{class_suffix}.csv", index=False, encoding="utf-8")
#
#     fig, ax = plt.subplots(figsize=(5.6, 4.0))
#     x = df_summary["threshold"].to_numpy(float)
#     for c, label, lw in [
#         ("precision", "Precision", 1.9),
#         ("recall", "Recall", 1.9),
#         ("f1", "F1", 2.2),
#         ("balanced_accuracy", "Balanced accuracy", 2.2),
#     ]:
#         y = df_summary[f"{c}_mean"].to_numpy(float)
#         s = df_summary[f"{c}_std"].to_numpy(float)
#         ax.plot(x, y, linewidth=lw, label=label)
#         ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.10, linewidth=0)
#
#     best_f1 = df_summary.loc[df_summary["f1_mean"].idxmax()]
#     best_bacc = df_summary.loc[df_summary["balanced_accuracy_mean"].idxmax()]
#     ax.axvline(best_f1["threshold"], linestyle="--", linewidth=1.0, color="0.35")
#     ax.axvline(best_bacc["threshold"], linestyle=":", linewidth=1.2, color="0.20")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(f"Decision threshold for {positive_class} class")
#     ax.set_ylabel("Metric value")
#     ax.set_title(f"Mean threshold analysis — {_normalise_algorithm_name(algorithm_name)} — {positive_class}")
#     ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
#     ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)
#     _save_figure(fig, out_dir / f"final_threshold_analysis_{algorithm_name}_{class_suffix}")
#     plt.close(fig)
#     return df_summary
#
#
# # ---------------------------------------------------------------------
# # 4) Model comparison: tabla final media ± std por algoritmo y clase positiva
# # ---------------------------------------------------------------------
# def _aggregate_model_comparison(base: Path, out_dir: Path, seeds: Iterable[int]) -> pd.DataFrame:
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / "model_comparison.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         df = df.copy()
#         if "positive_class_suffix" not in df.columns:
#             # Compatibilidad con CSV antiguos: eran métricas de Profano como clase positiva.
#             df["positive_class_suffix"] = "profano"
#             df["positive_class"] = "Profano"
#             df["positive_label_id"] = 1
#             df["score_column"] = "score_profano"
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         rows.append(df)
#     if not rows:
#         return pd.DataFrame()
#
#     df_all = pd.concat(rows, ignore_index=True)
#     df_all.to_csv(out_dir / "final_model_comparison_all_seeds.csv", index=False, encoding="utf-8")
#     for class_suffix in sorted(df_all["positive_class_suffix"].dropna().unique()):
#         df_all[df_all["positive_class_suffix"] == class_suffix].to_csv(
#             out_dir / f"final_model_comparison_all_seeds_{class_suffix}.csv",
#             index=False,
#             encoding="utf-8",
#         )
#
#     metrics = [
#         "accuracy", "balanced_accuracy", "precision", "recall", "f1",
#         "precision_macro", "recall_macro", "f1_macro", "roc_auc",
#     ]
#     available = [m for m in metrics if m in df_all.columns]
#     summary_rows = []
#     group_cols = ["algorithm_name", "positive_class_suffix"]
#     for (algorithm_name, class_suffix), g in df_all.groupby(group_cols):
#         row = {
#             "algorithm_name": algorithm_name,
#             "positive_class_suffix": class_suffix,
#             "positive_class": g["positive_class"].dropna().iloc[0] if "positive_class" in g.columns and len(g["positive_class"].dropna()) else _positive_display(class_suffix),
#             "n_seeds": int(g["seed"].nunique()),
#         }
#         if "algorithm_id" in g.columns:
#             row["algorithm_id"] = g["algorithm_id"].dropna().iloc[0] if len(g["algorithm_id"].dropna()) else np.nan
#         for m in available:
#             vals = pd.to_numeric(g[m], errors="coerce")
#             row[f"{m}_mean"] = float(vals.mean())
#             row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
#         summary_rows.append(row)
#
#     df_summary = pd.DataFrame(summary_rows)
#     sort_col = "balanced_accuracy_mean" if "balanced_accuracy_mean" in df_summary.columns else df_summary.columns[0]
#     df_summary = df_summary.sort_values(["positive_class_suffix", sort_col], ascending=[True, False]).reset_index(drop=True)
#     df_summary.to_csv(out_dir / "final_model_comparison_mean_std.csv", index=False, encoding="utf-8")
#     for class_suffix in sorted(df_summary["positive_class_suffix"].dropna().unique()):
#         df_summary[df_summary["positive_class_suffix"] == class_suffix].to_csv(
#             out_dir / f"final_model_comparison_mean_std_{class_suffix}.csv",
#             index=False,
#             encoding="utf-8",
#         )
#     return df_summary
#
#
# # ---------------------------------------------------------------------
# # 5) Feature importance: top estable por algoritmo
# # ---------------------------------------------------------------------
# def _aggregate_feature_importance(base: Path, out_dir: Path, seeds: Iterable[int], algorithm_name: str, top_n: int, style: str) -> pd.DataFrame:
#     import matplotlib.pyplot as plt
#
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / f"feature_importance_{algorithm_name}.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         required = {"feature", "importance"}
#         if not required.issubset(df.columns):
#             print(f"[WARN] CSV feature importance inválido: {p}")
#             continue
#         df = df.copy()
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         df["algorithm_name"] = algorithm_name
#         rows.append(df)
#     if not rows:
#         return pd.DataFrame()
#
#     df_all = pd.concat(rows, ignore_index=True)
#     df_all.to_csv(out_dir / f"final_feature_importance_all_seeds_{algorithm_name}.csv", index=False, encoding="utf-8")
#     df_all["importance"] = pd.to_numeric(df_all["importance"], errors="coerce")
#     if "importance_std" in df_all.columns:
#         df_all["importance_std"] = pd.to_numeric(df_all["importance_std"], errors="coerce")
#
#     df_summary = (
#         df_all.groupby(["algorithm_name", "feature"], as_index=False)
#         .agg(
#             importance_mean=("importance", "mean"),
#             importance_std_across_seeds=("importance", "std"),
#             n_seeds=("seed", "nunique"),
#         )
#         .sort_values("importance_mean", ascending=False)
#         .reset_index(drop=True)
#     )
#     df_summary["importance_std_across_seeds"] = df_summary["importance_std_across_seeds"].fillna(0.0)
#     df_summary.to_csv(out_dir / f"final_feature_importance_mean_std_{algorithm_name}.csv", index=False, encoding="utf-8")
#
#     dfp = df_summary.head(top_n).iloc[::-1]
#     fig, ax = plt.subplots(figsize=(7.4, max(4.4, 0.33 * len(dfp) + 1.2)))
#     _horizontal_mean_sd_barh(
#         ax,
#         np.arange(len(dfp)),
#         dfp["importance_mean"],
#         dfp["importance_std_across_seeds"],
#     )
#     ax.set_yticks(np.arange(len(dfp)))
#     ax.set_yticklabels([str(x).replace("_", " ") for x in dfp["feature"]])
#     ax.set_xlabel("Permutation importance, mean ± SD across seeds")
#     ax.set_title(f"Mean feature importance — {_normalise_algorithm_name(algorithm_name)}")
#     ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
#     ax.grid(False, axis="y")
#     _save_figure(fig, out_dir / f"final_feature_importance_{algorithm_name}")
#     plt.close(fig)
#     return df_summary
#
#
# # ---------------------------------------------------------------------
# # 6) Ablation e interpretability reutilizando sus CSV ya generados
# # ---------------------------------------------------------------------
# def _aggregate_ablation(base: Path, out_dir: Path, seeds: Iterable[int], algorithm_name: str, main_metric: str) -> pd.DataFrame:
#     import matplotlib.pyplot as plt
#
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / "ablation_study" / algorithm_name / f"ablation_metrics_{algorithm_name}.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         df = df.copy()
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         rows.append(df)
#     if not rows:
#         return pd.DataFrame()
#
#     df_all = pd.concat(rows, ignore_index=True)
#     df_all.to_csv(out_dir / f"final_ablation_all_seeds_{algorithm_name}.csv", index=False, encoding="utf-8")
#
#     metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
#     available = [m for m in metrics if m in df_all.columns]
#     summary = []
#     for condition, g in df_all.groupby("condition"):
#         row = {"algorithm_name": algorithm_name, "condition": condition, "n_seeds": int(g["seed"].nunique())}
#         if "n_features" in g.columns:
#             row["n_features_mean"] = float(pd.to_numeric(g["n_features"], errors="coerce").mean())
#         for m in available:
#             vals = pd.to_numeric(g[m], errors="coerce")
#             row[f"{m}_mean"] = float(vals.mean())
#             row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
#         summary.append(row)
#     df_summary = pd.DataFrame(summary)
#     df_summary.to_csv(out_dir / f"final_ablation_mean_std_{algorithm_name}.csv", index=False, encoding="utf-8")
#
#     if main_metric in available:
#         order = [c for c in ["baseline_generic", "full_musical"] if c in set(df_summary["condition"])]
#         dfp = df_summary.set_index("condition").loc[order].reset_index()
#         labels = ["Without musical\nfeatures" if c == "baseline_generic" else "With musical\nfeatures" for c in dfp["condition"]]
#         means = dfp[f"{main_metric}_mean"].to_numpy(float)
#         stds = dfp[f"{main_metric}_std"].to_numpy(float)
#         fig, ax = plt.subplots(figsize=(5.2, 4.2))
#         bars = ax.bar(np.arange(len(dfp)), means, yerr=stds, capsize=5, width=0.55)
#         for b, mean, std in zip(bars, means, stds):
#             ax.text(b.get_x() + b.get_width() / 2, mean + std + 0.012, f"{mean:.3f} ± {std:.3f}", ha="center", va="bottom", fontsize=10)
#         ax.set_xticks(np.arange(len(dfp)))
#         ax.set_xticklabels(labels)
#         ax.set_ylabel(main_metric.replace("_", " ").title())
#         ax.set_ylim(0, min(1.10, max(1.0, float(np.nanmax(means + stds)) + 0.12)))
#         ax.set_title(f"Mean ablation study — {_normalise_algorithm_name(algorithm_name)}")
#         ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
#         ax.grid(False, axis="x")
#         _save_figure(fig, out_dir / f"final_ablation_{main_metric}_{algorithm_name}")
#         plt.close(fig)
#     return df_summary
#
#
# def _aggregate_interpretability(base: Path, out_dir: Path, seeds: Iterable[int], algorithm_name: str, top_n: int) -> pd.DataFrame:
#     import matplotlib.pyplot as plt
#
#     rows = []
#     for seed in seeds:
#         p = base / str(seed) / "musical_interpretability" / algorithm_name / f"musical_interpretability_rank_{algorithm_name}.csv"
#         df = _read_csv_if_exists(p)
#         if df is None:
#             continue
#         df = df.copy()
#         df["seed"] = int(seed)
#         df["source_csv"] = str(p)
#         rows.append(df)
#     if not rows:
#         return pd.DataFrame()
#
#     df_all = pd.concat(rows, ignore_index=True)
#     df_all.to_csv(out_dir / f"final_musical_interpretability_all_seeds_{algorithm_name}.csv", index=False, encoding="utf-8")
#
#     def mode_or_first(s: pd.Series):
#         s = s.dropna()
#         if s.empty:
#             return ""
#         m = s.mode()
#         return m.iloc[0] if not m.empty else s.iloc[0]
#
#     rows_out = []
#     for feature, g in df_all.groupby("feature"):
#         row = {"algorithm_name": algorithm_name, "feature": feature, "n_seeds": int(g["seed"].nunique())}
#         for c in ["discriminative_score", "permutation_importance", "cohens_d_profano_minus_religioso", "abs_cohens_d"]:
#             if c in g.columns:
#                 vals = pd.to_numeric(g[c], errors="coerce")
#                 row[f"{c}_mean"] = float(vals.mean())
#                 row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
#         for c in ["enriched_in", "feature_family", "hypothesis_alignment", "description"]:
#             if c in g.columns:
#                 row[f"{c}_mode"] = mode_or_first(g[c])
#         if "is_idiomatic_musicological" in g.columns:
#             row["is_idiomatic_musicological"] = bool(g["is_idiomatic_musicological"].astype(bool).any())
#         rows_out.append(row)
#
#     sort_col = "discriminative_score_mean"
#     df_summary = pd.DataFrame(rows_out).sort_values(sort_col, ascending=False).reset_index(drop=True)
#     df_summary.to_csv(out_dir / f"final_musical_interpretability_mean_std_{algorithm_name}.csv", index=False, encoding="utf-8")
#
#     dfp = df_summary.head(top_n).iloc[::-1]
#     fig, ax = plt.subplots(figsize=(7.5, max(4.4, 0.34 * len(dfp) + 1.2)))
#     _horizontal_mean_sd_barh(
#         ax,
#         np.arange(len(dfp)),
#         dfp["discriminative_score_mean"],
#         dfp["discriminative_score_std"],
#     )
#     ax.set_yticks(np.arange(len(dfp)))
#     ax.set_yticklabels([str(x).replace("_", " ") for x in dfp["feature"]])
#     ax.set_xlabel("Discriminative score, mean ± SD across seeds")
#     ax.set_title(f"Stable discriminative features — {_normalise_algorithm_name(algorithm_name)}")
#     ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
#     ax.grid(False, axis="y")
#     _save_figure(fig, out_dir / f"final_musical_interpretability_{algorithm_name}")
#     plt.close(fig)
#     return df_summary
#
#
# # ---------------------------------------------------------------------
# # 7) API pública
# # ---------------------------------------------------------------------
# def aggregate_final_results(
#     base_results_dir: str | os.PathLike,
#     seeds: Iterable[int],
#     algorithm_names: Iterable[str],
#     style: str = "tableau-colorblind10",
#     main_metric: str = "balanced_accuracy",
#     top_n: int = 20,
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Genera el último paso del pipeline: resultados finales multi-seed.
#
#     Todo se guarda en <base_results_dir>/final_results. La función solo lee CSVs
#     generados previamente; si falta alguno, avisa y continúa.
#     """
#     base = Path(base_results_dir)
#     out_dir = base / "final_results"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     _setup_paper_style(style)
#
#     outputs: Dict[str, pd.DataFrame] = {}
#     outputs["model_comparison"] = _aggregate_model_comparison(base, out_dir, seeds)
#
#     for algorithm_name in algorithm_names:
#         print("\n==============================")
#         print("FINAL MULTI-SEED RESULTS")
#         print("==============================")
#         print("Modelo:", algorithm_name)
#
#         for class_suffix in ("profano", "religioso"):
#             print("Clase positiva:", _positive_display(class_suffix))
#             roc_all = _collect_roc(base, seeds, algorithm_name, class_suffix=class_suffix)
#             outputs[f"roc_{algorithm_name}_{class_suffix}"] = _aggregate_roc(
#                 roc_all, out_dir, algorithm_name, class_suffix=class_suffix
#             )
#
#             pr_all = _collect_pr(base, seeds, algorithm_name, class_suffix=class_suffix)
#             outputs[f"precision_recall_{algorithm_name}_{class_suffix}"] = _aggregate_pr(
#                 pr_all, out_dir, algorithm_name, class_suffix=class_suffix
#             )
#
#             thr_all = _collect_threshold(base, seeds, algorithm_name, class_suffix=class_suffix)
#             outputs[f"threshold_{algorithm_name}_{class_suffix}"] = _aggregate_threshold(
#                 thr_all, out_dir, algorithm_name, class_suffix=class_suffix
#             )
#
#         outputs[f"feature_importance_{algorithm_name}"] = _aggregate_feature_importance(
#             base, out_dir, seeds, algorithm_name, top_n=top_n, style=style
#         )
#
#         outputs[f"ablation_{algorithm_name}"] = _aggregate_ablation(
#             base, out_dir, seeds, algorithm_name, main_metric=main_metric
#         )
#         outputs[f"interpretability_{algorithm_name}"] = _aggregate_interpretability(
#             base, out_dir, seeds, algorithm_name, top_n=top_n
#         )
#
#     _write_index(out_dir, algorithm_names, seeds)
#     print("[OK] Resultados finales guardados en:", out_dir)
#     return outputs
#
#
# def _write_index(out_dir: Path, algorithm_names: Iterable[str], seeds: Iterable[int]) -> None:
#     lines = [
#         "# Final multi-seed results",
#         "",
#         f"Seeds: {', '.join(str(s) for s in seeds)}",
#         f"Algorithms: {', '.join(str(a) for a in algorithm_names)}",
#         "",
#         "This folder contains the final aggregated CSV and PNG/EPS figures computed only from previously saved per-seed CSV files.",
#         "Missing files are reported as warnings and do not interrupt the aggregation.",
#         "",
#         "Main outputs per algorithm:",
#         "- final_roc_curve_<algorithm>_profano.csv/png/eps",
#         "- final_roc_curve_<algorithm>_religioso.csv/png/eps",
#         "- final_precision_recall_curve_<algorithm>_profano.csv/png/eps",
#         "- final_precision_recall_curve_<algorithm>_religioso.csv/png/eps",
#         "- final_threshold_analysis_<algorithm>_profano.csv/png/eps",
#         "- final_threshold_analysis_<algorithm>_religioso.csv/png/eps",
#         "- final_feature_importance_<algorithm>.csv/png/eps",
#         "- final_ablation_<metric>_<algorithm>.csv/png/eps",
#         "- final_musical_interpretability_<algorithm>.csv/png/eps",
#         "",
#     ]
#     (out_dir / "README_final_results.md").write_text("\n".join(lines), encoding="utf-8")
#
#
#
