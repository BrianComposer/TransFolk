# -*- coding: utf-8 -*-
"""
feature_distributions.py

Distribuciones agregadas multi-seed Profano vs Religioso por feature para el
pipeline TEIMUS, con test de Kolmogorov-Smirnov y corrección de Bonferroni.

Este módulo NO entrena modelos ni modifica predicciones. Lee los corpus_split.csv
ya generados en cada seed, extrae/cachea las features MusicXML por seed y genera
un único conjunto final de figuras agregadas en:

    <base_results_dir>/final_results/feature_distributions/

Convención del proyecto:
    Profano   = 1
    Religioso = 0

Salidas principales:
    - final_feature_distribution_values_all_seeds.csv
    - final_feature_distribution_stats_by_seed.csv
    - final_feature_distribution_stats_mean_std.csv
    - final_feature_distribution_ks_bonferroni.csv
    - final_feature_distribution_curves.csv
    - final_feature_distribution_effects_top_<N>.png/.eps
    - final_distribution_<feature>_profano_vs_religioso.png/.eps

Nota estadística:
    Las features de una pieza no cambian entre seeds. Por tanto, el test KS se
    calcula sobre piezas únicas por __path para evitar pseudorreplicación. Las
    seeds se usan para estimar la estabilidad visual de las distribuciones.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None

try:
    from transfolk_features.extract_features import FEATURE_TITLES
except Exception:  # pragma: no cover
    FEATURE_TITLES = {}


_TABLEAU_COLORBLIND10 = [
    "#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1",
    "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF",
]
PROFANO_COLOR = _TABLEAU_COLORBLIND10[0]
RELIGIOSO_COLOR = _TABLEAU_COLORBLIND10[1]
BAND_ALPHA = 0.18
LINE_WIDTH = 2.2


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


def _save_figure(fig, out_base: Path, save_eps: bool = True) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), format="png")
    if save_eps:
        fig.savefig(str(out_base.with_suffix(".eps")), format="eps")


def _safe_name(text: str) -> str:
    s = str(text).strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "feature"


def _pretty_feature(feature: str) -> str:
    return FEATURE_TITLES.get(feature, str(feature).replace("_", " "))


def _feature_columns(df: pd.DataFrame) -> List[str]:
    meta = {"__path", "__file", "__y", "__label", "__split", "seed"}
    return [c for c in df.columns if c not in meta]


def _format_p_value(p_value: float) -> str:
    try:
        p_value = float(p_value)
    except Exception:
        return "n/a"
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}"


def _significance_stars(p_adjusted: float) -> str:
    try:
        p_adjusted = float(p_adjusted)
    except Exception:
        return ""
    if not np.isfinite(p_adjusted):
        return ""
    if p_adjusted < 0.001:
        return "***"
    if p_adjusted < 0.01:
        return "**"
    if p_adjusted < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------
# Extracción y cache por seed, sin generar gráficas por seed
# ---------------------------------------------------------------------
def _extract_or_load_feature_table_for_seed(
    df_split: pd.DataFrame,
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    cache_csv_path: Path,
    seed: int,
    force_recompute: bool = False,
) -> pd.DataFrame:
    if cache_csv_path.exists() and not force_recompute:
        print("[INFO] Cargando cache de distribuciones:", cache_csv_path)
        df_cached = pd.read_csv(cache_csv_path)
        if "__y" in df_cached.columns and df_cached["__y"].nunique(dropna=True) >= 2:
            df_cached["seed"] = int(seed)
            return df_cached
        print("[WARN] Cache de distribuciones inválida o con una sola clase. Se recalcula.")

    required = {"__path", "__file", "label_id", "split"}
    missing = required.difference(df_split.columns)
    if missing:
        raise ValueError(f"df_split no contiene las columnas necesarias: {sorted(missing)}")

    rows = []
    print(f"[INFO] Extrayendo features para distribuciones agregadas. Seed={seed}")
    for i, r in df_split.reset_index(drop=True).iterrows():
        p = str(r["__path"])
        try:
            feats = extract_features_musicxml(p)
            if not feats:
                print("[WARN] Sin features:", os.path.basename(p))
                continue
            y = int(r["label_id"])
            row = dict(feats)
            row["__path"] = p
            row["__file"] = os.path.basename(p)
            row["__y"] = y
            row["__label"] = "Profano" if y == 1 else "Religioso"
            row["__split"] = str(r["split"])
            rows.append(row)
        except Exception as exc:
            print("[WARN] Error extrayendo features:", os.path.basename(p), "->", repr(exc))

        if (i + 1) % 50 == 0:
            print(f"[INFO] Seed={seed}. Procesadas {i + 1}/{len(df_split)} obras")

    if not rows:
        raise RuntimeError(f"No se pudieron extraer features para distribuciones en seed={seed}.")

    df = pd.DataFrame(rows)
    meta = {"__path", "__file", "__y", "__label", "__split"}
    for c in df.columns:
        if c not in meta:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["__y"].nunique(dropna=True) < 2:
        raise RuntimeError(f"La tabla de features de seed={seed} contiene una sola clase.")

    cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv_path, index=False, encoding="utf-8")
    df["seed"] = int(seed)
    print("[OK] Cache de distribuciones guardada:", cache_csv_path)
    return df


def _collect_feature_values(
    base_results_dir: str | os.PathLike,
    seeds: Iterable[int],
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    force_recompute_features: bool = False,
) -> pd.DataFrame:
    base = Path(base_results_dir)
    rows = []

    for seed in seeds:
        split_csv = base / str(seed) / "corpus_split.csv"
        if not split_csv.exists():
            print("[WARN] No existe corpus_split.csv para distribuciones:", split_csv)
            continue
        df_split = pd.read_csv(split_csv)
        cache_csv = base / str(seed) / "feature_distributions_cache.csv"
        df_seed = _extract_or_load_feature_table_for_seed(
            df_split=df_split,
            extract_features_musicxml=extract_features_musicxml,
            cache_csv_path=cache_csv,
            seed=int(seed),
            force_recompute=force_recompute_features,
        )
        rows.append(df_seed)

    if not rows:
        raise RuntimeError("No se encontraron datos válidos para agregar distribuciones por feature.")

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------
# Estadísticos descriptivos y tests inferenciales
# ---------------------------------------------------------------------
def _cohens_d(profano: np.ndarray, religioso: np.ndarray) -> float:
    profano = np.asarray(profano, dtype=float)
    religioso = np.asarray(religioso, dtype=float)
    profano = profano[np.isfinite(profano)]
    religioso = religioso[np.isfinite(religioso)]
    if len(profano) < 2 or len(religioso) < 2:
        return float("nan")
    sp = float(np.std(profano, ddof=1))
    sr = float(np.std(religioso, ddof=1))
    pooled = np.sqrt((sp ** 2 + sr ** 2) / 2.0)
    if pooled <= 1e-12 or not np.isfinite(pooled):
        return 0.0
    return float((np.mean(profano) - np.mean(religioso)) / pooled)


def _compute_stats_by_seed(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, g_seed in df_all.groupby("seed"):
        for feature in _feature_columns(g_seed):
            prof = pd.to_numeric(g_seed.loc[g_seed["__y"] == 1, feature], errors="coerce").dropna().to_numpy(float)
            reli = pd.to_numeric(g_seed.loc[g_seed["__y"] == 0, feature], errors="coerce").dropna().to_numpy(float)
            prof = prof[np.isfinite(prof)]
            reli = reli[np.isfinite(reli)]
            if len(prof) == 0 or len(reli) == 0:
                continue
            mean_prof = float(np.mean(prof))
            mean_reli = float(np.mean(reli))
            median_prof = float(np.median(prof))
            median_reli = float(np.median(reli))
            diff = mean_prof - mean_reli
            d = _cohens_d(prof, reli)
            rows.append({
                "seed": int(seed),
                "feature": feature,
                "feature_label": _pretty_feature(feature),
                "n_profano": int(len(prof)),
                "n_religioso": int(len(reli)),
                "mean_profano": mean_prof,
                "mean_religioso": mean_reli,
                "std_profano": float(np.std(prof, ddof=1)) if len(prof) > 1 else 0.0,
                "std_religioso": float(np.std(reli, ddof=1)) if len(reli) > 1 else 0.0,
                "median_profano": median_prof,
                "median_religioso": median_reli,
                "difference_profano_minus_religioso": diff,
                "abs_difference": abs(diff),
                "cohens_d_profano_minus_religioso": d,
                "abs_cohens_d": abs(d) if np.isfinite(d) else np.nan,
                "enriched_in": "Profano" if diff > 0 else "Religioso" if diff < 0 else "Neutral",
            })
    if not rows:
        raise RuntimeError("No se pudieron calcular estadísticos Profano vs Religioso por seed.")
    return pd.DataFrame(rows)


def _aggregate_stats(stats_by_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mean_profano", "mean_religioso", "std_profano", "std_religioso",
        "median_profano", "median_religioso", "difference_profano_minus_religioso",
        "abs_difference", "cohens_d_profano_minus_religioso", "abs_cohens_d",
    ]
    rows = []
    for feature, g in stats_by_seed.groupby("feature"):
        row = {
            "feature": feature,
            "feature_label": _pretty_feature(feature),
            "n_seeds": int(g["seed"].nunique()),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        diff_mean = row["difference_profano_minus_religioso_mean"]
        row["enriched_in_mean"] = "Profano" if diff_mean > 0 else "Religioso" if diff_mean < 0 else "Neutral"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("abs_cohens_d_mean", ascending=False).reset_index(drop=True)


def _compute_ks_bonferroni(df_all: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if ks_2samp is None:
        raise ImportError("scipy no está disponible. Instala scipy para usar ks_2samp: pip install scipy")

    work = df_all.drop_duplicates(subset=["__path"]).copy() if "__path" in df_all.columns else df_all.copy()
    rows = []
    for feature in _feature_columns(work):
        prof = pd.to_numeric(work.loc[work["__y"] == 1, feature], errors="coerce").dropna().to_numpy(float)
        reli = pd.to_numeric(work.loc[work["__y"] == 0, feature], errors="coerce").dropna().to_numpy(float)
        prof = prof[np.isfinite(prof)]
        reli = reli[np.isfinite(reli)]
        if len(prof) < 2 or len(reli) < 2:
            stat, p_value = np.nan, np.nan
        else:
            res = ks_2samp(prof, reli, alternative="two-sided", mode="auto")
            stat, p_value = float(res.statistic), float(res.pvalue)
        rows.append({
            "feature": feature,
            "feature_label": _pretty_feature(feature),
            "ks_statistic": stat,
            "p_value": p_value,
            "n_profano_unique": int(len(prof)),
            "n_religioso_unique": int(len(reli)),
        })

    if not rows:
        raise RuntimeError("No se pudieron calcular tests KS por feature.")

    df = pd.DataFrame(rows)
    valid = df["p_value"].notna() & np.isfinite(df["p_value"])
    m = int(valid.sum())
    df["n_tests_bonferroni"] = m
    df["alpha"] = float(alpha)
    df["alpha_bonferroni"] = float(alpha / m) if m > 0 else np.nan
    df["p_value_bonferroni"] = np.nan
    if m > 0:
        df.loc[valid, "p_value_bonferroni"] = np.minimum(df.loc[valid, "p_value"].astype(float) * m, 1.0)
    df["significant_bonferroni"] = df["p_value_bonferroni"].astype(float) < float(alpha)
    df["significance"] = df["p_value_bonferroni"].apply(_significance_stars)
    return df.sort_values(["significant_bonferroni", "ks_statistic"], ascending=[False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Histogramas agregados multi-seed
# ---------------------------------------------------------------------
def _global_bins(values: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None

    uniq = np.unique(values)
    if len(uniq) <= 2 and np.all(np.isin(uniq, [0, 1])):
        return np.array([-0.5, 0.5, 1.5], dtype=float)
    if len(uniq) <= 12:
        lo = float(np.min(uniq))
        hi = float(np.max(uniq))
        if np.isclose(lo, hi):
            return np.array([lo - 0.5, lo + 0.5], dtype=float)
        step = max((hi - lo) / max(len(uniq), 2), 1e-6)
        return np.linspace(lo - step / 2.0, hi + step / 2.0, min(len(uniq) + 1, n_bins + 1))

    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = float(np.min(values)), float(np.max(values))
    if np.isclose(lo, hi):
        eps = max(abs(lo) * 0.05, 0.5)
        lo, hi = lo - eps, hi + eps
    return np.linspace(float(lo), float(hi), n_bins + 1)


def _density_by_seed(df_all: pd.DataFrame, feature: str, bins: np.ndarray, class_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = (bins[:-1] + bins[1:]) / 2.0
    densities = []
    for _seed, g_seed in df_all.groupby("seed"):
        vals = pd.to_numeric(g_seed.loc[g_seed["__y"] == class_id, feature], errors="coerce").dropna().to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        hist, _ = np.histogram(vals, bins=bins, density=True)
        hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
        densities.append(hist)
    if not densities:
        z = np.zeros_like(centers, dtype=float)
        return centers, z, z
    arr = np.vstack(densities)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1], dtype=float)
    return centers, mean, std


def _plot_distribution_feature(
    df_all: pd.DataFrame,
    feature: str,
    out_dir: Path,
    style: str,
    n_bins: int,
    ks_info: Optional[Dict[str, object]] = None,
) -> Optional[pd.DataFrame]:
    import matplotlib.pyplot as plt

    vals = pd.to_numeric(df_all[feature], errors="coerce").dropna().to_numpy(float)
    bins = _global_bins(vals, n_bins=n_bins)
    if bins is None or len(bins) < 2:
        return None

    x_p, y_p, s_p = _density_by_seed(df_all, feature, bins, class_id=1)
    x_r, y_r, s_r = _density_by_seed(df_all, feature, bins, class_id=0)

    _setup_paper_style(style)
    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    ax.plot(x_p, y_p, color=PROFANO_COLOR, linewidth=LINE_WIDTH, label="Profano")
    ax.fill_between(x_p, np.maximum(y_p - s_p, 0), y_p + s_p, color=PROFANO_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(x_r, y_r, color=RELIGIOSO_COLOR, linewidth=LINE_WIDTH, label="Religioso")
    ax.fill_between(x_r, np.maximum(y_r - s_r, 0), y_r + s_r, color=RELIGIOSO_COLOR, alpha=BAND_ALPHA, linewidth=0)

    sig = ""
    subtitle = ""
    if ks_info is not None:
        sig = str(ks_info.get("significance", "") or "")
        ks_stat = float(ks_info.get("ks_statistic", np.nan))
        p_adj = float(ks_info.get("p_value_bonferroni", np.nan))
        subtitle = f"KS D={ks_stat:.3f}; Bonferroni p={_format_p_value(p_adj)}"
        if bool(ks_info.get("significant_bonferroni", False)) and sig:
            subtitle += f" {sig}"

    ax.set_xlabel(_pretty_feature(feature))
    ax.set_ylabel("Density")
    ax.set_title(f"Profano vs Religioso — {_pretty_feature(feature)}{(' ' + sig) if sig else ''}")
    if subtitle:
        ax.text(
            0.02, 0.98, subtitle,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    _save_figure(fig, out_dir / f"final_distribution_{_safe_name(feature)}_profano_vs_religioso")
    plt.close(fig)

    return pd.DataFrame({
        "feature": feature,
        "bin_center": x_p,
        "profano_density_mean": y_p,
        "profano_density_std": s_p,
        "religioso_density_mean": y_r,
        "religioso_density_std": s_r,
    })


def _plot_effect_summary(df_summary: pd.DataFrame, out_dir: Path, style: str, top_n: int) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    dfp = df_summary.head(top_n).copy().iloc[::-1]
    y = np.arange(len(dfp))
    vals = dfp["cohens_d_profano_minus_religioso_mean"].astype(float).to_numpy()
    errs = dfp["cohens_d_profano_minus_religioso_std"].astype(float).fillna(0.0).to_numpy()
    labels = []
    for _, r in dfp.iterrows():
        base_label = str(r["feature"]).replace("_", " ")
        stars = str(r.get("significance", "") or "")
        labels.append(f"{base_label} {stars}" if stars else base_label)

    fig, ax = plt.subplots(figsize=(7.4, max(4.4, 0.34 * len(dfp) + 1.2)))
    colors = [PROFANO_COLOR if v >= 0 else RELIGIOSO_COLOR for v in vals]
    sig_col = dfp["significant_bonferroni"] if "significant_bonferroni" in dfp.columns else pd.Series(False, index=dfp.index)
    edgecolors = ["black" if bool(x) else "none" for x in sig_col]
    linewidths = [1.1 if bool(x) else 0.0 for x in sig_col]
    ax.barh(y, vals, xerr=errs, capsize=3, color=colors, edgecolor=edgecolors, linewidth=linewidths, alpha=0.95)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d, Profano − Religioso, mean ± SD across seeds")
    ax.set_title("Largest Profano–Religioso feature differences (KS/Bonferroni highlighted)")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, axis="y")
    ax.text(0.01, 0.02, "Religioso  ←     →  Profano", transform=ax.transAxes, ha="left", va="bottom")
    ax.text(0.99, 0.02, "* pBonf < .05; ** < .01; *** < .001", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    _save_figure(fig, out_dir / f"final_feature_distribution_effects_top_{top_n}")
    plt.close(fig)


# ---------------------------------------------------------------------
# API pública: solo agregado final multi-seed
# ---------------------------------------------------------------------
def aggregate_feature_distributions(
    base_results_dir: str | os.PathLike,
    seeds: Iterable[int],
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    style: str = "tableau-colorblind10",
    top_n: int = 20,
    n_bins: int = 40,
    max_features: Optional[int] = None,
    force_recompute_features: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Genera distribuciones agregadas multi-seed Profano vs Religioso.

    Además calcula, para cada feature, un test de Kolmogorov-Smirnov de dos
    muestras y aplica corrección de Bonferroni sobre todos los tests válidos.
    Las features significativas se resaltan con borde negro y asteriscos en la
    figura resumen, y con anotación KS/pBonf en cada distribución individual.
    """
    base = Path(base_results_dir)
    out_dir = base / "final_results" / "feature_distributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = _collect_feature_values(
        base_results_dir=base,
        seeds=seeds,
        extract_features_musicxml=extract_features_musicxml,
        force_recompute_features=force_recompute_features,
    )
    df_all.to_csv(out_dir / "final_feature_distribution_values_all_seeds.csv", index=False, encoding="utf-8")

    stats_by_seed = _compute_stats_by_seed(df_all)
    stats_by_seed.to_csv(out_dir / "final_feature_distribution_stats_by_seed.csv", index=False, encoding="utf-8")

    df_ks = _compute_ks_bonferroni(df_all, alpha=alpha)
    df_ks.to_csv(out_dir / "final_feature_distribution_ks_bonferroni.csv", index=False, encoding="utf-8")

    df_summary = _aggregate_stats(stats_by_seed)
    df_summary = df_summary.merge(
        df_ks[[
            "feature", "ks_statistic", "p_value", "p_value_bonferroni",
            "significant_bonferroni", "significance", "alpha", "alpha_bonferroni",
            "n_tests_bonferroni", "n_profano_unique", "n_religioso_unique",
        ]],
        on="feature",
        how="left",
    )
    df_summary.to_csv(out_dir / "final_feature_distribution_stats_mean_std.csv", index=False, encoding="utf-8")

    _plot_effect_summary(df_summary, out_dir, style=style, top_n=min(top_n, len(df_summary)))

    features = df_summary["feature"].tolist()
    if max_features is not None:
        features = features[:int(max_features)]

    curve_rows = []
    for feature in features:
        print("[INFO] Generando distribución agregada:", feature)
        ks_row = df_ks[df_ks["feature"] == feature]
        ks_info = ks_row.iloc[0].to_dict() if not ks_row.empty else None
        df_curve = _plot_distribution_feature(
            df_all=df_all,
            feature=feature,
            out_dir=out_dir,
            style=style,
            n_bins=n_bins,
            ks_info=ks_info,
        )
        if df_curve is not None:
            curve_rows.append(df_curve)

    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(
            out_dir / "final_feature_distribution_curves.csv",
            index=False,
            encoding="utf-8",
        )

    print("\n==============================")
    print("FEATURE DISTRIBUTIONS — FINAL MULTI-SEED + KS/BONFERRONI")
    print("==============================")
    print("Salida:", out_dir)
    print(df_summary.head(top_n)[[
        "feature", "enriched_in_mean", "cohens_d_profano_minus_religioso_mean",
        "cohens_d_profano_minus_religioso_std", "ks_statistic",
        "p_value_bonferroni", "significant_bonferroni", "n_seeds",
    ]].to_csv(sep="\t", index=False))

    return df_summary
