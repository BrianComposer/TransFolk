# -*- coding: utf-8 -*-
"""
interpretability.py

Interpretabilidad musical para el clasificador TEIMUS Profano vs Religioso.

Objetivo:
    1) Detectar qué features pesan más en cada clasificador.
    2) Identificar qué patrones discriminan realmente entre Religioso y Profano.
    3) Generar tablas, gráficas e interpretación automática para el paper.

Convención de clases:
    Religioso = 0
    Profano   = 1

El módulo respeta el pipeline existente: recibe df_split, reutiliza los modelos ya
entrenados en model_dir, usa extract_features_musicxml() del proyecto y guarda todo
dentro de la carpeta de resultados del experimento/seed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from transfolk_classifier.classifier import ALGORITHM_NAME


# ---------------------------------------------------------------------
# 1) Metadatos musicológicos legibles
# ---------------------------------------------------------------------
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "grace_note_ratio": "Proporción de notas de adorno o apoyaturas gráficas.",
    "short_ornament_window_ratio": "Densidad de gestos ornamentales breves en ventanas locales.",
    "turn_like_ratio": "Frecuencia de giros melódicos tipo bordadura/turn.",
    "appoggiatura_like_ratio": "Frecuencia de apoyaturas melódicas detectadas por patrón interválico-rítmico.",
    "triplet_in_binary_ratio": "Presencia de tresillos en contexto binario.",
    "dotted_rhythm_ratio": "Presencia de ritmos apuntillados, relevantes para repertorio profano.",
    "initial_interval_is_6th": "Comienzo con intervalo de sexta.",
    "final_leading_tone_appoggiatura": "Apoyatura de sensible cerca del final de frase.",
    "strong_weak_semitone_resolution_ratio": "Resolución por semitono desde parte fuerte a parte débil.",
    "chromatic_usage_ratio": "Uso relativo de cromatismos.",
    "retardo_la_solsharp_ratio": "Patrón de retardo la–sol#.",
    "retardo_si_la_ratio": "Patrón de retardo si–la.",
    "minor_leading_to_mediant_ratio": "Giro de sensible a mediante en modo menor.",
    "note_density": "Densidad de notas por unidad temporal.",
    "mean_dur": "Duración media de los eventos.",
    "cv_dur": "Variabilidad relativa de duraciones.",
    "short_note_ratio": "Proporción de notas breves.",
    "rhythmic_entropy": "Entropía de las duraciones.",
    "npvi_rhythmic": "Variabilidad rítmica normalizada entre eventos sucesivos.",
    "rhythmic_energy": "Energía/actividad rítmica global.",
    "strong_beat_note_ratio": "Proporción de notas situadas en parte fuerte.",
    "syncopation_index": "Índice de síncopa.",
    "reciting_pitch_ratio": "Tendencia a insistir en altura recitativa.",
    "pitch_entropy": "Entropía de clases de altura.",
    "final_tonic_match": "Coincidencia de la nota final con la tónica estimada.",
    "last_note_duration_ratio": "Peso relativo de la duración final.",
    "last_interval_abs": "Tamaño absoluto del último intervalo.",
    "last_interval_is_step": "El último intervalo es conjunto.",
    "diatonic_ratio": "Proporción de alturas diatónicas respecto a la tonalidad estimada.",
    "best_corr": "Correlación tonal máxima del perfil estimado.",
    "key_clarity": "Claridad tonal estimada.",
    "best_mode_minor": "Indicador de modo menor estimado.",
    "mean_abs_semitones": "Tamaño medio absoluto de intervalo.",
    "interval_std": "Dispersión de tamaños interválicos.",
    "max_leap": "Mayor salto melódico.",
    "step_ratio": "Proporción de movimiento conjunto.",
    "unison_ratio": "Proporción de repeticiones de nota.",
    "interval_2nd_ratio": "Proporción de segundas.",
    "interval_3rd_ratio": "Proporción de terceras.",
    "interval_4th_ratio": "Proporción de cuartas.",
    "interval_5th_ratio": "Proporción de quintas.",
    "interval_6th_ratio": "Proporción de sextas.",
    "interval_7th_ratio": "Proporción de séptimas.",
    "interval_octave_plus_ratio": "Proporción de octavas o saltos mayores.",
    "tritone_ratio": "Proporción de tritonos.",
    "consonant_interval_ratio": "Proporción de intervalos consonantes.",
    "range_semitones": "Ámbito melódico total en semitonos.",
    "range_p95": "Ámbito robusto basado en percentiles.",
    "range_relative": "Ámbito relativo normalizado.",
    "proximity_step_le2": "Proporción de movimientos de hasta dos semitonos.",
    "proximity_inv_mean": "Índice inverso de proximidad melódica.",
    "up_ratio": "Proporción de movimientos ascendentes.",
    "down_ratio": "Proporción de movimientos descendentes.",
    "stay_ratio": "Proporción de repeticiones.",
    "num_direction_changes": "Número de cambios de dirección melódica.",
    "climax_pos": "Posición relativa del clímax melódico.",
    "direction_balance": "Balance entre movimiento ascendente y descendente.",
    "pitch_time_slope": "Pendiente global altura-tiempo.",
    "mean_ioi": "Distancia temporal media entre ataques.",
}

IDIOMATIC_FEATURES = {
    "grace_note_ratio", "short_ornament_window_ratio", "turn_like_ratio",
    "appoggiatura_like_ratio", "triplet_in_binary_ratio", "dotted_rhythm_ratio",
    "initial_interval_is_6th", "final_leading_tone_appoggiatura",
    "strong_weak_semitone_resolution_ratio", "chromatic_usage_ratio",
    "retardo_la_solsharp_ratio", "retardo_si_la_ratio",
    "minor_leading_to_mediant_ratio",
}

RELIGIOUS_HYPOTHESIS_FEATURES = {
    "appoggiatura_like_ratio", "final_leading_tone_appoggiatura",
    "strong_weak_semitone_resolution_ratio", "chromatic_usage_ratio",
    "retardo_la_solsharp_ratio", "retardo_si_la_ratio",
    "minor_leading_to_mediant_ratio", "initial_interval_is_6th",
    "triplet_in_binary_ratio",
}

PROFANE_HYPOTHESIS_FEATURES = {"dotted_rhythm_ratio"}


def _feature_family(feature: str) -> str:
    if feature in IDIOMATIC_FEATURES:
        return "musicological_idiomatic"
    if any(k in feature for k in ["rhythm", "dur", "ioi", "beat", "syncop", "density"]):
        return "rhythm"
    if any(k in feature for k in ["interval", "leap", "step", "tritone", "consonant", "semitone"]):
        return "intervallic"
    if any(k in feature for k in ["pitch", "key", "mode", "diatonic", "tonic"]):
        return "tonal_pitch"
    if any(k in feature for k in ["range", "climax", "direction", "up_", "down_", "stay", "proximity"]):
        return "contour_range"
    return "generic_symbolic"


# ---------------------------------------------------------------------
# 2) Estilo gráfico compatible con classifier_curves.py
# ---------------------------------------------------------------------
def _setup_paper_style(style: str = "tableau-colorblind10", font_size: int = 11) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("default")
    try:
        if style in plt.style.available:
            plt.style.use(style)
    except Exception:
        pass

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
        "grid.alpha": 0.35,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_figure(fig, out_base: Path, save_eps: bool = True) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=400, bbox_inches="tight")
    if save_eps:
        fig.savefig(str(out_base.with_suffix(".eps")), format="eps", bbox_inches="tight")


# ---------------------------------------------------------------------
# 3) Carga/extracción de features
# ---------------------------------------------------------------------
def _extract_or_load_feature_table(
    df_split: pd.DataFrame,
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    cache_csv_path: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    if cache_csv_path.exists() and not force_recompute:
        print("[INFO] Cargando cache de features para interpretabilidad:", cache_csv_path)
        return pd.read_csv(cache_csv_path)

    rows = []
    print("[INFO] Extrayendo features para interpretabilidad musical...")
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
        raise RuntimeError("No se pudieron extraer features para interpretabilidad musical.")

    df = pd.DataFrame(rows)
    meta = {"__path", "__file", "__y", "__split"}
    for c in df.columns:
        if c not in meta:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv_path, index=False, encoding="utf-8")
    print("[OK] Cache guardada:", cache_csv_path)
    return df


def _load_model_and_features(model_dir: str | os.PathLike) -> Tuple[object, List[str], Dict[str, object]]:
    model_dir = Path(model_dir)
    model_path = model_dir / "model.joblib"
    feat_path = model_dir / "features.json"
    summary_path = model_dir / "train_summary.json"

    if not model_path.exists() or not feat_path.exists():
        raise FileNotFoundError(f"No encuentro model.joblib o features.json en: {model_dir}")

    model = joblib.load(model_path)
    feature_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    summary: Dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return model, feature_cols, summary


def _available_eval_data(df_features: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    eval_df = df_features[df_features["__split"].astype(str).isin(["test", "eval"])].copy()
    if eval_df.empty:
        raise RuntimeError("No hay split test/eval para calcular interpretabilidad.")

    missing = [c for c in feature_cols if c not in eval_df.columns]
    if missing:
        print("[WARN] Features del modelo ausentes en la extracción; se rellenan con NaN:", missing)
        for c in missing:
            eval_df[c] = np.nan

    X = eval_df[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    y = eval_df["__y"].astype(int).to_numpy()
    return X, y


# ---------------------------------------------------------------------
# 4) Importancias y patrones discriminantes
# ---------------------------------------------------------------------
def _safe_score_auc(model, X: pd.DataFrame, y: np.ndarray) -> float:
    try:
        if hasattr(model, "predict_proba"):
            return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
        if hasattr(model, "decision_function"):
            return float(roc_auc_score(y, model.decision_function(X)))
    except Exception:
        return np.nan
    return np.nan


def _native_importance(model, feature_cols: Sequence[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") and "clf" in model.named_steps else model
    signed = None
    importance = None

    if hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        signed = np.asarray(clf.coef_, dtype=float)
        if signed.ndim == 2:
            signed = signed[0]
        importance = np.abs(signed)
    elif isinstance(clf, CalibratedClassifierCV) and hasattr(clf, "estimator") and hasattr(clf.estimator, "coef_"):
        signed = np.asarray(clf.estimator.coef_, dtype=float)
        if signed.ndim == 2:
            signed = signed[0]
        importance = np.abs(signed)

    if importance is None or len(importance) != len(feature_cols):
        return pd.DataFrame({"feature": list(feature_cols), "native_importance": np.nan, "signed_weight": np.nan})

    df = pd.DataFrame({"feature": list(feature_cols), "native_importance": importance})
    df["signed_weight"] = signed if signed is not None and len(signed) == len(feature_cols) else np.nan
    return df


def _class_pattern_statistics(df_features: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    work = df_features.copy()
    available = [c for c in feature_cols if c in work.columns]

    for f in available:
        vals_rel = pd.to_numeric(work.loc[work["__y"] == 0, f], errors="coerce")
        vals_pro = pd.to_numeric(work.loc[work["__y"] == 1, f], errors="coerce")
        rel_mean = float(vals_rel.mean()) if vals_rel.notna().any() else np.nan
        pro_mean = float(vals_pro.mean()) if vals_pro.notna().any() else np.nan
        rel_std = float(vals_rel.std(ddof=1)) if vals_rel.notna().sum() > 1 else 0.0
        pro_std = float(vals_pro.std(ddof=1)) if vals_pro.notna().sum() > 1 else 0.0
        pooled = np.sqrt((rel_std ** 2 + pro_std ** 2) / 2.0)
        effect = np.nan if pooled == 0 or np.isnan(pooled) else (pro_mean - rel_mean) / pooled
        diff = pro_mean - rel_mean
        enriched = "Profano" if diff > 0 else "Religioso" if diff < 0 else "Neutral"

        if f in RELIGIOUS_HYPOTHESIS_FEATURES and enriched == "Religioso":
            hypothesis = "supports_religious_hypothesis"
        elif f in PROFANE_HYPOTHESIS_FEATURES and enriched == "Profano":
            hypothesis = "supports_profane_hypothesis"
        elif f in IDIOMATIC_FEATURES:
            hypothesis = "idiomatic_but_direction_unexpected_or_neutral"
        else:
            hypothesis = "generic_descriptor"

        rows.append({
            "feature": f,
            "mean_religioso": rel_mean,
            "mean_profano": pro_mean,
            "difference_profano_minus_religioso": diff,
            "abs_difference": abs(diff) if not np.isnan(diff) else np.nan,
            "cohens_d_profano_minus_religioso": effect,
            "abs_cohens_d": abs(effect) if not np.isnan(effect) else np.nan,
            "enriched_in": enriched,
            "feature_family": _feature_family(f),
            "is_idiomatic_musicological": f in IDIOMATIC_FEATURES,
            "hypothesis_alignment": hypothesis,
            "description": FEATURE_DESCRIPTIONS.get(f, "Descriptor simbólico extraído del MusicXML."),
        })

    return pd.DataFrame(rows)


def _compute_interpretability_tables(
    model,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    df_features: pd.DataFrame,
    algorithm_name: str,
    random_state: int,
    n_repeats: int,
    scoring: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("[INFO] Calculando permutation importance para:", algorithm_name)
    result = permutation_importance(
        estimator=model,
        X=X_eval,
        y=y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": list(X_eval.columns),
        "permutation_importance": result.importances_mean,
        "permutation_importance_std": result.importances_std,
    })
    df_native = _native_importance(model, list(X_eval.columns))
    df_patterns = _class_pattern_statistics(df_features, list(X_eval.columns))

    df = df_perm.merge(df_native, on="feature", how="left").merge(df_patterns, on="feature", how="left")
    max_perm = float(df["permutation_importance"].abs().max()) if len(df) else 0.0
    max_effect = float(df["abs_cohens_d"].max()) if len(df) and df["abs_cohens_d"].notna().any() else 0.0
    df["importance_norm"] = 0.0 if max_perm == 0 else df["permutation_importance"].clip(lower=0) / max_perm
    df["effect_norm"] = 0.0 if max_effect == 0 else df["abs_cohens_d"].fillna(0) / max_effect
    df["discriminative_score"] = 0.70 * df["importance_norm"] + 0.30 * df["effect_norm"]
    df["algorithm_name"] = algorithm_name
    df = df.sort_values(["discriminative_score", "permutation_importance"], ascending=False).reset_index(drop=True)

    family = (
        df.groupby("feature_family", dropna=False)
        .agg(
            n_features=("feature", "count"),
            total_permutation_importance=("permutation_importance", "sum"),
            mean_permutation_importance=("permutation_importance", "mean"),
            max_discriminative_score=("discriminative_score", "max"),
            n_idiomatic=("is_idiomatic_musicological", "sum"),
        )
        .reset_index()
        .sort_values("total_permutation_importance", ascending=False)
    )
    family["algorithm_name"] = algorithm_name
    return df, family


# ---------------------------------------------------------------------
# 5) Gráficas
# ---------------------------------------------------------------------
def _plot_top_features(df_rank: pd.DataFrame, out_dir: Path, algorithm_name: str, style: str, top_n: int) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    dfp = df_rank.head(top_n).copy().iloc[::-1]
    labels = [f.replace("_", " ") for f in dfp["feature"]]

    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.32 * len(dfp) + 1.2)))
    ax.barh(np.arange(len(dfp)), dfp["permutation_importance"].astype(float))
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Permutation importance (balanced accuracy drop)")
    ax.set_title(f"Most influential features — {algorithm_name.replace('_', ' ').title()}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"interpretability_top_features_{algorithm_name}")
    plt.close(fig)


def _plot_pattern_effects(df_rank: pd.DataFrame, out_dir: Path, algorithm_name: str, style: str, top_n: int) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    dfp = df_rank.sort_values("abs_cohens_d", ascending=False).head(top_n).copy().iloc[::-1]
    labels = [f.replace("_", " ") for f in dfp["feature"]]
    values = dfp["cohens_d_profano_minus_religioso"].astype(float)

    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.32 * len(dfp) + 1.2)))
    ax.barh(np.arange(len(dfp)), values)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d: Profano − Religioso")
    ax.set_title(f"Most discriminative musical patterns — {algorithm_name.replace('_', ' ').title()}")
    ax.text(0.01, 0.02, "Religioso ←     → Profano", transform=ax.transAxes, ha="left", va="bottom")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"interpretability_pattern_effects_{algorithm_name}")
    plt.close(fig)


def _plot_family_importance(df_family: pd.DataFrame, out_dir: Path, algorithm_name: str, style: str) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    dfp = df_family.copy().sort_values("total_permutation_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(np.arange(len(dfp)), dfp["total_permutation_importance"].astype(float))
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in dfp["feature_family"]])
    ax.set_xlabel("Total permutation importance")
    ax.set_title(f"Importance by feature family — {algorithm_name.replace('_', ' ').title()}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"interpretability_family_importance_{algorithm_name}")
    plt.close(fig)


def _write_interpretation(
    df_rank: pd.DataFrame,
    df_family: pd.DataFrame,
    out_path: Path,
    algorithm_name: str,
    model_score: float,
    top_n: int = 12,
) -> str:
    top = df_rank.head(top_n).copy()
    idiom = top[top["is_idiomatic_musicological"] == True]
    rel_support = top[top["hypothesis_alignment"] == "supports_religious_hypothesis"]
    prof_support = top[top["hypothesis_alignment"] == "supports_profane_hypothesis"]

    fam_text = ""
    if not df_family.empty:
        best_fam = df_family.iloc[0]
        fam_text = f"The strongest family-level contribution is `{best_fam['feature_family']}` with total permutation importance {float(best_fam['total_permutation_importance']):.4f}."

    lines = [
        "# Musical interpretability report",
        "",
        f"**Model:** `{algorithm_name}`",
        f"**Evaluation ROC-AUC:** {model_score:.4f}" if np.isfinite(model_score) else "**Evaluation ROC-AUC:** not available",
        "",
        "## What features weigh more?",
        "",
        "| Rank | Feature | Importance | Enriched in | Pattern effect | Family | Interpretation |",
        "|---:|---|---:|---|---:|---|---|",
    ]
    for i, r in top.iterrows():
        lines.append(
            f"| {int(i) + 1} | `{r['feature']}` | {float(r['permutation_importance']):.4f} | "
            f"{r['enriched_in']} | {float(r['cohens_d_profano_minus_religioso']) if pd.notna(r['cohens_d_profano_minus_religioso']) else np.nan:.3f} | "
            f"{r['feature_family']} | {r['description']} |"
        )

    lines += ["", "## What patterns really discriminate?", ""]
    if len(idiom) > 0:
        names = ", ".join([f"`{x}`" for x in idiom["feature"].head(8)])
        lines.append(f"Among the top-ranked variables, the explicitly musicological features are: {names}.")
    else:
        lines.append("No explicitly musicological feature appears in the top-ranked variables for this seed/model; the discrimination is dominated by generic symbolic descriptors.")

    if len(rel_support) > 0:
        names = ", ".join([f"`{x}`" for x in rel_support["feature"].head(8)])
        lines.append(f"The features aligned with the religious-style hypothesis are: {names}.")
    if len(prof_support) > 0:
        names = ", ".join([f"`{x}`" for x in prof_support["feature"].head(8)])
        lines.append(f"The features aligned with the profane-style hypothesis are: {names}.")

    lines += ["", fam_text, "", "## Suggested paper sentence", ""]
    if len(idiom) > 0:
        lines.append(
            "The interpretability analysis indicates that the classifier is not relying only on generic pitch/rhythm statistics: several high-ranked variables correspond to musicologically motivated idiomatic cues, which helps connect the computational decision boundary with concrete stylistic patterns."
        )
    else:
        lines.append(
            "The interpretability analysis suggests that, for this configuration, the classifier mainly exploits broad symbolic descriptors; the idiomatic hypotheses should therefore be discussed cautiously or reinforced through feature selection and multi-seed aggregation."
        )

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------
# 6) API pública por seed/modelo
# ---------------------------------------------------------------------
def run_musical_interpretability(
    df_split: pd.DataFrame,
    model_dir: str,
    output_dir: str,
    extract_features_musicxml: Callable[[str], Optional[Dict[str, float]]],
    algorithm_name: Optional[str] = None,
    random_state: int = 42,
    style: str = "tableau-colorblind10",
    scoring: str = "balanced_accuracy",
    n_repeats: int = 30,
    top_n: int = 20,
    force_recompute_features: bool = False,
) -> pd.DataFrame:
    """Ejecuta la interpretabilidad musical de un modelo ya entrenado."""
    output_path = Path(output_dir)
    model, feature_cols, summary = _load_model_and_features(model_dir)
    algorithm_name = algorithm_name or str(summary.get("algorithm_name", Path(model_dir).name))

    interp_dir = output_path / "musical_interpretability" / algorithm_name
    interp_dir.mkdir(parents=True, exist_ok=True)

    df_features = _extract_or_load_feature_table(
        df_split=df_split,
        extract_features_musicxml=extract_features_musicxml,
        cache_csv_path=output_path / "interpretability_features_cache.csv",
        force_recompute=force_recompute_features,
    )

    X_eval, y_eval = _available_eval_data(df_features, feature_cols)
    df_rank, df_family = _compute_interpretability_tables(
        model=model,
        X_eval=X_eval,
        y_eval=y_eval,
        df_features=df_features,
        algorithm_name=algorithm_name,
        random_state=random_state,
        n_repeats=n_repeats,
        scoring=scoring,
    )

    model_auc = _safe_score_auc(model, X_eval, y_eval)
    try:
        model_bacc = float(balanced_accuracy_score(y_eval, model.predict(X_eval)))
    except Exception:
        model_bacc = np.nan
    df_rank["model_eval_roc_auc"] = model_auc
    df_rank["model_eval_balanced_accuracy"] = model_bacc

    rank_csv = interp_dir / f"musical_interpretability_rank_{algorithm_name}.csv"
    family_csv = interp_dir / f"musical_interpretability_families_{algorithm_name}.csv"
    df_rank.to_csv(rank_csv, index=False, encoding="utf-8")
    df_family.to_csv(family_csv, index=False, encoding="utf-8")

    _plot_top_features(df_rank, interp_dir, algorithm_name, style, top_n=min(top_n, 20))
    _plot_pattern_effects(df_rank, interp_dir, algorithm_name, style, top_n=min(top_n, 20))
    _plot_family_importance(df_family, interp_dir, algorithm_name, style)

    interpretation = _write_interpretation(
        df_rank=df_rank,
        df_family=df_family,
        out_path=interp_dir / f"musical_interpretability_report_{algorithm_name}.md",
        algorithm_name=algorithm_name,
        model_score=model_auc,
        top_n=min(top_n, 12),
    )

    print("\n==============================")
    print("MUSICAL INTERPRETABILITY")
    print("==============================")
    print("Modelo:", algorithm_name)
    print("Balanced accuracy:", "nan" if pd.isna(model_bacc) else f"{model_bacc:.4f}")
    print("ROC-AUC:", "nan" if pd.isna(model_auc) else f"{model_auc:.4f}")
    print("[OK] Ranking guardado en:", rank_csv)
    print("\nTOP FEATURES")
    print(df_rank.head(15)[[
        "feature", "permutation_importance", "enriched_in", "cohens_d_profano_minus_religioso",
        "feature_family", "hypothesis_alignment"
    ]].to_csv(sep="\t", index=False))
    print(interpretation)
    return df_rank


# ---------------------------------------------------------------------
# 7) Agregación multi-seed para paper
# ---------------------------------------------------------------------
def _collect_interpretability_files(
    base_results_dir: str | os.PathLike,
    seeds: Iterable[int],
    algorithm_names: Iterable[str],
) -> pd.DataFrame:
    base = Path(base_results_dir)
    rows = []
    for seed in seeds:
        for algorithm_name in algorithm_names:
            p = base / str(seed) / "musical_interpretability" / algorithm_name / f"musical_interpretability_rank_{algorithm_name}.csv"
            if not p.exists():
                print("[WARN] No existe interpretability rank:", p)
                continue
            df = pd.read_csv(p)
            df["seed"] = int(seed)
            df["source_csv"] = str(p)
            rows.append(df)
    if not rows:
        raise RuntimeError("No se encontraron CSV de interpretabilidad para agregar.")
    return pd.concat(rows, ignore_index=True)


def _plot_multiseed_top(df_summary: pd.DataFrame, out_dir: Path, algorithm_name: str, style: str, top_n: int) -> None:
    import matplotlib.pyplot as plt

    _setup_paper_style(style)
    dfp = df_summary[df_summary["algorithm_name"] == algorithm_name].head(top_n).copy().iloc[::-1]
    if dfp.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, max(4.4, 0.34 * len(dfp) + 1.2)))
    ax.barh(np.arange(len(dfp)), dfp["discriminative_score_mean"].astype(float), xerr=dfp["discriminative_score_std"].astype(float), capsize=3)
    ax.set_yticks(np.arange(len(dfp)))
    ax.set_yticklabels([str(x).replace("_", " ") for x in dfp["feature"]])
    ax.set_xlabel("Mean discriminative score across seeds")
    ax.set_title(f"Stable discriminative features — {algorithm_name.replace('_', ' ').title()}")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.grid(False, axis="y")
    _save_figure(fig, out_dir / f"musical_interpretability_multiseed_top_{algorithm_name}")
    plt.close(fig)


def _write_multiseed_report(df_summary: pd.DataFrame, out_path: Path, top_n: int = 15) -> str:
    lines = [
        "# Multi-seed musical interpretability report",
        "",
        "This report aggregates feature importance and class-contrast statistics across seeds.",
        "",
    ]
    for algorithm_name, g in df_summary.groupby("algorithm_name"):
        top = g.head(top_n)
        lines += [f"## {algorithm_name}", "", "| Rank | Feature | Score | Importance | Enriched in | Family | Interpretation |", "|---:|---|---:|---:|---|---|---|"]
        for i, r in top.reset_index(drop=True).iterrows():
            lines.append(
                f"| {i + 1} | `{r['feature']}` | {float(r['discriminative_score_mean']):.4f} ± {float(r['discriminative_score_std']):.4f} | "
                f"{float(r['permutation_importance_mean']):.4f} | {r['enriched_in_mode']} | {r['feature_family']} | {r['description']} |"
            )
        idiom = top[top["is_idiomatic_musicological"] == True]
        lines += ["", "**Reading:** "]
        if len(idiom) > 0:
            names = ", ".join([f"`{x}`" for x in idiom["feature"].head(8)])
            lines.append(f"The stable top-ranked set includes musicological cues: {names}.")
        else:
            lines.append("The stable top-ranked set is dominated by generic symbolic descriptors rather than explicitly idiomatic cues.")
        lines.append("")

    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def aggregate_musical_interpretability(
    base_results_dir: str,
    seeds: Iterable[int],
    algorithm_names: Iterable[str],
    style: str = "tableau-colorblind10",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Agrega la interpretabilidad musical de todas las seeds.

    Outputs en <base_results_dir>/musical_interpretability_summary/:
        - musical_interpretability_all_seed_rankings.csv
        - musical_interpretability_summary_mean_std.csv
        - musical_interpretability_multiseed_top_<algorithm>.png/.eps
        - musical_interpretability_multiseed_report.md
    """
    base = Path(base_results_dir)
    out_dir = base / "musical_interpretability_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = _collect_interpretability_files(base, seeds, algorithm_names)
    df_all.to_csv(out_dir / "musical_interpretability_all_seed_rankings.csv", index=False, encoding="utf-8")

    rows = []
    group_cols = ["algorithm_name", "feature"]
    for (algorithm_name, feature), g in df_all.groupby(group_cols):
        def mode_or_first(s: pd.Series):
            s = s.dropna()
            if s.empty:
                return ""
            return s.mode().iloc[0] if not s.mode().empty else s.iloc[0]

        vals_score = g["discriminative_score"].astype(float)
        vals_imp = g["permutation_importance"].astype(float)
        vals_d = g["cohens_d_profano_minus_religioso"].astype(float)
        rows.append({
            "algorithm_name": algorithm_name,
            "feature": feature,
            "n_seeds": int(g["seed"].nunique()),
            "discriminative_score_mean": float(vals_score.mean()),
            "discriminative_score_std": float(vals_score.std(ddof=1)) if len(vals_score) > 1 else 0.0,
            "permutation_importance_mean": float(vals_imp.mean()),
            "permutation_importance_std": float(vals_imp.std(ddof=1)) if len(vals_imp) > 1 else 0.0,
            "cohens_d_mean": float(vals_d.mean()),
            "cohens_d_std": float(vals_d.std(ddof=1)) if len(vals_d) > 1 else 0.0,
            "abs_cohens_d_mean": float(g["abs_cohens_d"].astype(float).mean()),
            "enriched_in_mode": mode_or_first(g["enriched_in"]),
            "feature_family": mode_or_first(g["feature_family"]),
            "is_idiomatic_musicological": bool(g["is_idiomatic_musicological"].astype(bool).any()),
            "hypothesis_alignment_mode": mode_or_first(g["hypothesis_alignment"]),
            "description": mode_or_first(g["description"]),
        })

    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values(["algorithm_name", "discriminative_score_mean"], ascending=[True, False]).reset_index(drop=True)
    df_summary.to_csv(out_dir / "musical_interpretability_summary_mean_std.csv", index=False, encoding="utf-8")

    for algorithm_name in df_summary["algorithm_name"].dropna().unique():
        _plot_multiseed_top(df_summary, out_dir, str(algorithm_name), style, top_n=top_n)

    report = _write_multiseed_report(df_summary, out_dir / "musical_interpretability_multiseed_report.md", top_n=min(top_n, 15))

    print("\n==============================")
    print("MULTI-SEED MUSICAL INTERPRETABILITY")
    print("==============================")
    print(df_summary.groupby("algorithm_name").head(10).to_string(index=False))
    print("\n" + report)
    print("[OK] Resumen multi-seed guardado en:", out_dir)
    return df_summary
