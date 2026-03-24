# -----------------------------
# CALCULOS AUC
# -----------------------------
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def auc_report_to_latex(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    feature_list=None,
    top_k: int = 12,
    exclude_cols=None,
    decimals: int = 3,
    table_caption_top: str = None,
    table_label_top: str = "tab:top-features",
    table_caption_all: str = None,
    table_label_all: str = "tab:auc-all",
    print_skipped: bool = True,
) -> pd.DataFrame:
    """
    Calcula AUC univariado por feature (A vs B) y imprime tablas LaTeX listas para paper.

    - df_a: DataFrame del corpus A (y=1).
    - df_b: DataFrame del corpus B (y=0).
    - label_a / label_b: nombres para la dirección (p.ej. 'Profano', 'Religioso').
    - feature_list: lista de columnas a evaluar. Si None, usa intersección de columnas numéricas.
    - exclude_cols: columnas a excluir (paths, ids, etc.).
    - top_k: nº de features en la tabla Top.
    Devuelve un DataFrame con el ranking completo.
    """

    if exclude_cols is None:
        exclude_cols = set()
    else:
        exclude_cols = set(exclude_cols)

    # Si no pasas feature_list: intersección de columnas, excluyendo no numéricas
    if feature_list is None:
        common = [c for c in df_a.columns if c in df_b.columns and c not in exclude_cols]
        feature_list = []
        for c in common:
            if pd.api.types.is_numeric_dtype(df_a[c]) and pd.api.types.is_numeric_dtype(df_b[c]):
                feature_list.append(c)

    rows = []
    skipped = []

    for feat in feature_list:
        if feat in exclude_cols:
            continue
        if feat not in df_a.columns or feat not in df_b.columns:
            skipped.append((feat, "no está en ambos DF"))
            continue

        a = pd.to_numeric(df_a[feat], errors="coerce").to_numpy()
        b = pd.to_numeric(df_b[feat], errors="coerce").to_numpy()

        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) < 2 or len(b) < 2:
            skipped.append((feat, "demasiados NaN / pocos datos"))
            continue

        x = np.concatenate([a, b])
        y = np.concatenate([np.ones(len(a), dtype=int), np.zeros(len(b), dtype=int)])

        # Si es constante (o casi), AUC no tiene sentido
        if np.nanstd(x) == 0:
            skipped.append((feat, "sin varianza (constante)"))
            continue

        try:
            auc = float(roc_auc_score(y, x))
        except Exception as e:
            skipped.append((feat, f"error AUC: {e}"))
            continue

        # Normalización: AUC* siempre >= 0.5
        if auc >= 0.5:
            auc_star = auc
            direction = f"{label_a} > {label_b}"
        else:
            auc_star = 1.0 - auc
            direction = f"{label_b} > {label_a}"

        rows.append(
            {
                "feature": feat,
                "auc": auc,
                "auc_star": auc_star,
                "direction": direction,
                "n_a": len(a),
                "n_b": len(b),
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        print("[ERROR] No se pudo calcular AUC para ninguna feature (revisa columnas/NaNs).")
        if print_skipped and skipped:
            print("\n[SKIPPED]")
            for f, why in skipped:
                print(f" - {f}: {why}")
        return res

    res = res.sort_values("auc_star", ascending=False).reset_index(drop=True)

    # --- Helpers LaTeX ---
    def tex_escape_feature(name: str) -> str:
        # Si quieres mantenerlo en monoespaciado y evitar escapes manuales:
        # \texttt{...} en LaTeX suele tolerar '_' si está bien, pero mejor escaparlo.
        safe = name.replace("_", r"\_")
        return r"\texttt{" + safe + "}"

    def fmt(x: float) -> str:
        return f"{x:.{decimals}f}"

    # --- Tabla TOP-K ---
    top = res.head(top_k).copy()
    if table_caption_top is None:
        table_caption_top = (
            f"\\textbf{{Top-{top_k}}} de \\emph{{features}} por separabilidad univariada "
            f"(AUC$^\\star$) entre {label_a} y {label_b}."
        )

    print("\n% =====================")
    print("%  TOP FEATURES (LaTeX)")
    print("% =====================")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Feature} & $\mathbf{AUC^\star}$ & \textbf{Dirección} \\")
    print(r"\midrule")
    for _, r in top.iterrows():
        print(f"{tex_escape_feature(r['feature'])} & {fmt(r['auc_star'])} & {r['direction']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\caption{{{table_caption_top}}}")
    print(rf"\label{{{table_label_top}}}")
    print(r"\end{table}")

    # --- Tabla completa en 2 columnas (dos features por fila) ---
    if table_caption_all is None:
        table_caption_all = (
            f"Ranking completo de AUC$^\\star$ (dos \\emph{{features}} por fila) "
            f"para {label_a} vs {label_b}."
        )

    print("\n% =========================")
    print("%  ALL FEATURES (LaTeX 2col)")
    print("% =========================")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\scriptsize")
    print(r"\begin{tabular}{lcc|lcc}")
    print(r"\toprule")
    print(r"\textbf{Feature} & $\mathbf{AUC^\star}$ & \textbf{Dir.} & "
          r"\textbf{Feature} & $\mathbf{AUC^\star}$ & \textbf{Dir.}\\")
    print(r"\midrule")

    # Relleno para pares
    items = res[["feature", "auc_star", "direction"]].to_dict("records")
    if len(items) % 2 == 1:
        items.append({"feature": "", "auc_star": np.nan, "direction": ""})

    for i in range(0, len(items), 2):
        left = items[i]
        right = items[i + 1]

        lf = tex_escape_feature(left["feature"]) if left["feature"] else ""
        la = fmt(left["auc_star"]) if np.isfinite(left["auc_star"]) else ""
        ld = left["direction"]

        rf = tex_escape_feature(right["feature"]) if right["feature"] else ""
        ra = fmt(right["auc_star"]) if np.isfinite(right["auc_star"]) else ""
        rd = right["direction"]

        print(f"{lf} & {la} & {ld} & {rf} & {ra} & {rd} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\caption{{{table_caption_all}}}")
    print(rf"\label{{{table_label_all}}}")
    print(r"\end{table}")

    # --- Opcional: mostrar skips ---
    if print_skipped and skipped:
        print("\n% =====================")
        print("%  FEATURES OMITIDAS")
        print("% =====================")
        for f, why in skipped:
            print(f"% - {f}: {why}")

    return res

