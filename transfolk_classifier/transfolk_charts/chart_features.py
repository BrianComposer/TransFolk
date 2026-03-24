from transfolk_features.extract_features import  *


def _plot_feature_grid(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    feature_list: List[str],
    title: str,
    bins: int = 30,
    out_png: Optional[str] = None,
):
    feats = [f for f in feature_list if (f in df_a.columns and f in df_b.columns)]
    if not feats:
        print(f"[WARN] Grupo '{title}': ninguna feature disponible.")
        return

    feats = [
        f for f in feats
        if pd.api.types.is_numeric_dtype(df_a[f]) and pd.api.types.is_numeric_dtype(df_b[f])
    ]
    if not feats:
        print(f"[WARN] Grupo '{title}': ninguna feature numérica disponible.")
        return

    n = len(feats)

    # ---- 3 columnas siempre ----
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig = plt.figure(figsize=(5.0 * ncols, 3.6 * nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    # última fila centrada si sobran 1 o 2
    last_row_count = n % ncols
    if last_row_count == 0:
        last_row_count = ncols

    axes = []
    feat_idx = 0

    # filas completas (todas menos la última)
    for r in range(max(0, nrows - 1)):
        for c in range(ncols):
            ax = fig.add_subplot(gs[r, c])
            axes.append(ax)
            feat_idx += 1

    # última fila centrada
    r = nrows - 1
    if last_row_count == 3:
        cols = [0, 1, 2]
    elif last_row_count == 2:
        cols = [0, 2]   # centrado simétrico
    else:
        cols = [1]      # centro

    for c in cols:
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

    # --- dibujar histogramas en % ---
    for ax, feat in zip(axes, feats):
        a = df_a[feat].dropna().astype(float).values
        b = df_b[feat].dropna().astype(float).values
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        if len(a) == 0 or len(b) == 0:
            pretty = FEATURE_TITLES.get(feat, feat) if "FEATURE_TITLES" in globals() else feat
            ax.set_title(f"{pretty} ({feat}) (sin datos)")
            ax.axis("off")
            continue

        combined = np.concatenate([a, b])
        lo, hi = float(np.min(combined)), float(np.max(combined))

        # pesos: cada obra pesa 100/N => eje Y es "% de obras"
        wa = np.ones_like(a, dtype=float) * (100.0 / len(a))
        wb = np.ones_like(b, dtype=float) * (100.0 / len(b))

        if np.isclose(lo, hi):
            ax.hist(a, bins=1, alpha=0.6, label=label_a, weights=wa)
            ax.hist(b, bins=1, alpha=0.6, label=label_b, weights=wb)
        else:
            edges = np.linspace(lo, hi, bins + 1)
            ax.hist(a, bins=edges, alpha=0.6, label=label_a, weights=wa)
            ax.hist(b, bins=edges, alpha=0.6, label=label_b, weights=wb)

        pretty = FEATURE_TITLES.get(feat, feat) if "FEATURE_TITLES" in globals() else feat
        ax.set_title(f"{pretty} ({feat})")
        ax.set_ylabel("% obras")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    # ---- título + leyenda sin solape ----
    fig.suptitle(f"{title}: {label_a} vs {label_b}", y=0.995)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965)
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if out_png:
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[OK] Guardado: {out_png}")
    else:
        plt.show()


def plot_grouped_histograms(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    bins: int = 30,
    out_dir: Optional[str] = None,
):
    """
    Genera una figura por categoría definida en FEATURE_GROUPS.
    - Si out_dir != None, guarda PNGs en esa carpeta.
    - Si out_dir == None, muestra ventanas.
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for group_name, feats in FEATURE_GROUPS.items():
        out_png = None
        if out_dir is not None:
            safe_name = group_name.lower().replace(" ", "_").replace("/", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
            out_png = os.path.join(out_dir, f"hist_{safe_name}.png")

        _plot_feature_grid(
            df_a=df_a,
            df_b=df_b,
            label_a=label_a,
            label_b=label_b,
            feature_list=feats,
            title=group_name,
            bins=bins,
            out_png=out_png,
        )


# -----------------------------
# Main callable function
# -----------------------------
def compare_two_corpora_and_plot_grouped(
    corpus_a_dir: str,
    corpus_b_dir: str,
    label_a: str = "Corpus A",
    label_b: str = "Corpus B",
    bins: int = 30,
    out_dir: Optional[str] = None,   # None => show windows
    save_csv: bool = False,
    out_csv_a: Optional[str] = None,
    out_csv_b: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lee dos corpus MusicXML, extrae features por archivo, y genera histogramas agrupados por categoría.
    Devuelve (df_a, df_b).
    """
    print(f"[INFO] Cargando corpus A: {corpus_a_dir}")
    df_a = load_corpus_features(corpus_a_dir, label_a)
    print(f"[INFO] Corpus A: {len(df_a)} obras válidas")

    print(f"[INFO] Cargando corpus B: {corpus_b_dir}")
    df_b = load_corpus_features(corpus_b_dir, label_b)
    print(f"[INFO] Corpus B: {len(df_b)} obras válidas")

    plot_grouped_histograms(df_a, df_b, label_a, label_b, bins=bins, out_dir=out_dir)

    if save_csv:
        if out_csv_a is None:
            out_csv_a =  f"features_{label_a.replace(' ', '_')}.csv"
        if out_csv_b is None:
            out_csv_b = f"features_{label_b.replace(' ', '_')}.csv"
        df_a.to_csv(out_csv_a, index=False)
        df_b.to_csv(out_csv_b, index=False)
        print(f"[OK] CSVs guardados: {out_csv_a}, {out_csv_b}")

    return df_a, df_b





