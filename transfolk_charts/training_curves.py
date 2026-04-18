import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from transfolk.utils.training_logger import load_training_json
from matplotlib.ticker import MultipleLocator


def plot_training_loss(json_path, TIME_SIGNATURE, TONALITY, save_path=None, smooth=True):
    """
    Genera una gráfica profesional de la evolución de la loss durante el training.

    - json_path: ruta al JSON generado durante el entrenamiento
    - save_path: ruta para guardar la figura (opcional)
    - smooth: si True, aplica suavizado spline
    """
    # ---------------------------------------------
    # 1. Cargar datos desde JSON
    # ---------------------------------------------
    epochs, losses = load_training_json(json_path)
    epochs = np.array(epochs)
    losses = np.array(losses)

    # ---------------------------------------------
    # 2. Smoothing con spline (opcional)
    # ---------------------------------------------
    if smooth and len(epochs) > 3:
        xs = np.linspace(epochs.min(), epochs.max(), 400)
        try:
            spline = make_interp_spline(epochs, losses, k=3)
            ys = spline(xs)
        except Exception:
            xs, ys = epochs, losses  # fallback en casos degenerados
    else:
        xs, ys = epochs, losses

    # ---------------------------------------------
    # 3. Estética profesional (estilo paper)
    # ---------------------------------------------
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))

    # Curva suavizada
    ax.plot(xs, ys, color="#4A90E2", linewidth=2.8, label="Training loss")

    # Puntos originales
    ax.scatter(epochs, losses, color="#F26B38", s=45, alpha=0.9, zorder=3)

    # ---------------------------------------------
    # 4. Títulos y etiquetas
    # ---------------------------------------------
    ax.set_title("Training Loss Evolution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)

    # ---------------------------------------------
    # 5. Configuración del eje X (epochs)
    # ---------------------------------------------
    ax.set_xticks(np.arange(0, 41, 5))  # 0–40, en saltos de 5
    ax.set_xlim(0, 40)  # límite visual

    # ---------------------------------------------
    # 6. Configuración del eje Y (loss)
    # ---------------------------------------------
    # Buscar rango de la loss para definir Y ticks
    y_min = max(0, np.floor(losses.min() * 2) / 2)
    y_max = np.ceil(losses.max() * 2) / 2

    ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))  # ticks de 0.5 en 0.5
    ax.set_ylim(y_min, y_max)

    # Minor ticks (subdivisión en 5 → 0.1 cada uno)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # ---------------------------------------------
    # 7. Grid y estética final
    # ---------------------------------------------
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()

    # ---------------------------------------------
    # 6. Guardar la figura si se pide
    # ---------------------------------------------

    if save_path:
        plt.savefig(f"training_curve_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.png", dpi=600,
                    bbox_inches='tight')

        plt.savefig(f"training_curve_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.svg", format='svg',
                    bbox_inches='tight')
        print(f"Saved loss curve figure at: {save_path}")

    plt.show()







def plot_training_loss_all(charts_dir, train_dir, corpus, algorithm, smooth=True, font_size = 14, axis_size=12, show_tittle=True, show_chart=True):


    json_files = sorted(glob.glob(f"{train_dir}/training_loss_{corpus}_*.json"))
    if not json_files:
        raise RuntimeError("No JSON files found")

    # Paleta Paul Tol – Bright + Muted (estética de artículo científico)
    colors = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#DDCC77", "#CC6677", "#AA4499", "#882255"
    ]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))

    global_min = 1e9
    global_max = -1e9
    color_idx = 0

    for jf in json_files:
        base = os.path.basename(jf)
        tag = base.replace("training_loss_", "").replace(".json", "")
        parts = tag.split("_")

        if len(parts) != 4:
            raise RuntimeError(f"Invalid filename format: {base}")

        _, tonality, num, den = parts
        label = f"{tonality} · {num}/{den}"

        epochs, losses = load_training_json(jf)
        epochs = np.array(epochs, float)
        losses = np.array(losses, float)

        if smooth and len(epochs) > 3:
            xs = np.linspace(epochs.min(), epochs.max(), 400)
            try:
                spline = make_interp_spline(epochs, losses, k=3)
                ys = spline(xs)
            except Exception:
                xs, ys = epochs, losses
        else:
            xs, ys = epochs, losses

        c = colors[color_idx % len(colors)]
        ls = linestyles[color_idx % len(linestyles)]

        # Líneas limpias
        ax.plot(xs, ys, color=c, linewidth=1.8, linestyle=ls, label=label)

        # Puntos muy pequeños estilo científico
        ax.scatter(epochs, losses, color=c, s=10, alpha=0.85, zorder=3)

        global_min = min(global_min, losses.min())
        global_max = max(global_max, losses.max())

        color_idx += 1

    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_ylabel("Loss", fontsize=font_size)
    if show_tittle:
        ax.set_title(f"Training Loss Comparison ({corpus}, {algorithm})", fontsize=font_size, fontweight="bold", pad=12)

    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_xlim(0, 40)

    y_min = max(0, np.floor(global_min * 2) / 2)
    y_max = np.ceil(global_max * 2) / 2

    # Tamaño de los números de los ejes
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=axis_size
    )

    ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(False, which="major", linestyle="--", linewidth=0.2, alpha=0.2)
    ax.grid(False, which="minor", linestyle=":", linewidth=0.2, alpha=0.1)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.legend(frameon=False, fontsize=font_size-2, loc="upper right")
    plt.tight_layout()

    out_png = f"{charts_dir}/training_curve_todos_{corpus}_{algorithm}.png"
    out_svg = f"{charts_dir}/training_curve_todos_{corpus}_{algorithm}.svg"

    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')

    if show_chart:
        plt.show()




def plot_training_loss_all_paper_old(
    charts_dir, train_dir, corpus, algorithm,
    smooth=True, font_size=14, axis_size=12,
    show_tittle=True, show_chart=True
):

    import glob
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    from matplotlib.ticker import MultipleLocator

    json_files = sorted(glob.glob(f"{train_dir}/training_loss_{corpus}_*.json"))
    if not json_files:
        raise RuntimeError("No JSON files found")

    colors = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#DDCC77", "#CC6677", "#AA4499", "#882255"
    ]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))

    global_min = 1e9
    global_max = -1e9
    color_idx = 0

    for jf in json_files:
        base = os.path.basename(jf)
        tag = base.replace("training_loss_", "").replace(".json", "")
        parts = tag.split("_")

        if len(parts) != 4:
            raise RuntimeError(f"Invalid filename format: {base}")

        _, tonality, num, den = parts
        label = f"{tonality} · {num}/{den}"

        epochs, losses = load_training_json(jf)
        epochs = np.array(epochs, float)
        losses = np.array(losses, float)

        if smooth and len(epochs) > 3:
            xs = np.linspace(epochs.min(), epochs.max(), 400)
            try:
                spline = make_interp_spline(epochs, losses, k=3)
                ys = spline(xs)
            except Exception:
                xs, ys = epochs, losses
        else:
            xs, ys = epochs, losses

        c = colors[color_idx % len(colors)]
        ls = linestyles[color_idx % len(linestyles)]

        ax.plot(xs, ys, color=c, linewidth=1.8, linestyle=ls, label=label)
        ax.scatter(epochs, losses, color=c, s=10, zorder=3)

        global_min = min(global_min, losses.min())
        global_max = max(global_max, losses.max())

        color_idx += 1

    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_ylabel("Loss", fontsize=font_size)

    if show_tittle:
        ax.set_title(
            f"Training Loss Comparison ({corpus}, {algorithm})",
            fontsize=font_size,
            fontweight="bold",
            pad=12
        )

    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_xlim(0, 40)

    y_min = max(0, np.floor(global_min * 2) / 2)
    y_max = np.ceil(global_max * 2) / 2

    ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # ===== TICKS Y EJES EN NEGRO (como quieres) =====
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=axis_size,
        color="black",
        labelcolor="black",
        width=0.8,
        length=3
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # ===== GRID SUAVE EN GRIS CLARO (SIN ALPHA) =====
    major_color = "#e6e6e6"  # antes #dddddd
    minor_color = "#f2f2f2"  # antes #eeeeee
    ax.grid(True, which="major",
            linestyle="--",
            linewidth=0.4,
            color=major_color)

    ax.grid(True, which="minor",
            linestyle=":",
            linewidth=0.3,
            color=minor_color)

    ax.legend(frameon=False, fontsize=font_size-2, loc="upper right")

    plt.tight_layout()

    out_png = f"{charts_dir}/training_curve_todos_{corpus}_{algorithm}.png"
    out_eps = f"{charts_dir}/training_curve_todos_{corpus}_{algorithm}.eps"

    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_eps, format='eps', bbox_inches='tight')

    if show_chart:
        plt.show()


def plot_training_loss_all_paper(
    charts_dir,
    train_dir,
    architectures_names,
    corpus,
    algorithm,
    smooth=True,
    font_size=14,
    axis_size=12,
    show_tittle=True,
    show_chart=True
):
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    from matplotlib.ticker import MultipleLocator

    # =========================
    # Estilo
    # =========================
    plt.style.use("tableau-colorblind10")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))

    global_min = float("inf")
    global_max = float("-inf")

    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    # =========================
    # Loop por arquitecturas (orden garantizado)
    # =========================
    for idx, arch_name in enumerate(architectures_names):

        pattern = os.path.join(
            train_dir,
            arch_name,
            corpus,
            algorithm,
            f"training_loss_{corpus}_*.json"
        )

        matches = sorted(glob.glob(pattern))

        if not matches:
            print(f"[WARN] No hay JSON para {arch_name}")
            continue

        # Si hay varios, cogemos el más reciente
        jf = max(matches, key=os.path.getmtime)

        epochs, losses = load_training_json(jf)
        epochs = np.array(epochs, dtype=float)
        losses = np.array(losses, dtype=float)

        if len(epochs) == 0:
            continue

        # =========================
        # Suavizado
        # =========================
        if smooth and len(epochs) > 3:
            xs = np.linspace(epochs.min(), epochs.max(), 400)
            try:
                spline = make_interp_spline(epochs, losses, k=3)
                ys = spline(xs)
            except Exception:
                xs, ys = epochs, losses
        else:
            xs, ys = epochs, losses

        ls = linestyles[idx % len(linestyles)]

        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            linestyle=ls,
            label=arch_name  # ← AQUÍ ya usas directamente la arquitectura
        )

        ax.scatter(
            epochs,
            losses,
            s=10,
            zorder=3
        )

        global_min = min(global_min, losses.min())
        global_max = max(global_max, losses.max())

    # =========================
    # Labels
    # =========================
    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_ylabel("Loss", fontsize=font_size)

    if show_tittle:
        ax.set_title(
            f"Training Loss Comparison ({corpus}, {algorithm})",
            fontsize=font_size,
            fontweight="bold",
            pad=12
        )

    # =========================
    # Ejes
    # =========================
    ax.set_xticks(np.arange(0, 41, 5))
    ax.set_xlim(0, 40)

    if np.isfinite(global_min) and np.isfinite(global_max):
        y_min = max(0, np.floor(global_min * 2) / 2)
        y_max = np.ceil(global_max * 2) / 2

        ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # =========================
    # Estética
    # =========================
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=axis_size,
        color="black",
        labelcolor="black",
        width=0.8,
        length=3
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.grid(True, which="major", linestyle="--", linewidth=0.4, color="#e6e6e6")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="#f2f2f2")

    ax.legend(frameon=False, fontsize=font_size - 2, loc="upper right")

    plt.tight_layout()

    os.makedirs(charts_dir, exist_ok=True)

    out_png = os.path.join(
        charts_dir,
        f"training_curve_todos_{corpus}_{algorithm}.png"
    )
    out_eps = os.path.join(
        charts_dir,
        f"training_curve_todos_{corpus}_{algorithm}.eps"
    )

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_eps, format="eps", bbox_inches="tight")

    if show_chart:
        plt.show()
    else:
        plt.close(fig)