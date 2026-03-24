
from transfolk_classifier import corpus_membership_classifier
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from music21 import converter, note, chord, meter
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "cmr10",  # usa Computer Modern integrada
    "mathtext.fontset": "cm",  # activa Computer Modern para texto matemático y ejes
})



############################################################################
# CURVA MEMBERSHIP VS TEMPERATURE (FIDELITY CURVE)
############################################################################
def Style_Fidelity_Curve(DATA_DIR, PROD_DIR, CORPUS, ALGORITHM, TIME_SIGNATURE, TONALITY, TEMPERATURES, font_size = 14, axis_size=12, show_tittle=True, show_chart=True):
    # Curva de calibración estilística (Style Fidelity Curve)
    #
    # Eje X: temperatura.
    # Eje Y: probabilidad media de pertenencia según el clasificador.
    #
    # Puede añadirse una segunda línea para “Perceptual rating” (evaluación humana), mostrando correlación.


    model_dir = rf"classifier\{CORPUS}"  # carpeta del modelo del clasificador
    globalMemberships = []
    stdMemberships = []


    for TEMPERATURE in TEMPERATURES:
        print(f"🔥 Midiendo membership global para temperatura: {TEMPERATURE}")
        new_dir = f"{PROD_DIR}/{TIME_SIGNATURE.replace('/', '_')}/{TONALITY}/{TEMPERATURE:.1f}"
        average, std = folk_membership_controlled.evaluate_model(new_dir, model_dir)
        globalMemberships.append(average)
        stdMemberships.append(std)

    # --- Gráfica ---
    plt.figure(figsize=(7, 5))
    plt.plot(TEMPERATURES, globalMemberships, marker='o', linewidth=1.5, markersize=5, color='black')

    # Añadir barras de error (desviación típica)
    plt.errorbar(
        TEMPERATURES,
        globalMemberships,
        yerr=stdMemberships,
        fmt='o',
        ecolor='gray',
        elinewidth=1,
        capsize=4,
        capthick=1,
        markersize=5,
        color='black'
    )


    # Estilo tipo paper científico
    if show_tittle:
        plt.title(f"Average Membership vs. Temperature ({TIME_SIGNATURE}, mode {TONALITY})", fontsize=font_size+2, fontweight='bold', pad=15)
    plt.xlabel("Temperature", fontsize=font_size)
    plt.ylabel("Membership", fontsize=font_size)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    plt.xticks(TEMPERATURES, [f"{t:.1f}" for t in TEMPERATURES])
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    # Bordes del gráfico más finos, estilo minimalista
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')

    # Guardar figura con alta resolución
    out_png = f"{PROD_DIR}/global_membership_vs_temperature__{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.png"
    out_svg = f"{PROD_DIR}/global_membership_vs_temperature__{CORPUS}_{ALGORITHM}_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.svg"

    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    # plt.savefig(f"{PROD_DIR}/{TIME_SIGNATURE.replace('/', '_')}/{TONALITY}/global_membership_vs_temperature_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.png", dpi=600, bbox_inches='tight')
    # plt.savefig(f"{PROD_DIR}/{TIME_SIGNATURE.replace('/', '_')}/{TONALITY}/global_membership_vs_temperature_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.svg", format='svg', bbox_inches='tight')
    if show_chart:
        plt.show()


    # Retorna datos por si se quieren analizar posteriormente
    return list(zip(TEMPERATURES, globalMemberships))



############################################################################
# Gráfica de dispersión Membership–Entropy
# (usando la entropía de tokens generados por temperatura).
############################################################################
def extract_tokens_from_musicxml(path):
    """Convierte una partitura MusicXML en una lista de tokens simples (pitches y duraciones)."""
    try:
        score = converter.parse(path)
    except Exception:
        return []

    flat = score.flat.notesAndRests
    tokens = []

    for el in flat:
        if el.isNote:
            tokens.append(f"NOTE_{el.pitch.midi}")
        elif el.isRest:
            tokens.append("REST")
        elif el.isChord:
            tokens.append(f"CHORD_{'-'.join(str(p.midi) for p in el.pitches)}")
    return tokens

def entropy_of_tokens(tokens):
    """Calcula la entropía de una secuencia de tokens (bits/token)."""
    if not tokens:
        return 0.0
    values, counts = np.unique(tokens, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def mean_entropy_for_folder(folder):
    """
    Calcula la entropía media de todas las obras en una carpeta.
    Cada archivo .musicxml o .xml se procesa individualmente.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.musicxml"))) + \
            sorted(glob.glob(os.path.join(folder, "*.xml")))

    entropies = []
    for path in paths:
        tokens = extract_tokens_from_musicxml(path)
        H = entropy_of_tokens(tokens)
        entropies.append(H)
    return float(np.mean(entropies)) if entropies else 0.0

def Membership_Entropy_Scatter_Plot(MODEL, TIME_SIGNATURE, TONALITY, PROD_DIR, TEMPERATURES, show=True):
    model_dir = rf"classifier\{MODEL}"  # carpeta del modelo del clasificador
    globalMemberships = []
    stdMemberships = []
    meanEntropies = []

    for TEMPERATURE in TEMPERATURES:
        print(f"🔥 Midiendo membership global para temperatura: {TEMPERATURE}")
        new_dir = f"{PROD_DIR}/{TIME_SIGNATURE.replace('/', '_')}/{TONALITY}/{TEMPERATURE:.1f}"
        average, std = corpus_membership_classifier.evaluate_model(new_dir, model_dir)
        globalMemberships.append(average)
        stdMemberships.append(std)

        # Cálculo de entropía media para esta temperatura usando las funciones de MusicXML
        mean_entropy = mean_entropy_for_folder(new_dir)
        meanEntropies.append(mean_entropy)

    # === Gráfica de dispersión Membership–Entropy ===
    plt.figure(figsize=(7, 5))

    # Colores según la temperatura (gradiente suave)
    colors = plt.cm.plasma((TEMPERATURES - TEMPERATURES.min()) / (TEMPERATURES.max() - TEMPERATURES.min()))

    # Conversión a arrays y limpieza de NaN o inf
    meanEntropies_arr = np.array(meanEntropies, dtype=float)
    globalMemberships_arr = np.array(globalMemberships, dtype=float)
    stdMemberships_arr = np.array(stdMemberships, dtype=float)

    mask = (
            np.isfinite(meanEntropies_arr)
            & np.isfinite(globalMemberships_arr)
            & ~np.isnan(meanEntropies_arr)
            & ~np.isnan(globalMemberships_arr)
    )
    meanEntropies_clean = meanEntropies_arr[mask]
    globalMemberships_clean = globalMemberships_arr[mask]
    stdMemberships_clean = stdMemberships_arr[mask]
    temps_clean = np.array(TEMPERATURES)[mask]

    # Dispersión con barras de error verticales
    plt.errorbar(
        meanEntropies_clean,
        globalMemberships_clean,
        yerr=stdMemberships_clean,
        fmt='o',
        markersize=6,
        color='none',
        ecolor='gray',
        elinewidth=1,
        capsize=3,
        zorder=1
    )

    # Puntos coloreados según temperatura
    plt.scatter(
        meanEntropies_clean,
        globalMemberships_clean,
        s=40 + stdMemberships_clean * 2,
        c=colors[:len(meanEntropies_clean)],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9,
        zorder=2
    )

    # Línea de tendencia y cálculo de R²
    if len(meanEntropies_clean) >= 3 and np.ptp(meanEntropies_clean) > 0:
        z = np.polyfit(meanEntropies_clean, globalMemberships_clean, 2)
        p = np.poly1d(z)
        x_fit = np.linspace(min(meanEntropies_clean), max(meanEntropies_clean), 100)
        y_fit = p(x_fit)
        plt.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=1)

        # R²
        y_pred = p(meanEntropies_clean)
        ss_res = np.sum((globalMemberships_clean - y_pred) ** 2)
        ss_tot = np.sum((globalMemberships_clean - np.mean(globalMemberships_clean)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        plt.text(
            0.05, 0.95,
            f"$R^2$ = {r2:.3f}",
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top'
        )

    # Etiquetas de temperatura
    for i, T in enumerate(temps_clean):
        plt.text(
            meanEntropies_clean[i] + 0.02,
            globalMemberships_clean[i] + 0.5,
            f"{T:.1f}",
            fontsize=8
        )

    # Estilo tipo paper científico
    plt.title(f"Global Membership vs. Mean Token Entropy ({TIME_SIGNATURE}, mode {TONALITY})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mean Token Entropy (bits per token)", fontsize=12)
    plt.ylabel("Global Membership (%)", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')


    # Guardar figura con alta resolución
    plt.savefig(f"{PROD_DIR}/{TIME_SIGNATURE.replace('/', '_')}/{TONALITY}/global_membership_entropy_scatter_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.png", dpi=600,
               bbox_inches='tight')
    plt.savefig(f"{PROD_DIR}/{TIME_SIGNATURE.replace('/', '_')}/{TONALITY}/global_membership_entropy_scatter_{TIME_SIGNATURE.replace('/', '_')}_{TONALITY}.svg", format='svg',
               bbox_inches='tight')
    if show:
        plt.show()

    # # === Gráfica de dispersión Membership–Entropy ===
    # plt.figure(figsize=(7, 5))
    #
    # # Colores según la temperatura (gradiente suave)
    # colors = plt.cm.plasma((TEMPERATURES - TEMPERATURES.min()) / (TEMPERATURES.max() - TEMPERATURES.min()))
    #
    # # Conversión a arrays y limpieza de NaN o inf
    # meanEntropies_arr = np.array(meanEntropies, dtype=float)
    # globalMemberships_arr = np.array(globalMemberships, dtype=float)
    # stdMemberships_arr = np.array(stdMemberships, dtype=float)
    #
    # mask = (
    #         np.isfinite(meanEntropies_arr)
    #         & np.isfinite(globalMemberships_arr)
    #         & ~np.isnan(meanEntropies_arr)
    #         & ~np.isnan(globalMemberships_arr)
    # )
    # meanEntropies_clean = meanEntropies_arr[mask]
    # globalMemberships_clean = globalMemberships_arr[mask]
    # stdMemberships_clean = stdMemberships_arr[mask]
    # temps_clean = np.array(TEMPERATURES)[mask]
    #
    # # Dispersión
    # plt.scatter(meanEntropies_clean, globalMemberships_clean,
    #             s=40 + stdMemberships_clean * 2,
    #             c=colors[:len(meanEntropies_clean)], edgecolor='black', linewidth=0.5, alpha=0.9)
    #
    # # Línea de tendencia solo si hay datos suficientes
    # if len(meanEntropies_clean) >= 3 and np.ptp(meanEntropies_clean) > 0:
    #     z = np.polyfit(meanEntropies_clean, globalMemberships_clean, 2)
    #     p = np.poly1d(z)
    #     x_fit = np.linspace(min(meanEntropies_clean), max(meanEntropies_clean), 100)
    #     plt.plot(x_fit, p(x_fit), color='black', linestyle='--', linewidth=1)
    #
    # # Etiquetas
    # for i, T in enumerate(temps_clean):
    #     plt.text(meanEntropies_clean[i] + 0.02, globalMemberships_clean[i] + 0.5, f"{T:.1f}", fontsize=8)
    #
    # # Estilo tipo paper científico
    # plt.title("Global Membership vs. Mean Token Entropy", fontsize=14, fontweight='bold', pad=15)
    # plt.xlabel("Mean Token Entropy (bits per token)", fontsize=12)
    # plt.ylabel("Global Membership (%)", fontsize=12)
    # plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    # plt.tick_params(axis='both', labelsize=12)
    # plt.tight_layout()
    #
    # ax = plt.gca()
    # for spine in ax.spines.values():
    #     spine.set_linewidth(0.8)
    #     spine.set_color('black')
    #
    # plt.show()


