
import numpy as np
import matplotlib.pyplot as plt


def radar_expert_validation():


    metrics = [
        "Stylistic fidelity",
        "Melodic coherence",
        "Rhythmic naturalness",
        "Playability / singability",
        "Aesthetic interest"
    ]

    corpus_means = np.array([4.8, 4.5, 4.6, 4.7, 4.4])
    generated_means = np.array([4.1, 4.0, 3.8, 3.9, 4.0])

    # Cerrar correctamente los valores
    values_corpus = np.concatenate((corpus_means, [corpus_means[0]]))
    values_generated = np.concatenate((generated_means, [generated_means[0]]))

    # Calcular los ángulos y cerrar el círculo
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N + 1, endpoint=True)

    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    color_corpus = "#4A90E2"  # azul pastel
    color_generated = "#F26B38"  # coral suave

    # Polígonos cerrados
    ax.plot(angles, values_corpus, linewidth=2.4, color=color_corpus, label="Corpus")
    ax.fill(angles, values_corpus, color=color_corpus, alpha=0.25)

    ax.plot(angles, values_generated, linewidth=2.4, color=color_generated, label="Generated")
    ax.fill(angles, values_generated, color=color_generated, alpha=0.25)

    # Etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, fontweight='medium')

    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_ylim(0, 5)

    # Estética
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    ax.patch.set_facecolor('white')
    ax.spines['polar'].set_visible(False)

    ax.set_title("Expert Evaluation Radar Chart\n(corpus vs generated)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1),
              frameon=False, fontsize=11)

    plt.tight_layout()
    plt.show()

