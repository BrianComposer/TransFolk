import json
import os


# ----------------------------------------------------------
# 1. Guardar un epoch y su loss en un JSON acumulativo
# ----------------------------------------------------------
def save_loss_to_json(filepath, epoch, loss):
    """
    Registra la loss de un epoch en un archivo JSON acumulativo.

    Estructura JSON:
    {
        "epochs": [1, 2, 3, ...],
        "losses": [0.45, 0.39, 0.33, ...]
    }
    """
    # Crear archivo si no existe
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"epochs": [], "losses": []}
    else:
        data = {"epochs": [], "losses": []}

    # Añadir valores
    data["epochs"].append(int(epoch))
    data["losses"].append(float(loss))

    # Guardar
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ----------------------------------------------------------
# 2. Cargar datos del JSON (para graficar u otros usos)
# ----------------------------------------------------------
def load_training_json(filepath):
    """
    Lee un archivo JSON de training y devuelve (epochs, losses).

    Retorna:
        epochs: lista de ints
        losses: lista de floats

    Valida y ordena los datos automáticamente.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training log not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data.get("epochs", [])
    losses = data.get("losses", [])

    if len(epochs) != len(losses):
        raise ValueError("El archivo JSON está corrupto: epochs y losses no coinciden.")

    # Ordenar por epoch por si el archivo no está en orden
    combined = sorted(zip(epochs, losses), key=lambda x: x[0])
    epochs_sorted = [int(e) for e, _ in combined]
    losses_sorted = [float(l) for _, l in combined]

    return epochs_sorted, losses_sorted
