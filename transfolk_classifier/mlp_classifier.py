# -*- coding: utf-8 -*-
"""
mlp_classifier.py

Clasificador MLP para el pipeline TEIMUS Profano vs Religioso.

Este módulo mantiene la integración mínima con classifier.py: expone una única
factoría que devuelve el estimador scikit-learn que se inserta dentro del mismo
Pipeline usado por el resto de clasificadores:

    SimpleImputer(strategy="median") -> StandardScaler() -> MLPClassifier

La entrada del MLP es exactamente el mismo vector tabular de features musicales
que ya utiliza el pipeline clásico; no introduce tokenización, embeddings ni
cambios en la extracción de rasgos.
"""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier


def build_mlp_classifier(random_state: int = 42) -> MLPClassifier:
    """Construye el clasificador MLP usado como algorithm_id=9.

    Diseño conservador para datasets tabulares pequeños/medianos:
        - dos capas ocultas moderadas para capturar interacciones no lineales
          entre features sin disparar demasiado la varianza;
        - regularización L2 mediante alpha;
        - early_stopping para reservar validación interna y reducir sobreajuste;
        - learning_rate adaptativo para estabilizar la convergencia.

    El escalado se hace fuera, en classifier.make_model(), para respetar la
    arquitectura común del pipeline.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        tol=1e-4,
        random_state=random_state,
    )
