# ------------------------------
# training/optimizer_factory.py
# ------------------------------
from __future__ import annotations

import torch
from transfolk_config.entities.runtime_train import RuntimeTrain

class OptimizerFactory:
    """
    Construye optimizadores a partir de runtime_train.
    """

    @staticmethod
    def build(runtime_train: RuntimeTrain, model):
        opt_type = (runtime_train.optimizer or "adamw").lower()

        lr = runtime_train.learning_rate or 1e-4
        weight_decay = runtime_train.weight_decay or 0.0

        if opt_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        elif opt_type == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        elif opt_type == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )

        else:
            raise ValueError(f"Unsupported optimizer: {runtime_train.optimizer}")