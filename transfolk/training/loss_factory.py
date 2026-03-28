# ------------------------------
# training/loss_factory.py
# ------------------------------
from __future__ import annotations

import torch
from transfolk_config.entities.runtime_train import RuntimeTrain

class LossFactory:
    """
    Construye funciones de pérdida a partir de runtime_train.
    """

    @staticmethod
    def build(runtime_train: RuntimeTrain):
        loss_type = (runtime_train.loss or "cross_entropy").lower()

        if loss_type == "cross_entropy":
            return torch.nn.CrossEntropyLoss(ignore_index=0)

        else:
            raise ValueError(f"Unsupported loss: {runtime_train.loss}")