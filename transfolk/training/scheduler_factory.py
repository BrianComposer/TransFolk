# ------------------------------
# training/scheduler_factory.py
# ------------------------------
from __future__ import annotations

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


class SchedulerFactory:
    """
    Construye schedulers a partir de runtime_train.
    Soporta:
    - cosine (con warmup)
    - linear (con warmup)
    - none
    """

    @staticmethod
    def build(runtime_train, optimizer, total_steps: int):
        scheduler_type = (runtime_train.scheduler or "none").lower()
        warmup_steps = runtime_train.warmup_steps or 0

        if scheduler_type == "none":
            return None

        if scheduler_type == "cosine":
            return LambdaLR(
                optimizer,
                lr_lambda=SchedulerFactory._cosine_with_warmup(
                    warmup_steps,
                    total_steps
                )
            )

        elif scheduler_type == "linear":
            return LambdaLR(
                optimizer,
                lr_lambda=SchedulerFactory._linear_with_warmup(
                    warmup_steps,
                    total_steps
                )
            )

        else:
            raise ValueError(f"Unsupported scheduler: {runtime_train.scheduler}")

    # --------------------------------------------------
    # FUNCIONES DE LR
    # --------------------------------------------------

    @staticmethod
    def _cosine_with_warmup(warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )

            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    @staticmethod
    def _linear_with_warmup(warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            return max(
                0.0,
                float(total_steps - current_step) / float(
                    max(1, total_steps - warmup_steps)
                )
            )

        return lr_lambda