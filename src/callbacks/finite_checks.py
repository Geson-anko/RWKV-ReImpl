"""This module implements the finite checks callback."""
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


class FiniteChecks(Callback):
    """This callback checks for finite values in the model.

    Log True if the model contains any infinite values.
    """

    def __init__(self, check_interval: int, check_on_epoch_end: bool = False) -> None:
        """Initialize the callback.

        Args:
            check_interval (int): The interval at which to check for finite values. (Step)
            check_on_epoch_end (bool): Whether to check for finite values at the end of each epoch.
        """
        super().__init__()

        self.check_interval = check_interval
        self.check_on_epoch_end = check_on_epoch_end

    def check_infinite_parameters(self, model: LightningModule) -> bool:
        """Check if the parameters of the model contain any infinite values.

        Args:
            model (LightningModule): The model to check.

        Returns:
            bool: True if the model contains any infinite values.
        """
        for param in model.parameters():
            if torch.isfinite(param).all():
                return False
        return True

    def log_check_infinite(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the result of the check for infinite values."""
        trainer.logger.log_metrics(
            {"is_finite": not self.check_infinite_parameters(pl_module)}, step=trainer.global_step
        )

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args: Any, **kwds: Any
    ) -> None:
        """Check for finite values at the end of each training batch."""
        if trainer.global_step % self.check_interval == 0:
            self.log_check_infinite(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Check for finite values at the end of each training epoch."""
        if self.check_on_epoch_end:
            self.log_check_infinite(trainer, pl_module)
