from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class NaNStopping(Callback):
    """This callback stops training if the loss contains any NaN values."""

    def __init__(self, check_interval: int, monitoring_metrics: str = "loss") -> None:
        """Initialize the callback.

        Args:
            monitoring_metrics (str): The name of the metric to monitor.
            check_interval (int): The interval at which to check for NaN values. (Step)
        """
        super().__init__()

        self.monitoring_metrics = monitoring_metrics
        self.check_interval = check_interval

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.check_interval == 0:
            if torch.isnan(outputs[self.monitoring_metrics]).any():
                trainer.should_stop = True
                print("NaN detected. Stopping training.")
                return
