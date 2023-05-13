from functools import partial
from typing import Any, Optional

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from mlflow import MlflowClient
from torch import Tensor
from torchmetrics import MeanMetric

from ..data.components.text_dataset import SPTokenizingTextDataset
from ..utils.generations import rwkv_generate_fixed_length


class RWKVWikiLitModule(LightningModule):
    """LightningModule for RWKV Wiki Corpus Language Modeling."""

    logger: MLFlowLogger

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: partial[torch.optim.Optimizer],
        scheduler: Optional[partial[torch.optim.lr_scheduler.LRScheduler]] = None,
        scheduler_update_frequecy: int = 1,
        monitoring_text_dataset: Optional[SPTokenizingTextDataset] = None,
        monitoring_interval: int = 1,  # Steps
        monitoring_n_samples: int = 5,
        num_generating_tokens: int = 1024,  # Per sample.
    ):
        """LightningModule for RWKV Wiki Corpus Language Modeling.

        Args:
            net: Model.
            optimizer: Optimizer (Partial instance).
            scheduler: LR Scheduler (Partial instance).
            scheduler_update_frequecy: Scheduler update frequency.
            monitoring_text_dataset: Text dataset for monitoring model performance.
            monitoring_interval: Monitoring interval in steps.
            monitoring_n_samples: Number of samples to generate.
            num_generating_tokens: Number of tokens to generate per sample.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "monitoring_text_dataset"])

        self.net = net
        self.monitoring_text_dataset = monitoring_text_dataset

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss_avg = MeanMetric()

    def forward(self, x: Tensor) -> Tensor:
        """Forward path of the model. This method is used for pure inference, i.e. when it is used
        as layer module in another model.

        Args:
            x: Input tensor. Shape (len, batch).
        """
        return self.net(x)

    def model_step(self, batch: Tensor) -> tuple[Tensor, ...]:
        """Model step. This method is used for training, validation and testing. So, batch shape is
        (len, batch) and it is not split into x and y.

        Args:
            batch: Shape (batch, len).

        Returns:
            loss: Loss tensor.
            logits: Output of the model. Shape (batch, len, vocab_size).
            preds: Predictions. Shape (batch, len).
        """
        batch = batch.T
        x, y = batch[:-1], batch[1:].flatten()
        logits = self.forward(x)  # (len, batch, vocab_size)
        loss = self.criterion(logits.view(len(y), -1), y)
        preds = torch.argmax(logits, dim=-1).T  # (batch, len)
        return loss, logits.transpose(1, 0), preds

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch: Shape (batch, len).
            batch_idx: Batch index.
        """
        if hasattr(self.net, "clear_hidden"):
            self.net.clear_hidden()

        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss_avg(loss)
        self.log("train/loss_avg", self.train_loss_avg, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called when the train batch ends.

        This method is used for logging artifacts.
        """

        if self.global_step % self.hparams.monitoring_interval == 0:
            self._log_generation_from_monitoring_texts()

    @torch.no_grad()
    def _log_generation_from_monitoring_texts(self):
        """Log generation from monitoring texts to logger.

        logger must be MLFlowLogger.
        """

        if self.monitoring_text_dataset is None:
            return

        num_samples = self.hparams.monitoring_n_samples
        if len(self.monitoring_text_dataset) < num_samples:
            num_samples = len(self.monitoring_text_dataset)

        sp_processor = self.monitoring_text_dataset.sp_processor

        log_texts = []

        for i in range(num_samples):
            if hasattr(self.net, "clear_hidden"):
                self.net.clear_hidden()

            prompt = self.monitoring_text_dataset[i].to(self.device)
            generated = rwkv_generate_fixed_length(
                self.net, prompt, self.hparams.num_generating_tokens, True
            )
            prompt_text = sp_processor.DecodeIds(prompt.cpu().tolist())
            generated_text = sp_processor.DecodeIds(generated.cpu().tolist())

            log_texts.append(f"Sample {i}\nPrompt: {prompt_text}\nGenerated: {generated_text}\n")

        mlf: MlflowClient = self.logger.experiment
        mlf.log_text(
            self.logger.run_id,
            "\n".join(log_texts),
            f"generated_texts_step{self.global_step:012d}.txt",
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""

        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss_avg",
                    "interval": "step",
                    "frequency": self.hparams.scheduler_update_frequecy,
                },
            }
        return {"optimizer": optimizer}
