from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch.loggers import MLFlowLogger
from pytest_mock import MockerFixture
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MeanMetric

from src.data.components.text_dataset import SPTokenizingTextDataset
from src.models.components.rwkv_lang import RWKV, RWKVLang
from src.models.rwkv_wiki_module import RWKVWikiLitModule


@pytest.fixture
def rwkv_module(
    partial_adam_optimizer: partial[Adam],
    partial_lambda_lr_scheduler: partial[LambdaLR],
    mock_sentencepieceprocessor: MagicMock,
    dummy_text_data_dir: Path,
) -> RWKVWikiLitModule:
    dim = 32
    depth = 1
    vocab_size = 32

    net = RWKVLang(
        RWKV(dim, depth=depth),
        dim=dim,
        vocab_size=vocab_size,
    )

    mock_sp = mock_sentencepieceprocessor()
    mock_sp.EncodeAsIds.return_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mock_sp.DecodeIds.return_value = "dummy text"

    ds = SPTokenizingTextDataset(
        data_dirs=[str(dummy_text_data_dir)],
        sp_processor=mock_sp,
        max_len=10,
    )

    return RWKVWikiLitModule(
        net=net,
        optimizer=partial_adam_optimizer,
        scheduler=partial_lambda_lr_scheduler,
        scheduler_update_frequecy=1,
        monitoring_text_dataset=ds,
        monitoring_interval=8,
        monitoring_n_samples=10,
        num_generating_tokens=32,
    )


def test__init__(
    rwkv_module: RWKVWikiLitModule,
    partial_adam_optimizer: partial[Adam],
    partial_lambda_lr_scheduler: partial[LambdaLR],
):
    assert isinstance(rwkv_module.net, RWKVLang)
    assert rwkv_module.hparams.optimizer == partial_adam_optimizer
    assert rwkv_module.hparams.scheduler == partial_lambda_lr_scheduler
    assert rwkv_module.hparams.scheduler_update_frequecy == 1
    assert isinstance(rwkv_module.monitoring_text_dataset, SPTokenizingTextDataset)
    assert rwkv_module.hparams.monitoring_interval == 8
    assert rwkv_module.hparams.monitoring_n_samples == 10
    assert rwkv_module.hparams.num_generating_tokens == 32

    assert isinstance(rwkv_module.criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(rwkv_module.train_loss_avg, MeanMetric)


@pytest.mark.parametrize(
    """
    batch_size, ctx_len
    """,
    [
        (8, 32),
        (16, 8),
        (64, 4),
    ],
)
def test_model_step(rwkv_module: RWKVWikiLitModule, batch_size: int, ctx_len: int):
    rwkv_module.net.clear_hidden()
    vs = rwkv_module.net.vocab_size
    batch = torch.randint(0, vs, (batch_size, ctx_len + 1))
    loss, logits, preds, ce_loss, l2_loss = rwkv_module.model_step(batch)

    assert loss.shape == ()
    assert logits.shape == (batch_size, ctx_len, vs)
    assert preds.shape == (batch_size, ctx_len)
    assert ce_loss.shape == ()
    assert l2_loss.shape == ()
    assert loss == ce_loss + l2_loss * rwkv_module.hparams.l2_loss_factor

    loss.backward()


def test_training_step(rwkv_module: RWKVWikiLitModule, mocker: MockerFixture):
    log = mocker.spy(rwkv_module, "log")

    vs = rwkv_module.net.vocab_size
    batch = torch.randint(0, vs, (8, 32 + 1))
    loss = rwkv_module.training_step(batch, 0)

    assert loss.shape == ()
    loss.backward()

    log.assert_called()


@pytest.mark.parametrize("global_step", [0, 1])
def test_on_train_batch_end(
    rwkv_module: RWKVWikiLitModule,
    global_step: int,
    mocker: MockerFixture,
    mlflow_logger: MLFlowLogger,
):
    _log_generation_from_monitoring_texts = mocker.spy(
        rwkv_module, "_log_generation_from_monitoring_texts"
    )
    mock_global_step = mocker.patch.object(
        RWKVWikiLitModule,
        "global_step",
        new_callable=mocker.PropertyMock(return_value=global_step),
    )
    mock_logger = mocker.patch.object(
        RWKVWikiLitModule, "logger", new_callable=mocker.PropertyMock(return_value=mlflow_logger)
    )
    rwkv_module.on_train_batch_end(None, None, None)

    if global_step % rwkv_module.hparams.monitoring_interval == 0:
        _log_generation_from_monitoring_texts.assert_called()
    else:
        _log_generation_from_monitoring_texts.assert_not_called()


def test__log_generation_from_monitoring_texts(
    rwkv_module, mocker: MockerFixture, mlflow_logger: MLFlowLogger
):
    mocker.patch.object(
        RWKVWikiLitModule, "logger", new_callable=mocker.PropertyMock(return_value=mlflow_logger)
    )
    rwkv_module._log_generation_from_monitoring_texts()
    # See output directory: `/workspace/logs/test_mlflow_logging`


def test_configure_optimizers(
    rwkv_module: RWKVWikiLitModule,
):
    opt_dict = rwkv_module.configure_optimizers()
    assert isinstance(opt_dict["optimizer"], Adam)
    assert isinstance(opt_dict["lr_scheduler"]["scheduler"], LambdaLR)
    assert opt_dict["lr_scheduler"]["interval"] == "step"
    assert opt_dict["lr_scheduler"]["frequency"] == 1
    assert opt_dict["lr_scheduler"]["monitor"] == "train/loss_avg"
