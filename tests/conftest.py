"""This file prepares config fixtures for other tests."""

from datetime import datetime
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, open_dict
from pyrootutils import find_root
from pytest_mock import MockerFixture
from sentencepiece import SentencePieceProcessor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

root_dir = find_root(indicator=".project-root")


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> DictConfig:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture
def dummy_text_data_dir(tmp_path: Path) -> Path:
    """Create dummy text files and return datadir.

    Returns:
        datadir: dummy data directory.
    """
    tmpdir = tmp_path / "dummy_text_data_dir"
    tmpdir.mkdir()

    for i in range(10):
        with open(tmpdir / f"dummy_text_{i}.txt", "w") as f:
            f.write(f"dummy text\n<tag {i}>")

    return tmpdir


@pytest.fixture(scope="function")
def mock_sentencepieceprocessor(mocker: MockerFixture) -> MagicMock:
    """
    Returns:
        mock: mocked sentence piece processor.
    """
    return mocker.MagicMock(spec=SentencePieceProcessor)


@pytest.fixture(scope="function")
def partial_adam_optimizer() -> partial[Adam]:
    """
    Returns:
        mock: partial instance of adam optimizer.
    """
    return partial(Adam, lr=0.001)


@pytest.fixture(scope="function")
def partial_lambda_lr_scheduler() -> partial[LambdaLR]:
    """
    Returns:
        mock: partial instance of lambda lr scheduler.
    """
    return partial(LambdaLR, lr_lambda=lambda epoch: 0.95**epoch)


@pytest.fixture(scope="function")
def mlflow_logger() -> MLFlowLogger:
    """
    Returns:
        mlflow_logger: mlflow logger.
    """
    save_dir = root_dir / "logs" / "test_mlflow_logging" / datetime.now().strftime("%Y%m%d-%H%M%S")
    return MLFlowLogger(experiment_name="test", save_dir=save_dir)
