import pytest
import torch
from lightning import LightningModule, Trainer
from pytest_mock import MockerFixture
from torch import Tensor, nn

from src.callbacks.finite_checks import FiniteChecks


class MockModel(LightningModule):
    def __init__(self, param: torch.Tensor) -> None:
        super().__init__()
        self.param = nn.Parameter(param)


@pytest.fixture
def mock_trainer(mocker: MockerFixture):
    trainer = mocker.MagicMock(spec=Trainer)
    trainer.global_step = 0
    trainer.logger = mocker.MagicMock()
    return trainer


@pytest.mark.parametrize(
    "check_interval, check_on_epoch_end",
    [
        (1, False),
        (2, True),
    ],
)
def test__init__(check_interval, check_on_epoch_end):
    callback = FiniteChecks(check_interval, check_on_epoch_end)

    assert callback.check_interval == check_interval
    assert callback.check_on_epoch_end == check_on_epoch_end


@pytest.mark.parametrize(
    "param, expected_result",
    [
        (torch.tensor([1.0, 2.0, 3.0]), False),
        (torch.tensor([1.0, float("inf"), 3.0]), True),
        (torch.tensor([1.0, float("-inf"), 3.0]), True),
        (torch.tensor([1.0, float("nan"), 3.0]), True),
    ],
)
def test_check_infinite_parameters(param: Tensor, expected_result: bool):
    model = MockModel(param)
    callback = FiniteChecks(1)

    assert callback.check_infinite_parameters(model) is expected_result


def test_on_train_batch_end(mock_trainer):
    model = MockModel(torch.tensor([1.0, 2.0, 3.0]))
    callback = FiniteChecks(1)

    callback.on_train_batch_end(mock_trainer, model)

    # Assert log_metrics was called once with expected arguments
    mock_trainer.logger.log_metrics.assert_called_once_with(
        {"is_finite": True}, step=mock_trainer.global_step
    )


def test_on_train_epoch_end(mock_trainer):
    model = MockModel(torch.tensor([1.0, 2.0, 3.0]))
    callback = FiniteChecks(1, True)

    callback.on_train_epoch_end(mock_trainer, model)

    # Assert log_metrics was called once with expected arguments
    mock_trainer.logger.log_metrics.assert_called_once_with(
        {"is_finite": True}, step=mock_trainer.global_step
    )
