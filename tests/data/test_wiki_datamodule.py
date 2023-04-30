from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.data.wiki_datamodule import WikiDataModule


@pytest.mark.parametrize("batch_size", [2, 4])
def test_wiki_datamodule(
    batch_size: int,
    dummy_text_data_dir: Path,
    mock_sentencepieceprocessor: MagicMock,
):
    mock = mock_sentencepieceprocessor()
    mock.EncodeAsIds.return_value = [0, 1, 2, 3]
    mock.pad_id.return_value = -1

    dm = WikiDataModule(
        data_dirs=[str(dummy_text_data_dir)],
        sp_model_path="dummy_sp_model_path",
        batch_size=batch_size,
        ctx_len=8,
    )

    assert dm.data_train is None
    assert dm.hparams.data_dirs == [str(dummy_text_data_dir)]
    assert dm.hparams.sp_model_path == "dummy_sp_model_path"
    assert dm.hparams.batch_size == batch_size
    assert dm.hparams.ctx_len == 8

    dm.setup()
    assert dm.data_train
    assert isinstance(dm.data_train, IterableDataset)
    assert isinstance(dm.train_dataloader(), DataLoader)

    batch = next(iter(dm.train_dataloader()))
    assert batch.shape == (batch_size, 8)
    assert batch.dtype == torch.int64
