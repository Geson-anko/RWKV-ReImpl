from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torchdata.datapipes.iter import IterDataPipe

from src.data.components.wiki_dataset import SPTokenizingWikiDataset


def test__init__(
    dummy_text_data_dir: Path,
    mock_sentencepieceprocessor: MagicMock,
):
    ctx_len = 10
    mock = mock_sentencepieceprocessor()

    wiki_ds = SPTokenizingWikiDataset(
        data_dirs=[str(dummy_text_data_dir)], ctx_len=ctx_len, sp_processor=mock
    )

    assert wiki_ds.data_dirs == [str(dummy_text_data_dir)]
    assert wiki_ds.ctx_len == ctx_len
    assert wiki_ds.sp_processor is mock

    assert isinstance(wiki_ds.datapipe, IterDataPipe)
    assert isinstance(wiki_ds.sp_processor, MagicMock)

    for data in wiki_ds.datapipe:
        assert data[1] == "dummy text"
        # tags are removed by Filter.


@pytest.mark.parametrize("ctx_len", [8, 9, 10])
def test__iter__(
    dummy_text_data_dir: Path,
    mock_sentencepieceprocessor: MagicMock,
    ctx_len,
):
    mock = mock_sentencepieceprocessor()  # Mock instance is always same object.
    mock.EncodeAsIds.return_value = [0, 1, 2, 3, 4, 5, 6]
    mock.pad_id.return_value = -1

    wiki_ds = SPTokenizingWikiDataset(
        data_dirs=[str(dummy_text_data_dir)], ctx_len=ctx_len, sp_processor=mock
    )

    for ids in wiki_ds:
        assert len(ids) == ctx_len
        assert isinstance(ids, torch.Tensor)
        assert ids.dtype == torch.int64
