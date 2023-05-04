from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.data.components.text_dataset import SPTokenizingTextDataset


def test__init__(
    dummy_text_data_dir: Path,
    mock_sentencepieceprocessor: MagicMock,
):
    max_len = 10
    mock = mock_sentencepieceprocessor()

    text_ds = SPTokenizingTextDataset(
        data_dirs=[str(dummy_text_data_dir)],
        max_len=max_len,
        sp_processor=mock,
        ignoring_chars="",
        list_file_recusively=False,
    )

    assert text_ds.data_dirs == [str(dummy_text_data_dir)]
    assert text_ds.max_len == max_len
    assert text_ds.sp_processor is mock
    assert text_ds._ignoring_chars == ""

    assert text_ds.list_file_recusively is False
    assert text_ds.file_encoding == "utf-8"

    assert isinstance(text_ds.text_files, list)
    assert len(text_ds.text_files) == 10


def test__read_text_file(
    dummy_text_data_dir: Path,
    mock_sentencepieceprocessor: MagicMock,
):
    mock = mock_sentencepieceprocessor()

    text_ds = SPTokenizingTextDataset(
        data_dirs=[str(dummy_text_data_dir)],
        max_len=10,
        sp_processor=mock,
        ignoring_chars=r"\n",
        replacement=" ",
    )

    text = text_ds._read_text_file(dummy_text_data_dir / "dummy_text_0.txt")
    assert text == "dummy text <tag 0>"


def test__len__(
    dummy_text_data_dir: Path,
    mock_sentencepieceprocessor: MagicMock,
):
    mock = mock_sentencepieceprocessor()

    text_ds = SPTokenizingTextDataset(
        data_dirs=[str(dummy_text_data_dir)],
        max_len=10,
        sp_processor=mock,
        ignoring_chars=r"\n",
        replacement=" ",
    )

    assert len(text_ds) == len(text_ds.text_files)


@pytest.mark.parametrize("max_len", [4, 5, 6])
def test__getitem__(dummy_text_data_dir: Path, mock_sentencepieceprocessor: MagicMock, max_len):
    mock = mock_sentencepieceprocessor()
    mock.EncodeAsIds.return_value = [0, 1, 2, 3, 4, 5, 6]

    text_ds = SPTokenizingTextDataset(
        data_dirs=[str(dummy_text_data_dir)],
        max_len=max_len,
        sp_processor=mock,
        ignoring_chars=r"\n",
        replacement=" ",
    )

    out = text_ds[0]
    assert len(out) == max_len
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.int64
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 6][:max_len]
