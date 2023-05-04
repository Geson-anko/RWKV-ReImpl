"""This file contains Datasets for loading text file."""
import re
from typing import Any, Sequence

import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    IterableWrapper,
    StreamReader,
)


class SPTokenizingTextDataset(Dataset):
    """This class tokenize text file with sentencepiece and return tokenized ids. One text file is
    one data point, so the length of each data will be different.

    Note: This class can be parallelized.
    """

    def __init__(
        self,
        data_dirs: Sequence[str],
        sp_processor: spm.SentencePieceProcessor,
        max_len: int = 1024,
        ignoring_chars: str = "",
        replacement: str = "",
        list_file_recusively: bool = True,
        file_encoding: str = "utf-8",
    ):
        """Initialize SPTokeningingTextDataset.

        Args:
            data_dirs: data directories that have text files.
            sp_processor: sentencepiece processor.
            max_len: maximum length of each data. If the length of data is longer than this, it will be truncated.
            ignoring_chars: characters that will be ignored. This is a regular expression.
            replacement: replacement of ignoring characters.
            list_file_recusively: if True, list files recursively.
            file_encoding: encoding of text files.
        """
        super().__init__()
        self.data_dirs = data_dirs
        self.sp_processor = sp_processor
        self.max_len = max_len
        self.list_file_recusively = list_file_recusively
        self.file_encoding = file_encoding

        self._ignoring_chars = ignoring_chars
        self._ignoring_chars_ptrn = re.compile(ignoring_chars)
        self._replacement = replacement

        self.text_files = list(FileLister(data_dirs, recursive=list_file_recusively))

    def _read_text_file(self, file_path: Any) -> list[str]:
        """Read text from file.

        Args:
            file: path to the text file.

        Returns:
            text: contents of text file.
        """
        with open(file_path, encoding=self.file_encoding) as f:
            text = f.read()
            text = self._ignoring_chars_ptrn.sub(self._replacement, text)

            if text != "":
                return text
            else:
                raise ValueError(f"All text of {file_path} was ignored by {self._ignoring_chars}!")

    def __len__(self) -> int:
        return len(self.text_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Read text and tokenize it with sentencepiece.

        Args:
            idx: index of data.

        Returns:
            ids: tokenized ids.
        """
        text = self._read_text_file(self.text_files[idx])
        ids = self.sp_processor.EncodeAsIds(text)
        ids = ids[: self.max_len]
        ids = torch.tensor(ids, dtype=torch.int64)
        return ids
