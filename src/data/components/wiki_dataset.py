"""This file contains DataPipes and DataLoader for wiki corpus."""
import re
from typing import Any, Iterator, Sequence

import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    Filter,
    IterableWrapper,
    LineReader,
)


class SPTokenizingWikiDataset(IterableDataset):
    """IterableDataset for tokenizing wiki corpus with sentencepiece.

    Note: This class can not be parallelized.
    """

    def __init__(
        self, data_dirs: Sequence[str], ctx_len: int, sp_processor: spm.SentencePieceProcessor
    ):
        """Initialize SPTokenizingWikiDataset.

        Args:
            data_dirs: data directories that have wiki corpus text files.
            ctx_len: context length.
            sp_processor: sentencepiece processor.
        """
        super().__init__()
        self.data_dirs = data_dirs
        self.ctx_len = ctx_len
        self.sp_processor = sp_processor

        dp = IterableWrapper(data_dirs)
        dp = FileLister(dp, recursive=True)
        dp = FileOpener(dp, encoding="utf-8")
        dp = LineReader(dp)
        self._ignoring_chars_ptrn = re.compile(r"^<[^>]*>$\n?|^$\n?")
        
        dp = Filter(dp, self._filter_func)

        self.datapipe = dp

    def _filter_func(self, data: tuple[Any, str]) -> bool:
        """Filter function for filtering empty lines."""
        _, text = data
        if self._ignoring_chars_ptrn.sub("", text) != "":
            return True
        else:
            return False

    def __iter__(self) -> Iterator:
        """Read texts until `self.ctx_len`."""
        remained_ids = []
        for data in self.datapipe:
            _, text = data
            ids = self.sp_processor.EncodeAsIds(text)
            remained_ids += ids

            while len(remained_ids) >= self.ctx_len:
                yield torch.tensor(remained_ids[: self.ctx_len], dtype=torch.int64)
                remained_ids = remained_ids[self.ctx_len :]
        if len(remained_ids) != 0:
            pad_len = self.ctx_len - len(remained_ids)
            out = remained_ids + [self.sp_processor.pad_id()] * pad_len
            yield torch.tensor(out, dtype=torch.int64)
