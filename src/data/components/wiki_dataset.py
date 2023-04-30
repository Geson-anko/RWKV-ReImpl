"""This file contains DataPipes and DataLoader for wiki corpus."""
import re
from typing import Any, Iterator, Sequence

import sentencepiece as spm
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

    def __init__(self, data_dirs: Sequence[str], sp_model_path: Any, ctx_len: int):
        """Initialize SPTokenizingWikiDataset.

        Args:
            data_dirs: data directories that have wiki corpus text files.
            sp_model_path: sentencepiece model path.
            ctx_len: context length.
        """
        super().__init__()
        self.data_dirs = data_dirs
        self.sp_model_path = sp_model_path
        self.ctx_len = ctx_len

        dp = IterableWrapper(data_dirs)
        dp = FileLister(dp, recursive=True)
        dp = FileOpener(dp, encoding="utf-8")
        dp = LineReader(dp)
        ignoring_chars = re.compile(r"^<[^>]*>$\n?|^$\n?")
        dp = Filter(dp, lambda x: ignoring_chars.sub("", x[1]) != "")

        self.datapipe = dp

        self.sp_processor = spm.SentencePieceProcessor(model_file=sp_model_path)

    def __iter__(self) -> Iterator:
        """Read texts until `self.ctx_len`."""
        remained_ids = []
        for data in self.datapipe:
            _, text = data
            ids = self.sp_processor.EncodeAsIds(text)
            remained_ids += ids

            while len(remained_ids) >= self.ctx_len:
                yield remained_ids[: self.ctx_len]
                remained_ids = remained_ids[self.ctx_len :]
        if len(remained_ids) != 0:
            pad_len = self.ctx_len - len(remained_ids)
            yield remained_ids + [self.sp_processor.pad_id()] * pad_len
