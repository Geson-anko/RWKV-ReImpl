from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from .components.wiki_dataset import SPTokenizingWikiDataset


class WikiDataModule(LightningDataModule):
    """DataModule for Wikipedia dataset.
    Note: This class can not be parallelized. So DataLoader `num_workers` is always 0 or 1.

    Return Info:
        - shape: (batch_size, ctx_len)
        - dtype: torch.int64
    """

    def __init__(
        self,
        dataset: SPTokenizingWikiDataset,
        batch_size: int = 64,
        pin_memory: bool = False,
        num_workers: int = 0,
    ):
        """Initialize WikiDataModule.

        Args:
            data_dirs: data directories that have wiki corpus text files.
            sp_model_path: sentencepiece model path.
            ctx_len: context length.
            batch_size: batch size.
            pin_memory: whether to pin memory.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,  # Can not shuffle because iterable dataset.
        )
