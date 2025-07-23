from dataclasses import dataclass, field

import torch
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from ._dataset import Dataset, SampleType


@dataclass
class DataLoaderConfig:
    batchsize: int = 8
    num_workers: int = 8


@dataclass
class DataModuleConfig:
    train_ratio: float = 0.8
    train_dl_cfg: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    val_dl_cfg: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    test_dl_cfg: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    predict_dl_cfg: DataLoaderConfig = field(default_factory=DataLoaderConfig)


class HugganSmithsonianButterfliesSubsetDataModule(LightningDataModule):

    def __init__(self, config: DataModuleConfig = None, **dataset_kwargs):
        self.config = config or DataModuleConfig()

        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.ds_kw = dataset_kwargs

        def collate_fn(samples: list[SampleType]):
            t3 = []
            t4 = []
            for sample in samples:
                if sample["images"].dim() == 3:
                    t3.append(sample["images"])
                else:
                    t4.append(sample["images"])
            t4.append(torch.stack(t3, 0))
            return torch.cat(t4, 0)

        self.collate_fn = collate_fn

        super().__init__()

    def prepare_data(self) -> None:
        self.train_ds, self.val_ds = Dataset.split(
            self.config.train_ratio, True, **self.ds_kw
        )

    def setup(self, stage: TrainerFn) -> None:
        assert isinstance(self.train_ds, Dataset)
        assert isinstance(self.val_ds, Dataset)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(
            self.train_ds,
            batch_size=self.config.train_dl_cfg.batchsize,
            num_workers=self.config.train_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.val_dl_cfg.batchsize,
            num_workers=self.config.val_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.test_dl_cfg.batchsize,
            num_workers=self.config.test_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.predict_dl_cfg.batchsize,
            num_workers=self.config.predict_dl_cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        return dl


DataModule = HugganSmithsonianButterfliesSubsetDataModule
