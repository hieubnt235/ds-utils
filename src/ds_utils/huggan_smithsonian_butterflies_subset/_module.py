from dataclasses import dataclass
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from ._dataset import Dataset


@dataclass
class DataLoaderConfig:
    batchsize: int = 8
    num_workers: int = 8


@dataclass
class DataModuleConfig:
    train_ratio: float = 0.8
    train_dl_cfg: DataLoaderConfig = DataLoaderConfig()
    val_dl_cfg: DataLoaderConfig = DataLoaderConfig()
    test_dl_cfg: DataLoaderConfig = DataLoaderConfig()
    predict_dl_cfg: DataLoaderConfig = DataLoaderConfig()


class HugganSmithsonianButterfliesSubsetDataModule(LightningDataModule):

    def __init__(self, config: DataModuleConfig = None, **dataset_kwargs):
        self.config = config or DataModuleConfig()
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.ds_kw = dataset_kwargs

        super().__init__()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: TrainerFn) -> None:
        if not self.train_ds or not self.val_ds:
            self.train_ds, self.val_ds = Dataset.split(
                self.config.train_ratio, True, **self.ds_kw
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(
            self.train_ds,
            batch_size=self.config.train_dl_cfg.batchsize,
            num_workers=self.config.train_dl_cfg.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.val_dl_cfg.batchsize,
            num_workers=self.config.val_dl_cfg.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.test_dl_cfg.batchsize,
            num_workers=self.config.test_dl_cfg.num_workers,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_ds,
            batch_size=self.config.predict_dl_cfg.batchsize,
            num_workers=self.config.predict_dl_cfg.num_workers,
        )
