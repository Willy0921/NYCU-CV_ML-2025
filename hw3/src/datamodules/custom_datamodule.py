import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.utils.custom_instance_segmentation_dataset_utils import (
    CustomInstanceSegmentationDataset,
    segm_collate_fn,
)
from src.utils.transforms import (
    get_train_transform,
    get_val_transform,
)

from typing import Optional, Dict
import os


class CustomInstanceSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        val_split_ratio=0.2,
        seed=42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        full_dir = os.path.join(self.hparams.data_dir, "train")

        full_dataset = CustomInstanceSegmentationDataset(
            full_dir, transforms=get_train_transform()
        )

        total_len = len(full_dataset)
        val_len = int(total_len * self.hparams.val_split_ratio)
        train_len = total_len - val_len

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len]
        )

        self.val_dataset.dataset.transforms = get_val_transform()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=segm_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=segm_collate_fn,
        )
