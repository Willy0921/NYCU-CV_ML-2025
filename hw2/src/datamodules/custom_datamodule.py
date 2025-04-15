import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.utils.custom_dataset_utils import CustomObjectDetectionDataset, collate_fn
from src.utils.transforms import (
    get_train_transform,
    get_val_transform,
)

from typing import Optional, Dict
import os


class CustomObjectDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_split: str = "train",
        val_split: str = "valid",
        test_split: str = "test",
        annotation_filename_pattern: str = "{split}.json",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[CustomObjectDetectionDataset] = None
        self.val_dataset: Optional[CustomObjectDetectionDataset] = None
        self.test_dataset: Optional[CustomObjectDetectionDataset] = None

    def _get_data_path(self, split: str, require_ann: bool = True) -> Dict[str, str]:
        """Helper function to get image folder and annotation file path."""
        if not split:
            return None
        img_folder = os.path.join(self.hparams.data_root, split)
        ann_file = os.path.join(
            self.hparams.data_root,
            self.hparams.annotation_filename_pattern.format(split=split),
        )
        # Check if the paths exist
        if not os.path.isdir(img_folder):
            raise FileNotFoundError(
                f"Image folder not found for split '{split}': {img_folder}"
            )

        if require_ann and not os.path.isfile(ann_file):
            raise FileNotFoundError(
                f"Annotation file not found for split '{split}': {ann_file}"
            )

        return {
            "img_folder": img_folder,
            "ann_file": ann_file if os.path.isfile(ann_file) else None,
        }

    def setup(self, stage: Optional[str] = None):
        # Determine the stage and set up datasets accordingly
        if stage == "fit" or stage is None:
            train_paths = self._get_data_path(
                self.hparams.train_split, require_ann=True
            )
            if train_paths:
                self.train_dataset = CustomObjectDetectionDataset(
                    img_folder=train_paths["img_folder"],
                    ann_file=train_paths["ann_file"],
                    transforms=get_train_transform(),
                    is_predict=False,
                )
            else:
                print(
                    f"Could not set up train dataset for split: {self.hparams.train_split}"
                )

            val_paths = self._get_data_path(self.hparams.val_split, require_ann=True)
            if val_paths:
                self.val_dataset = CustomObjectDetectionDataset(
                    img_folder=val_paths["img_folder"],
                    ann_file=val_paths["ann_file"],
                    transforms=get_val_transform(),
                    is_predict=False,
                )
            else:
                print(
                    f"Could not set up validation dataset for split: {self.hparams.val_split}"
                )

        if stage == "test" or stage == "predict" or stage is None:
            test_paths = self._get_data_path(self.hparams.test_split, require_ann=False)
            if test_paths:
                self.test_dataset = CustomObjectDetectionDataset(
                    img_folder=test_paths["img_folder"],
                    ann_file=test_paths["ann_file"],
                    transforms=get_val_transform(),  # same as validation
                    is_predict=True,
                )
            else:
                print(
                    f"Could not set up test dataset for split: {self.hparams.test_split}"
                )

    def train_dataloader(self):
        if not self.train_dataset:
            raise RuntimeError(
                "Train dataset not initialized. Check data paths and setup stage."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,  # **important**
        )

    def val_dataloader(self):
        if not self.val_dataset:
            raise RuntimeError(
                "Validation dataset not initialized. Check data paths and setup stage."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,  # **important**
        )

    def test_dataloader(self):
        if not self.test_dataset:
            print(
                "Test dataset not found, attempting to use validation set for testing."
            )
            if not self.val_dataset:
                raise RuntimeError("Neither Test nor Validation dataset initialized.")
            ds = self.val_dataset
        else:
            ds = self.test_dataset
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        if not self.test_dataset:
            raise RuntimeError(
                "Test dataset not initialized. Check data paths and setup stage."
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,  # **important**
        )
