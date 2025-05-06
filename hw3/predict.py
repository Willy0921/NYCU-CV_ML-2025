# predict_lightning.py

import os
import json
import argparse

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.models.mask_rcnn_module import MaskRCNNModule
from src.utils.custom_instance_segmentation_dataset_utils import (
    CustomInstanceSegmentationTestDataset,
)
from src.utils.transforms import get_test_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/hw3-data/test_release")
    parser.add_argument("--mapping", type=str, default="test_image_name_to_ids.json")
    parser.add_argument("--output", type=str, default="test-results.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # load filename->id mapping
    with open(args.mapping, "r") as f:
        mapping_list = json.load(f)
    mapping = {os.path.basename(r["file_name"]): r["id"] for r in mapping_list}

    # load LightningModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskRCNNModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()

    # prepare dataloader
    ds = CustomInstanceSegmentationTestDataset(
        root_dir=args.data_dir, transforms=get_test_transform()
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lambda b: tuple(zip(*b)),
    )

    # predict
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    nested = trainer.predict(model, loader)  # List[List[dict]]
    preds = [x for batch in nested for x in batch]

    for p in preds:
        fname = p["image_id"]
        p["image_id"] = int(mapping[fname])

    # dump
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_file_path = os.path.join(
        os.path.dirname(args.ckpt), os.path.basename(args.output)
    )
    with open(output_file_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions → {output_file_path}")


if __name__ == "__main__":
    main()
