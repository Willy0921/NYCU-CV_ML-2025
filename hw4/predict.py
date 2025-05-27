# predict_lightning.py

import os
from PIL import Image
import argparse
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.models.promp_ir_module import PromptIRModule
from src.utils.dataset_utils import PromptIRTestDataset, collate_fn_test

# from src.utils.transforms import get_test_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--data_dir", type=str, default="data/hw4_release_dataset/test/degraded"
    )
    parser.add_argument("--output", type=str, default="pred.npz")
    parser.add_argument(
        "--output_dir", type=str, default="data/hw4_release_dataset/test/predicted"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # load LightningModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PromptIRModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device).eval()

    # prepare dataloader
    ds = PromptIRTestDataset(root_dir=args.data_dir, transforms=None)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=None,
    )

    # predict
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    predictions = trainer.predict(
        model,
        dataloaders=loader,
        return_predictions=True,
    )
    predictions = [x for batch in predictions for x in batch]

    # save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    for prediction in predictions:
        # print(prediction)
        img_name = prediction[0]
        img = prediction[1]  # [3, 256, 256]
        # 限制 tensor 值在 [0,1]，避免乘 255 后溢出
        img = torch.clamp(img, 0.0, 1.0)
        img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(img)
        img.save(os.path.join(args.output_dir, img_name))

    images_dict = {}

    # Loop through all files in the folder
    for filename in os.listdir(args.output_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(args.output_dir, filename)

            # Load image and convert to RGB
            image = Image.open(file_path).convert("RGB")
            img_array = np.array(image)

            # Rearrange to (3, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))

            # Add to dictionary
            images_dict[filename] = img_array

    # Save to .npz file
    np.savez(args.output, **images_dict)

    print(f"Saved {len(images_dict)} images to {args.output}")


if __name__ == "__main__":
    main()
