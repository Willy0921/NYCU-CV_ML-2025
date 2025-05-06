# src/utils/custom_dataset_utils.py
import torch
import os
import json
import skimage.io as sio
from PIL import Image
import os.path
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask
from torchvision.ops import masks_to_boxes
from typing import List, Dict, Any, Tuple
import numpy as np
import glob


class CustomInstanceSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, ids=None, transforms=None, patch_size=(512, 512)):
        self.root_dir = root_dir
        self.ids = sorted(ids) if ids is not None else sorted(os.listdir(root_dir))
        self.transforms = transforms
        self.patch_h, self.patch_w = patch_size
        # 生成所有 patch 信息: (sample_id, x0, y0, pw, ph)
        self.patch_infos = []
        for sample_id in self.ids:
            folder = os.path.join(root_dir, sample_id)
            img_path = os.path.join(folder, "image.tif")
            if not os.path.exists(img_path):
                continue
            with Image.open(img_path) as img:
                W, H = img.size
            for y in range(0, H, self.patch_h):
                for x in range(0, W, self.patch_w):
                    pw = min(self.patch_w, W - x)
                    ph = min(self.patch_h, H - y)
                    self.patch_infos.append((sample_id, x, y, pw, ph))

    def __len__(self):
        return len(self.patch_infos)

    def __getitem__(self, idx):
        sample_id, x0, y0, pw, ph = self.patch_infos[idx]
        folder = os.path.join(self.root_dir, sample_id)
        # load image and crop
        img = Image.open(os.path.join(folder, "image.tif")).convert("RGB")
        img_patch = img.crop((x0, y0, x0 + pw, y0 + ph))
        # load masks and crop
        mask_paths = sorted(glob.glob(os.path.join(folder, "class*.tif")))
        masks_list, labels_list = [], []
        for p in mask_paths:
            arr = sio.imread(p)
            cls = int(os.path.basename(p).split("class")[1].split(".")[0])
            # crop instance segm masks
            patch_arr = arr[y0 : y0 + ph, x0 : x0 + pw]
            for uid in np.unique(patch_arr)[1:]:  # 跳过背景 0
                m = patch_arr == uid
                if m.any():
                    masks_list.append(m)
                    labels_list.append(cls)
        # to tensor
        if len(masks_list) > 0:
            masks = torch.as_tensor(
                np.stack(masks_list), dtype=torch.uint8
            )  # [N,ph,pw]
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            boxes = masks_to_boxes(masks)
            # filter out invalid bbox
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths > 0) & (heights > 0)
            boxes = boxes[valid]
            labels = labels[valid]
            masks = masks[valid]
        else:
            # no instance
            masks = torch.zeros((0, ph, pw), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": BoundingBoxes(
                boxes, format=BoundingBoxFormat.XYXY, canvas_size=(ph, pw)
            ),
            "labels": labels,
            "masks": Mask(masks),
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        # apply transform
        if self.transforms:
            img_patch, target = self.transforms(img_patch, target)
        return img_patch, target


class CustomInstanceSegmentationTestDataset(torch.utils.data.Dataset):
    """Only load test_release/*.tif, return (image, image_name)"""

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.tif")))
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        name = os.path.basename(path)
        if self.transforms:
            img = self.transforms(img)
        return img, name


def segm_collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return imgs, targets
