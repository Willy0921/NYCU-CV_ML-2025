# src/utils/dataset_utils.py
import torch
import os
import random
import copy
import glob
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from torchvision.transforms import functional as F


def data_augmentation(img: Image.Image, mode: int) -> Image.Image:
    """
    img: 输入的 PIL RGB 图像
    mode: 0-7 分别对应：
        0 原图
        1 垂直翻转
        2 逆时针旋转 90°
        3 逆时针旋转 90° 后垂直翻转
        4 旋转 180°
        5 旋转 180° 后垂直翻转
        6 逆时针旋转 270°
        7 逆时针旋转 270° 后垂直翻转
    """
    if mode == 0:
        return img
    elif mode == 1:
        return F.vflip(img)
    elif mode == 2:
        return F.rotate(img, angle=90, expand=True)
    elif mode == 3:
        return F.vflip(F.rotate(img, angle=90, expand=True))
    elif mode == 4:
        return F.rotate(img, angle=180, expand=True)
    elif mode == 5:
        return F.vflip(F.rotate(img, angle=180, expand=True))
    elif mode == 6:
        return F.rotate(img, angle=270, expand=True)
    elif mode == 7:
        return F.vflip(F.rotate(img, angle=270, expand=True))
    else:
        raise ValueError(f"Invalid augmentation mode: {mode}")


class PromptDataset(Dataset):
    def __init__(self, root_dir, ids=None, transforms=None, patch_size=(256, 256)):
        super(PromptDataset, self).__init__()

        self.root_dir = root_dir

        self.sample_ids = (
            sorted(ids)
            if ids is not None
            else sorted(glob.glob(root_dir + "/degraded/*.png"))
        )

        self.transforms = transforms

        self.patch_size = patch_size

        self.is_train = True

        self.toTensor = ToTensor()

    def _get_clean_id(self, clean_id):
        name = clean_id.split("/")[-1].split(".")[0]
        type = name.split("-")[0]
        idx = name.split("-")[1]
        return f"{type}_clean-{idx}.png"

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        clean_dir = os.path.join(
            self.root_dir,
            "clean",
        )

        degrad_img = Image.open(sample_id).convert("RGB")
        clean_id = self._get_clean_id(sample_id)
        clean_img = Image.open(os.path.join(clean_dir, clean_id)).convert("RGB")

        if self.is_train:

            flag_aug = random.randint(0, 7)
            degrad_img = data_augmentation(degrad_img, flag_aug)
            clean_img = data_augmentation(clean_img, flag_aug)

        degrad_img = self.toTensor(degrad_img)
        clean_img = self.toTensor(clean_img)

        return degrad_img, clean_img

    def __len__(self):
        return len(self.sample_ids)


def collate_fn(batch):
    batch_size = len(batch)
    degrad_patch = torch.zeros((batch_size, 3, 256, 256), dtype=torch.float32)
    clean_patch = torch.zeros((batch_size, 3, 256, 256), dtype=torch.float32)

    for i in range(batch_size):
        degrad_patch[i] = batch[i][0]
        clean_patch[i] = batch[i][1]

    return degrad_patch, clean_patch


class PromptIRTestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        super(PromptIRTestDataset, self).__init__()

        self.root_dir = root_dir
        self.transforms = transforms

        self.sample_ids = sorted(glob.glob(os.path.join(root_dir, "*.png")))

        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img = np.array(Image.open(sample_id).convert("RGB"))

        if self.transforms:
            img = self.transforms(img)

        img = self.toTensor(img)

        file_name = os.path.basename(sample_id)

        return file_name, img


def collate_fn_test(batch):
    batch_size = len(batch)
    imgs = torch.zeros((batch_size, 3, 256, 256), dtype=torch.float32)

    for i in range(batch_size):
        imgs[i] = batch[i][1]

    return imgs
