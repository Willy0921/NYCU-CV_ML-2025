import os
import numpy as np
import torch
import random
import torch.nn as nn
from torch.nn import init
from PIL import Image


# crop an image to the multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[
        crop_h // 2 : h - crop_h + crop_h // 2,
        crop_w // 2 : w - crop_w + crop_w // 2,
        :,
    ]


def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception("Invalid choice of image transformation")
    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out
