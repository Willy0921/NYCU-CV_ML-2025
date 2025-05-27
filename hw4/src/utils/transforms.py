import torch
import torchvision.transforms.v2 as T
import torchvision.transforms as tvT
import kornia.augmentation as K


def get_train_transform(patch_size=256):
    return T.Compose(
        [
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(degrees=45),
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # T.RandomCrop(patch_size),
            # motion blur
            # K.RandomMotionBlur(
            #     kernel_size=(3, 3), angle=(0, 360), direction=(0, 1), p=0.5
            # ),
            # Gaussian blur
            # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # To tensor 並歸一化至 [0,1]
            T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        ]
    )


def get_val_transform(patch_size=256):
    return T.Compose(
        [T.CenterCrop(patch_size), T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    )


def get_test_transform():
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
