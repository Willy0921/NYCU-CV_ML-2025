import torch
import torchvision.transforms.v2 as T
import torchvision.transforms as tvT
import kornia.augmentation as K


def get_train_transform():
    return T.Compose(
        [
            T.ToImage(),  # Convert PIL image to tensor
            # specific for cell instance segmentation
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomPerspective(distortion_scale=0.4, p=0.5),
            K.RandomGaussianBlur((3, 3), sigma=(0.1, 2.0), p=0.3),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            K.RandomElasticTransform((31, 31), sigma=4.0, alpha=34.0, p=0.3),
            # general augmentations
            T.RandomPhotometricDistort(p=0.5),
            T.RandomZoomOut(fill={torch.tensor: (123, 117, 104), "others": 0}),
            T.RandomIoUCrop(),
            T.RandomAffine(
                degrees=180,  # rotating +/- 180 degrees
                translate=(0.1, 0.1),  # horizontal and vertical translation 10%
                scale=(0.8, 1.2),  # scale 80% to 120%
                shear=5,  # shear +/- 5 degrees
            ),
            # T.Resize((size, size)),  # Resize to a fixed size
            T.ToDtype(torch.float32, scale=True),  # Normalize to [0.0, 1.0]
            T.SanitizeBoundingBoxes(min_size=1),
        ]
    )


def get_val_transform():
    return T.Compose(
        [
            T.ToImage(),
            # T.Resize((size, size)),  # Resize to a fixed size
            T.ToDtype(torch.float32, scale=True),
            T.SanitizeBoundingBoxes(min_size=1),
        ]
    )


def get_test_transform():
    return tvT.Compose(
        [
            # tvT.ToImage(),
            tvT.ToTensor(),  # PIL→C×H×W, float32 in [0,1]
        ]
    )
