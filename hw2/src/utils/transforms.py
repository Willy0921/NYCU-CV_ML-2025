import torch
import torchvision.transforms.v2 as T


def get_train_transform():
    return T.Compose(
        [
            T.ToImage(),  # Convert PIL image to tensor
            T.RandomPhotometricDistort(p=0.5),
            T.RandomZoomOut(fill={torch.tensor: (123, 117, 104), "others": 0}),
            T.RandomIoUCrop(),
            T.RandomAffine(
                degrees=10,  # rotating +/- 10 degrees
                translate=(0.1, 0.1),  # horizontal and vertical translation 10%
                scale=(0.8, 1.2),  # scale 80% to 120%
                shear=5,  # shear +/- 5 degrees
            ),
            # T.Resize((224, 224)),  # Resize to a fixed size
            T.ToDtype(torch.float32, scale=True),  # Normalize to [0.0, 1.0]
            T.SanitizeBoundingBoxes(min_size=1),
        ]
    )


def get_val_transform():
    return T.Compose(
        [
            T.ToImage(),
            # T.Resize((224, 224)),  # Resize to a fixed size
            T.ToDtype(torch.float32, scale=True),
        ]
    )
