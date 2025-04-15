# src/utils/custom_dataset_utils.py
import torch
import os
import json
from PIL import Image
import os.path
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from typing import List, Dict, Any, Tuple


class CustomObjectDetectionDataset(torch.utils.data.Dataset):
    """
    This class is a custom dataset for object detection tasks.
    It loads images and their corresponding annotations from a specified folder
    and annotation file.
    """

    def __init__(
        self, img_folder: str, ann_file: str, transforms=None, is_predict: bool = False
    ):
        super().__init__()
        self.img_folder = img_folder
        self.transforms = transforms
        self.is_predict = is_predict  # flag for prediction mode

        self.images_info: Dict[int, Dict] = {}
        self.img_id_to_anns: Dict[int, List[Dict]] = {}
        self.ids: List[int] = []

        if self.is_predict:
            # In prediction mode, load images from the folder only
            print(f"Prediction mode: Loading images from {img_folder}")
            if not os.path.isdir(img_folder):
                raise FileNotFoundError(f"Image folder not found: {img_folder}")
            image_files = [
                f
                for f in os.listdir(img_folder)
                if os.path.isfile(os.path.join(img_folder, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            # Try to infer image_id from filename (assuming filename is like 'id.png')
            for filename in sorted(image_files):  # Sort for consistent order
                try:
                    # Extract ID assuming format like "123.png"
                    img_id = int(os.path.splitext(filename)[0])
                    self.images_info[img_id] = {"id": img_id, "file_name": filename}
                    self.ids.append(img_id)
                except ValueError:
                    print(
                        f"Warning: Could not parse image ID from filename '{filename}'. Skipping."
                    )
            print(f"Found {len(self.ids)} images for prediction.")
            if not self.ids:
                print(
                    "Warning: No images found for prediction in the specified folder."
                )
        elif ann_file:
            # In training/validation mode, load from the annotation file
            print(f"Loading annotations from: {ann_file}")
            if not os.path.exists(ann_file):
                raise FileNotFoundError(f"Annotation file not found: {ann_file}")

            with open(ann_file, "r") as f:
                data = json.load(f)

            self.images_info = {img["id"]: img for img in data["images"]}
            annotations = data["annotations"]

            # --- create the mapping from image_id to annotations ---
            self.img_id_to_anns: Dict[int, List[Dict]] = {
                img_id: [] for img_id in self.images_info.keys()
            }
            for ann in annotations:
                img_id = ann["image_id"]
                if img_id in self.img_id_to_anns:
                    self.img_id_to_anns[img_id].append(ann)

            # --- Filter out images without annotations ---
            self.ids = [
                img_id for img_id, anns in self.img_id_to_anns.items() if anns
            ]  # Only keep images with annotations
            # If you want to keep all images regardless of annotations, uncomment the line below:
            # self.ids = list(self.images_info.keys())

            if not self.ids:
                print(
                    f"Warning: No annotations found or no image IDs match annotations in {ann_file}. Dataset will be empty."
                )

            print(f"Found {len(self.ids)} images with annotations in {ann_file}")
        else:
            raise ValueError(
                "Either ann_file must be provided or is_predict must be True."
            )

    def _load_image(self, img_id: int) -> Image.Image:
        img_info = self.images_info[img_id]
        # Assuming 'file_name' is the filename of the image, e.g., '1.png'
        path = os.path.join(self.img_folder, img_info["file_name"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")

    def _load_target(self, img_id: int) -> List[Dict]:
        # Directly return the annotations for the image_id
        return self.img_id_to_anns.get(img_id, [])

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, torch.Tensor]]:

        img_id = self.ids[index]
        image = self._load_image(img_id)

        if self.is_predict:
            if self.transforms is not None:
                try:
                    image = self.transforms(image)
                except TypeError:
                    image, _ = self.transforms(
                        image,
                        {
                            "boxes": torch.zeros((0, 4)),
                            "labels": torch.zeros(0, dtype=torch.int64),
                        },
                    )

            return image, img_id
        else:
            target_anns = self._load_target(img_id)

            # --- Convert annotations to Faster R-CNN format ---
            boxes = []
            labels = []
            area = []
            iscrowd = []
            img_w, img_h = image.size

            for ann in target_anns:
                # bbox: [x_min, y_min, width, height]
                xmin, ymin, w, h = ann["bbox"]
                xmax = xmin + w
                ymax = ymin + h
                boxes.append([xmin, ymin, xmax, ymax])  # [xmin, ymin, xmax, ymax]

                # Ensure category_id exists
                if "category_id" not in ann:
                    raise ValueError(
                        f"Annotation with id {ann.get('id', 'N/A')} is missing 'category_id'. Image ID: {img_id}"
                    )
                labels.append(ann["category_id"])

                area.append(ann.get("area", w * h))
                iscrowd.append(ann.get("iscrowd", 0))

            target_dict = {}
            target_dict["boxes"] = BoundingBoxes(
                boxes if boxes else torch.zeros((0, 4)),  # handle empty boxes
                format=BoundingBoxFormat.XYXY,
                canvas_size=(img_h, img_w),
            )
            target_dict["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target_dict["image_id"] = torch.tensor([img_id])
            target_dict["area"] = torch.as_tensor(area, dtype=torch.float32)
            target_dict["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

            if self.transforms is not None:
                image, target_dict = self.transforms(image, target_dict)

            return image, target_dict

    def __len__(self) -> int:
        return len(self.ids)


def collate_fn(batch):
    """
    Handles batching for both training/validation (image, target_dict)
    and prediction (image, image_id).
    """
    # Check the type of the second element in the first item to determine mode
    if isinstance(batch[0][1], dict):  # Training/Validation mode
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets
    elif isinstance(batch[0][1], int):  # Prediction mode
        images = [item[0] for item in batch]
        img_ids = [item[1] for item in batch]
        return images, img_ids
    else:
        # Fallback or error for unexpected format
        return tuple(zip(*batch))
