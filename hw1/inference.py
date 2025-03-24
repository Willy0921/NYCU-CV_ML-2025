import torch
import torchvision.transforms as transforms
from model import (
    CustomResNeXt26_Simple,
    CustomResNeXt50_32x4d_Simple,
    CustomResNeXt101_32x8d_Simple,
    CustomSEResNeXt50_32x4d_Simple,
)
from PIL import Image
from tqdm import tqdm
import argparse
import os
import json
import csv


def load_model(model_path, num_classes, device):
    # model = CustomResNeXt26_Simple(num_classes=num_classes)
    # model = CustomResNeXt50_32x4d_Simple(num_classes=num_classes)
    model = CustomSEResNeXt50_32x4d_Simple(num_classes=num_classes)
    # model = CustomResNeXt101_32x8d_Simple(num_classes=num_classes)

    if "checkpoint" in model_path:
        checkpoint = torch.load(model_path, map_location=device)
        # delete '_orig.mod.' to the key to load the model in front of the key

        checkpoint["model_state_dict"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
            if "_orig_mod." in k
        }

        model.load_state_dict(
            state_dict=checkpoint["model_state_dict"],
        )
    elif "last_epoch_model" in model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        weights = torch.load(model_path)
        weights = {
            k.replace("_orig_mod.", ""): v
            for k, v in weights.items()
            if "_orig_mod." in k
        }
        model.load_state_dict(weights)

    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image_path, model, device, class_labels):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_class = class_labels[str(predicted_class)]
    return predicted_class


def process_folder(image_folder, model, device, class_labels, output_csv):
    image_files = [
        f for f in os.listdir(image_folder) if f.endswith(("png", "jpg", "jpeg"))
    ]
    os.makedirs("./results", exist_ok=True)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "pred_label"])

        for image_file in tqdm(image_files):
            image_path = os.path.join(image_folder, image_file)
            pred_label = predict(image_path, model, device, class_labels)

            image_name = os.path.basename(image_file).split(".")[0]
            writer.writerow([image_name, pred_label])
            # print(f"Processed {image_name}: Predicted label {pred_label}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with trained ResNet model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images",
    )
    parser.add_argument(
        "--class_labels", type=str, required=True, help="Path to class labels JSON file"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="prediction.csv",
        help="Path to save predictions CSV",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.class_labels, "r") as f:
        class_labels = json.load(f)

    num_classes = len(class_labels)

    model = load_model(args.model_path, num_classes, device)
    process_folder(args.image_folder, model, device, class_labels, args.output_csv)

    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
