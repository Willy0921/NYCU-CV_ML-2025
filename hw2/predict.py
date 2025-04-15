import hydra
import torch
import pytorch_lightning as pl
import json
import csv
import os
from omegaconf import DictConfig, OmegaConf, open_dict

from src.datamodules.custom_datamodule import CustomObjectDataModule
from src.models.faster_rcnn_module import FasterRCNNModule


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def predict(cfg: DictConfig) -> None:
    print("------ Prediction Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("------------------------------------")

    # --- Ensure checkpoint path is set ---
    if not cfg.predict.checkpoint_path or cfg.predict.checkpoint_path == "???":
        raise ValueError("Please set the 'checkpoint_path' in configs/config.yaml")
    if not os.path.isfile(cfg.predict.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {cfg.predict.checkpoint_path}"
        )

    if not cfg.predict.checkpoint_path:
        raise ValueError(
            "Checkpoint path is not set or invalid. Please check your configuration."
        )

    if not cfg.model.get("num_classes"):
        raise ValueError(
            "Model 'num_classes' could not be determined. Please set it in configs/config.yaml or ensure it's saved in the checkpoint hyperparameters."
        )

    pl.seed_everything(cfg.seed, workers=True)

    # --- Instantiate DataModule ---
    print("Instantiating DataModule for prediction...")
    datamodule = CustomObjectDataModule(
        data_root=cfg.data.data_root,
        train_split=None,
        val_split=None,
        test_split=cfg.data.test_split,
        annotation_filename_pattern=cfg.data.get(
            "annotation_filename_pattern", "{split}.json"
        ),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    datamodule.setup("predict")  # Explicitly setup for predict stage
    print("DataModule Instantiated.")

    # --- Load Model from Checkpoint ---
    print(f"Loading model from checkpoint: {cfg.predict.checkpoint_path}")
    model = FasterRCNNModule.load_from_checkpoint(
        checkpoint_path=cfg.predict.checkpoint_path,
        map_location="cpu",
        num_classes=cfg.model.num_classes,
        weights=None,
        cfg=cfg,
    )
    model.eval()
    print("Model Loaded.")

    # --- Instantiate Trainer ---
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
        enable_progress_bar=True,
    )
    print("Trainer Instantiated.")

    # --- Run Prediction ---
    print(
        f"Starting Prediction on data from: {os.path.join(cfg.data.data_root, cfg.data.test_split)}"
    )
    # return_predictions=True returns a list of outputs from predict_step
    # For our predict_step returning a list per batch, this will be a list of lists.
    raw_predictions = trainer.predict(
        model, datamodule=datamodule, return_predictions=True
    )
    print("Prediction Finished.")

    # --- Process Predictions ---
    print("Processing predictions...")
    # Flatten the list of lists from raw_predictions
    coco_results_task1 = [
        item for batch_result in raw_predictions for item in batch_result
    ]

    # --- Task 2: ---
    print("Generating Task 2 CSV (Placeholder)...")
    csv_results_task2 = []
    # Get unique image IDs from Task 1 results
    predicted_image_ids = sorted(list({res["image_id"] for res in coco_results_task1}))
    # If predict dataset might contain images with NO detections, get IDs from dataloader
    if not predicted_image_ids and datamodule.test_dataset:
        print("No detections found, getting image IDs from predict dataset.")
        predicted_image_ids = datamodule.test_dataset.ids  # Get all image IDs loaded

    for img_id in predicted_image_ids:

        detected_labels = [
            (res["category_id"] - 1, res["bbox"])  # category_id starts from 1
            for res in coco_results_task1
            if res["image_id"] == img_id
        ]

        center_x_coordinates = [(bbox[0] + bbox[2]) / 2 for _, bbox in detected_labels]
        # Sort detected labels based on their center x-coordinates
        detected_labels = [
            label for _, label in sorted(zip(center_x_coordinates, detected_labels))
        ]

        # Extract the labels in sorted order
        detected_labels = [label for label, _ in detected_labels]

        pred_label = "".join(
            [str(label) for label in detected_labels]
        )  # Convert to string (e.g., "1234" for digits)

        if not detected_labels:
            pred_label = "-1"  # No detections for this image

        csv_results_task2.append({"image_id": img_id, "pred_label": pred_label})
    print("Task 2 CSV data prepared (using placeholder logic).")

    # --- Save Outputs ---
    # Use Hydra's output directory for this run
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    # Save Task 1 results
    json_path = os.path.join(output_dir, "pred.json")
    print(
        f"Saving Task 1 COCO predictions ({len(coco_results_task1)} detections) to: {json_path}"
    )
    with open(json_path, "w") as f:
        json.dump(coco_results_task1, f)  # COCO format doesn't usually need indentation

    # Save Task 2 results
    csv_path = os.path.join(output_dir, "pred.csv")
    print(
        f"Saving Task 2 CSV predictions ({len(csv_results_task2)} images) to: {csv_path}"
    )
    if csv_results_task2:
        fieldnames = ["image_id", "pred_label"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_results_task2)
    else:
        # Create empty file with header if no predictions
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "pred_label"])
            writer.writeheader()
        print("No results to write for CSV (CSV file created with header only).")

    print(f"Prediction results saved in: {output_dir}")
    print("Prediction script finished.")


if __name__ == "__main__":
    predict()
