import torch._dynamo

torch._dynamo.config.suppress_errors = True

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from src.datamodules.custom_datamodule import (
    CustomObjectDetectionDataModule,
    CustomInstanceSegmentationDataModule,
)
from src.models.mask_rcnn_module import MaskRCNNModule
import wandb
import os
import torch


def print_separator(title=""):
    """Prints a separator line with an optional title."""
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"===== {title.upper()} =====")
        print(f"{line}")
    else:
        print(f"\n{line}")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print_separator("Configuration")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    print_separator("Setting Random Seed")
    pl.seed_everything(cfg.seed, workers=True)
    print(f"Seed set to: {cfg.seed}")

    print_separator("Initializing DataModule")
    print("Instantiating DataModule...")
    # datamodule = CustomObjectDetectionDataModule(
    #     data_root=cfg.data.data_root,
    #     train_split=cfg.data.get("train_split", None),
    #     val_split=cfg.data.get("val_split", None),
    #     test_split=cfg.data.get("test_split", None),
    #     annotation_filename_pattern=cfg.data.get(
    #         "annotation_filename_pattern", "{split}.json"
    #     ),
    #     batch_size=cfg.data.batch_size,
    #     num_workers=cfg.data.num_workers,
    #     pin_memory=cfg.data.pin_memory,
    # )
    datamodule = CustomInstanceSegmentationDataModule(
        data_dir=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        val_split_ratio=cfg.data.get("val_split_ratio", 0.2),
        seed=cfg.seed,
    )
    print("[DataModule] Instantiated successfully.")
    datamodule.setup(stage="fit")
    print("  Number of training samples:", len(datamodule.train_dataset))
    print("[DataModule] Setup completed.")

    total_num_devices = cfg.trainer.get("total_num_devices", 1)
    accumulate_grad_batches = cfg.trainer.get("accumulate_grad_batches", 1)
    total_batch_size = cfg.data.batch_size * accumulate_grad_batches * total_num_devices
    print("-" * 30)
    print(f"  Number of devices: {total_num_devices}")
    print(f"  Batch size per device: {cfg.data.batch_size}")
    print(f"  Accumulate grad batches: {accumulate_grad_batches}")
    print(f"  Effective Batch Size: {total_batch_size}")
    print("-" * 30)
    print("[Trainer] Instantiated successfully.")

    print_separator("Initializing Model")
    print("Instantiating Model...")
    num_step_per_epoch = len(datamodule.train_dataset) // total_batch_size
    model = MaskRCNNModule(
        num_classes=cfg.model.num_classes,
        # weights=cfg.model.weights,
        cfg=cfg,
        num_step_per_epoch=num_step_per_epoch,
    )
    print("[Model] Instantiated successfully.")

    if cfg.model.get("allow_torch_compile", False):
        print("Attempting to compile the model with torch.compile()...")
        try:
            model = torch.compile(model)
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: Failed to compile the model with torch.compile(): {e}")
            print("Proceeding without compilation.")

    print_separator("Initializing Logger")
    wandb_logger = None
    if cfg.logger.logger_name == "wandb":
        print("Attempting to instantiate Wandb Logger...")
        try:
            wandb_logger = WandbLogger(
                project=cfg.logger.project,
                name=cfg.logger.name,
                save_dir=cfg.logger.save_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print("[Logger] Wandb Logger watching model...")
            wandb_logger.watch(
                model, log="all", log_freq=cfg.trainer.get("log_every_n_steps", 50)
            )
            print("[Logger] Wandb Logger instantiated successfully.")
        except wandb.errors.UsageError as e:
            print(f"[Logger] WARNING: Failed to initialize Wandb logger: {e}")
            print("[Logger] Proceeding without Wandb logging.")
            wandb_logger = None
        except Exception as e:
            print(
                f"[Logger] ERROR: An unexpected error occurred during Wandb initialization: {e}"
            )
            wandb_logger = None
    else:
        print("[Logger] Wandb logger not configured. Skipping.")

    print_separator("Initializing Callbacks")
    print("Instantiating Callbacks...")
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor="val/map",
        mode="max",
        filename="epoch{epoch:02d}-map{val/map:.4f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    print("[Callbacks] ModelCheckpoint added.")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    print("[Callbacks] LearningRateMonitor added.")
    print("[Callbacks] Instantiated successfully.")

    print_separator("Initializing Trainer")
    print("Instantiating Trainer...")
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
        fast_dev_run=cfg.trainer.get("fast_dev_run", False),
        strategy=DDPStrategy(process_group_backend="gloo"),
        # enable_progress_bar=False,
    )

    print_separator("Starting Training")
    if cfg.trainer.get("resume_from_ckpt_path", None):
        ckpt_path = cfg.trainer.resume_from_ckpt_path
        print(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
    else:
        print("Starting training from scratch.")
        trainer.fit(model, datamodule=datamodule)
    print_separator("Training Finished")

    if wandb_logger:
        print_separator("Finalizing Wandb Run")
        wandb.finish()
        print("Wandb run finished.")
    else:
        print_separator("Run Finished")

    print("Script execution completed.")


if __name__ == "__main__":
    train()
