import torch
import torch.nn as nn
import hydra
import torchvision
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from src.models.multi_level_net import PromptIR

# from src.models.net import PromptIR
from torchmetrics.image import PeakSignalNoiseRatio
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class PromptIRModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_step_per_epoch: int = None, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.loss_fn = nn.MSELoss()

        self.data_range_for_psnr = 1.0
        self.val_psnr_metric = PeakSignalNoiseRatio(data_range=self.data_range_for_psnr)

        self.model = self._create_model()

        self.save_hyperparameters()

    def _create_model(self):
        model = PromptIR(decoder=True)
        return model

    def forward(self, x):
        if self.training and x is None:
            raise ValueError("In training mode, x should be passed")
        return self.model(x)

    def _to_01(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def training_step(self, batch, batch_idx):
        degrad_patch, clean_patch = batch
        restored, y2, y3 = self.model(degrad_patch)
        # restored = self.model(degrad_patch)

        restored_01 = self._to_01(restored)
        y2_01 = self._to_01(y2)
        y3_01 = self._to_01(y3)

        # 1. MSE Loss
        loss_mse = (
            self.loss_fn(restored_01, clean_patch)
            + 0.5 * self.loss_fn(y2_01, clean_patch)
            + 0.25 * self.loss_fn(y3_01, clean_patch)
        )

        # FFT Spectrum Loss
        X = torch.fft.rfft2(restored_01, norm="ortho")
        Y = torch.fft.rfft2(clean_patch, norm="ortho")
        loss_fft = F.l1_loss(torch.abs(X), torch.abs(Y))

        # Sobel / Edge‐aware Loss
        # create Sobel kernels
        device = restored_01.device
        dtype = restored_01.dtype
        # horizontal Sobel
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device
        ).view(1, 1, 3, 3)
        # vertical Sobel
        ky = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device
        ).view(1, 1, 3, 3)

        # apply to each channel, groups=channels
        C = restored_01.shape[1]
        kx = kx.repeat(C, 1, 1, 1)
        ky = ky.repeat(C, 1, 1, 1)

        # conv2d → gradient
        grad_rx = F.conv2d(restored_01, kx, padding=1, groups=C)
        grad_ry = F.conv2d(restored_01, ky, padding=1, groups=C)
        grad_cx = F.conv2d(clean_patch, kx, padding=1, groups=C)
        grad_cy = F.conv2d(clean_patch, ky, padding=1, groups=C)

        # combine gradient magnitudes (can also compute loss for x/y separately and weight them)
        grad_r = torch.sqrt(grad_rx**2 + grad_ry**2 + 1e-6)
        grad_c = torch.sqrt(grad_cx**2 + grad_cy**2 + 1e-6)

        loss_edge = F.l1_loss(grad_r, grad_c)

        # 5. Log the losses
        self.log(
            "train_loss_mse",
            loss_mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_fft",
            loss_fft,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_edge",
            loss_edge,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        loss_total = loss_mse + 0.5 * loss_fft + 0.5 * loss_edge

        self.log(
            "train_loss",
            loss_total,
            # loss_mse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # return loss_mse
        return loss_total

    def validation_step(self, batch, batch_idx):

        degrad_patch, clean_patch = batch
        restored, _, _ = self.model(degrad_patch)
        # restored = self.model(degrad_patch)

        restored_01 = self._to_01(restored)

        self.val_psnr_metric.update(restored_01, clean_patch)

    def predict_step(self, batch, batch_idx):
        img_names, imgs = batch
        restored_imgs, _, _ = self.model(imgs)
        # restored_imgs = self.model(imgs)

        results = []
        for img_name, restored_img in zip(img_names, restored_imgs):
            results.append((img_name, restored_img))
            print(f"append {img_name} to results")
        return results

    def on_validation_epoch_end(self):
        psnr_value = self.val_psnr_metric.compute()
        self.log(
            "val_psnr_epoch",
            psnr_value,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_psnr_metric.reset()

    def on_test_epoch_end(self):
        psnr_value = self.val_psnr_metric.compute()
        self.log(
            "test_psnr_epoch",
            psnr_value,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_psnr_metric.reset()

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(self.current_epoch)
    #     lr = scheduler.get_lr()

    def configure_optimizers(self):
        if self.cfg.optimizer.optimizer_name == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

        # Define the warmup scheduler
        total_steps = (
            self.cfg.optimizer.lr_scheduler.T_max * self.hparams.num_step_per_epoch
        )
        total_warmup_steps = (
            total_steps * self.cfg.optimizer.lr_scheduler.warmup_num_epochs_ratio
        )

        if self.cfg.optimizer.lr_scheduler.warmup_name == "linear":
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.cfg.optimizer.lr_scheduler.warmup_lr_factor,
                total_iters=total_warmup_steps,  # unit: step
            )
        else:
            raise ValueError(
                f"Unsupported warmup scheduler: {self.cfg.optimizer.lr_scheduler.warmup_name}"
            )

        if self.cfg.optimizer.lr_scheduler.name == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,  # unit: step
                eta_min=self.cfg.optimizer.lr_scheduler.eta_min,
            )
        else:
            raise ValueError(
                f"Unsupported LR scheduler: {self.cfg.optimizer.lr_scheduler.lr_scheduler_name}"
            )

        # Combine warmup and main scheduler
        scheduler_combined = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[total_warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_combined,
                "interval": "step",
                "frequency": 1,
            },
        }
