import torch
import torch.nn as nn
import hydra
import torchvision
import pytorch_lightning as pl
import numpy as np
from pycocotools import mask as maskutils
from torchvision.models import ResNet50_Weights, ResNeXt101_32X8D_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # for mAP calculation
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from torchvision.ops import misc as misc_nn_ops


class MaskRCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        cfg: DictConfig,
        num_step_per_epoch: int = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Load the pre-trained Faster R-CNN model
        self.model = self._create_model(num_classes)

        # Initialize mAP metrics
        # Note: Adjust iou_thresholds and rec_thresholds if needed
        self.val_map = MeanAveragePrecision(iou_type="segm")
        self.test_map = MeanAveragePrecision(iou_type="segm")

    # def _create_model(self, num_classes):
    #     # load an instance segmentation model pre-trained on COCO
    #     # model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    #     #     weights_backbone=ResNet50_Weights.IMAGENET1K_V2
    #     # )

    #     # # get number of input features for the classifier
    #     # in_features = model.roi_heads.box_predictor.cls_score.in_features
    #     # # replace the pre-trained head with a new one
    #     # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #     # # now get the number of input features for the mask classifier
    #     # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #     # hidden_layer = in_features
    #     # # and replace the mask predictor with a new one
    #     # model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #     #     in_features_mask, hidden_layer, num_classes
    #     # )

    #     # return model
    #     # 1) 构造带 FPN 的 ResNeXt-101 骨干，并加载 ImageNet 预训练权重
    #     backbone = resnet_fpn_backbone(
    #         backbone_name="resnext101_32x8d",
    #         weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
    #         trainable_layers=3,  # 可以 fine-tune 的 stage 数量
    #     )
    #     # backbone.out_channels 对应 FPN 输出通道数（一般是 256）

    #     # 2) 用这个骨干去构建 MaskRCNN
    #     model = MaskRCNN(backbone, num_classes)

    #     # 3) （可选）按需替换 box & mask head
    #     in_feat = model.roi_heads.box_predictor.cls_score.in_features
    #     model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    #     in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #     model.roi_heads.mask_predictor = MaskRCNNPredictor(
    #         in_feat_mask, in_feat, num_classes
    #     )

    #     return model

    def _create_model(self, num_classes):
        cnx = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        backbone_cnx = cnx.features  # nn.Sequential

        # backbone = cnx.features

        # x = torch.randn(1, 3, 224, 224)
        # for name, module in backbone.named_children():
        #     x = module(x[])
        #     print(f"  layer {name:>2s}: output shape = {tuple(x.shape)}")

        # 2) convnext_large.features 是一个长度 5 的 Sequential，结构约是：
        #    [0]=PatchEmbed, [1]=Stage1, [2]=Stage2, [3]=Stage3, [4]=Stage4
        return_layers = {
            "1": "0",  # take features[1] → FPN level0
            "3": "1",  # take features[2] → FPN level1
            "5": "2",  # take features[3] → FPN level2
            "7": "3",  # take features[4] → FPN level3
        }
        in_channels_list = [128, 256, 512, 1024]
        out_channels = 256

        backbone = BackboneWithFPN(
            backbone=backbone_cnx,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        # backbone.out_channels == out_channels

        model = MaskRCNN(backbone, num_classes)

        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

        in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_feat_mask, in_feat, num_classes
        )

        return model

    def forward(self, images, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Faster R-CNN returns a dict of losses during training
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # add prefix to keys
        loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}

        # log loss
        self.log_dict(
            loss_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        self.log(
            "train_loss_total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Inference mode, targets are not needed
        raw_outputs = self.model(images)

        preds, gts = [], []
        for out, tgt in zip(raw_outputs, targets):
            # out['masks']: Tensor[num_pred, 1, H, W], float
            pred_masks = (out["masks"].squeeze(1) > 0.5).to(
                torch.uint8
            )  # Tensor[num_pred, H, W]
            # CPU + numpy
            pred_masks = pred_masks.cpu()

            preds.append(
                {
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                    "masks": pred_masks,
                }
            )

            # tgt["masks"]: Tensor[num_gt, H, W]，
            gt_masks = tgt["masks"].to(torch.uint8).cpu()
            gts.append(
                {
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu(),
                    "masks": gt_masks,
                }
            )

        self.val_map.update(preds, gts)

        # Update mAP metrics
        # Torchmetrics expects preds and targets in specific formats
        # preds: List[Dict[str, Tensor]] with 'boxes', 'scores', 'labels'
        # targets: List[Dict[str, Tensor]] with 'boxes', 'labels'
        # self.val_map.update(outputs, targets)

    def predict_step(self, batch, batch_idx):
        """
        Returns a list of COCO-style dicts, each dict contains:
          image_id, bbox, score, category_id, segmentation (RLE)
        """
        images, img_ids = batch
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)

        results = []
        score_thr = self.cfg.trainer.get("score_threshold", 0.5)
        for i, output in enumerate(outputs):
            image_id = img_ids[i]

            boxes = output["boxes"].cpu()  # [N,4]
            scores = output["scores"].cpu()  # [N]
            labels = output["labels"].cpu()  # [N]
            masks = output["masks"].squeeze(1).cpu()  # [N,H,W], float

            keep = scores >= score_thr
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

            masks = (masks > 0.5).to(torch.uint8).numpy()

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                xmin, ymin, xmax, ymax = box.tolist()
                rle = maskutils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                results.append(
                    {
                        "image_id": image_id,
                        "bbox": [
                            float(xmin),
                            float(ymin),
                            float(xmax),
                            float(ymax),
                        ],
                        "score": float(score),
                        "category_id": int(label),
                        "segmentation": {
                            "size": [int(mask.shape[0]), int(mask.shape[1])],
                            "counts": rle["counts"],
                        },
                    }
                )
        return results

    # def test_step(self, batch, batch_idx):
    #     images, targets = batch
    #     outputs = self.model(images)
    #     self.test_map.update(outputs, targets)

    def on_validation_epoch_end(self):
        # compute and log mAP
        map_metrics = self.val_map.compute()
        # print(f"[DEBUG] map_metrics: {map_metrics}")
        map_metrics.pop("classes")

        # add prefix to keys
        map_metrics = {f"val/{k}": v for k, v in map_metrics.items()}

        self.log_dict(
            map_metrics,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_map.reset()

    def on_test_epoch_end(self):
        map_metrics = self.test_map.compute()
        map_metrics.pop("classes")

        # add prefix to keys
        map_metrics = {f"test/{k}": v for k, v in map_metrics.items()}

        self.log_dict(
            map_metrics,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.test_map.reset()

    def configure_optimizers(self):

        params = list(self.model.named_parameters())
        backbone_params = [p for n, p in params if "backbone" in n and p.requires_grad]
        # Head（box_predictor + mask_predictor）的参数组
        head_params = [
            p
            for n, p in params
            if ("box_predictor" in n or "mask_predictor" in n) and p.requires_grad
        ]

        if self.cfg.optimizer.optimizer_name == "adamw":
            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": self.cfg.optimizer.backbone_lr},
                    {"params": head_params, "lr": self.cfg.optimizer.head_lr},
                ],
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
