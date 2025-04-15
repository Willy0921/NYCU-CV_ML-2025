import torch
import torch.nn as nn
import hydra
import torchvision
import pytorch_lightning as pl
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # for mAP calculation
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torchvision.transforms.v2 as T


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.fc1(x) * self.swish(self.fc2(x))


# --- New Attention Box Predictor ---
class AttentionBoxPredictor(nn.Module):
    """
    Replaces FastRCNNPredictor. Uses query tokens and MHA.

    Takes RoI features (e.g., 1024-dim) and predicts class scores and box deltas.
    """

    def __init__(
        self,
        representation_size: int,
        num_classes: int,
        num_queries: int = 16,  # Number of learnable query tokens
        num_heads: int = 8,  # Number of attention heads
        ffn_hidden_dim: int = None,  # Hidden dim for FFN
        dropout: float = 0.1,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.embed_dim = representation_size  # Use representation size as attention dim

        # Learnable query tokens (shared across all RoIs in a batch)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, self.embed_dim))

        # Multi-Head Attention layer
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )  # Expects (batch, seq, feature)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)  # Norm before final prediction layers

        self.ffn = SwiGLU(self.embed_dim, self.embed_dim)
        self.activation = nn.Identity()  # SwiGLU includes activation

        self.dropout = nn.Dropout(dropout)

        # Final prediction layers
        self.cls_score = nn.Linear(self.embed_dim, num_classes)
        # Faster R-CNN predicts deltas for each class (excluding background sometimes, but standard is num_classes)
        self.bbox_pred = nn.Linear(self.embed_dim, num_classes * 4)

    def forward(self, x):
        """
        Args:
            x (Tensor): RoI features, shape (num_rois, representation_size)
        Returns:
            Tuple[Tensor, Tensor]: cls_scores (num_rois, num_classes),
                                   bbox_preds (num_rois, num_classes * 4)
        """
        if x.dim() == 0 or x.shape[0] == 0:
            # Handle cases with no proposals (common during training/inference)
            device = x.device
            return (
                torch.zeros((0, self.num_classes), device=device),
                torch.zeros((0, self.num_classes * 4), device=device),
            )

        num_rois, _ = x.shape

        # Prepare Query, Key, Value for MHA
        # Query: Expand query tokens for each RoI. Shape: (num_rois, num_queries, embed_dim)
        query = self.query_tokens.unsqueeze(0).expand(num_rois, -1, -1)

        # Key/Value: RoI features treated as a sequence of length 1 for each query.
        # Shape: (num_rois, 1, representation_size)
        key = x.unsqueeze(1)
        value = x.unsqueeze(1)

        # Apply MHA: Query attends to Key/Value
        # attn_output shape: (num_rois, num_queries, embed_dim)
        attn_output, _ = self.mha(query, key, value)  # Q, K, V

        # --- Processing after MHA ---
        # Add & Norm (like in Transformers) - applied to query tokens
        processed_queries = self.norm1(query + self.dropout(attn_output))

        # Apply FFN
        ffn_output = self.ffn(processed_queries)
        processed_queries = processed_queries + self.dropout(ffn_output)

        # Apply final norm
        final_features = self.norm2(processed_queries)

        cls_feature = final_features[:, 0, :]
        bbox_feature = processed_queries[:, 1:, :].mean(dim=1)

        # --- Prediction ---
        cls_scores = self.cls_score(cls_feature)
        bbox_preds = self.bbox_pred(bbox_feature)

        return cls_scores, bbox_preds


class FasterRCNNModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        weights: str,
        cfg: DictConfig,
        num_step_per_epoch: int = None,
        **kwargs,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Load the pre-trained Faster R-CNN model
        self.model = self._create_model(num_classes, weights)

        # Initialize mAP metrics
        # Note: Adjust iou_thresholds and rec_thresholds if needed
        self.val_map = MeanAveragePrecision(iou_type="bbox")
        self.test_map = MeanAveragePrecision(iou_type="bbox")

    def _create_model(self, num_classes, weights):
        # Load a pre-trained Faster R-CNN model (e.g., ResNet50 FPN V2)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, num_classes
        )

        # Get the input feature dimension for the predictor
        # This comes from the output of the roi_heads.box_head (typically TwoMLPHead)
        # if hasattr(model.roi_heads.box_head, "fc7"):
        #     representation_size = model.roi_heads.box_head.fc7.out_features
        # else:
        #     # Might need inspection if box_head structure changes
        #     # Example for older torchvision versions or different backbones:
        #     # representation_size = model.roi_heads.box_predictor.cls_score.in_features
        #     # Fallback or raise error if uncertain
        #     try:
        #         # Infer from the original predictor if possible
        #         representation_size = (
        #             model.roi_heads.box_predictor.cls_score.in_features
        #         )
        #         print(
        #             f"Warning: Inferring representation size ({representation_size}) from original predictor. Verify this is correct."
        #         )
        #     except AttributeError:
        #         raise RuntimeError(
        #             "Could not automatically determine representation size for RoI features."
        #         )

        # # --- Replace the box predictor ---
        # # Get parameters for the AttentionBoxPredictor from the config
        # attn_cfg = self.cfg.model.get(
        #     "attention_head", {}
        # )  # Add a section in your config
        # num_queries = attn_cfg.get("num_queries", 16)
        # num_heads = attn_cfg.get("num_heads", 8)
        # ffn_hidden_dim_attn = attn_cfg.get(
        #     "ffn_hidden_dim", None
        # )  # Default to simple Linear if None
        # dropout_attn = attn_cfg.get("dropout", 0.1)

        # print(f"Creating AttentionBoxPredictor with:")
        # print(f"  representation_size={representation_size}")
        # print(f"  num_classes={num_classes}")
        # print(f"  num_queries={num_queries}")
        # print(f"  num_heads={num_heads}")
        # print(f"  ffn_hidden_dim={ffn_hidden_dim_attn}")
        # print(f"  dropout={dropout_attn}")

        # # Instantiate the new predictor
        # attention_predictor = AttentionBoxPredictor(
        #     representation_size=representation_size,
        #     num_classes=num_classes,
        #     num_queries=num_queries,
        #     num_heads=num_heads,
        #     ffn_hidden_dim=ffn_hidden_dim_attn,
        #     dropout=dropout_attn,
        # )

        # # Replace the original predictor
        # model.roi_heads.box_predictor = attention_predictor
        # -----------------------------

        return model

    def forward(self, images, targets=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        processed_images = images
        processed_targets = targets

        # Faster R-CNN returns a dict of losses during training
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # add prefix to keys
        loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}

        # log loss
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Inference mode, targets are not needed
        outputs = self.model(images)

        # Update mAP metrics
        # Torchmetrics expects preds and targets in specific formats
        # preds: List[Dict[str, Tensor]] with 'boxes', 'scores', 'labels'
        # targets: List[Dict[str, Tensor]] with 'boxes', 'labels'
        self.val_map.update(outputs, targets)

    def predict_step(self, batch, batch_idx):
        """
        Performs prediction for Task 1 (Detection).
        Returns results in a format ready for COCO JSON conversion.
        """
        images, img_ids = batch  # Unpack batch from predict_dataloader

        # Get model predictions
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            outputs = self.model(
                images
            )  # List of dicts [{'boxes': T, 'labels': T, 'scores': T}, ...]

        results = []
        score_threshold = self.cfg.trainer.get(
            "score_threshold", 0.5
        )  # Get threshold from config

        for i, output in enumerate(outputs):
            image_id = img_ids[i]  # Get the corresponding image ID
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            # Filter by score threshold
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Convert to COCO format [{image_id, bbox, score, category_id}, ...]
            for box, score, label in zip(boxes, scores, labels):
                box = box.cpu().tolist()  # [xmin, ymin, xmax, ymax]
                score = score.cpu().item()
                label = label.cpu().item()

                # Convert to COCO bbox format [xmin, ymin, width, height]
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin

                results.append(
                    {
                        "image_id": image_id,
                        "bbox": [xmin, ymin, width, height],
                        "score": score,
                        "category_id": label,
                    }
                )

        return results  # Return the list of COCO-formatted detections for this batch

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

        self.log_dict(map_metrics, on_epoch=True, prog_bar=True)
        self.val_map.reset()

    def on_test_epoch_end(self):
        map_metrics = self.test_map.compute()
        map_metrics.pop("classes")

        # add prefix to keys
        map_metrics = {f"test/{k}": v for k, v in map_metrics.items()}

        self.log_dict(map_metrics, on_epoch=True, prog_bar=True)
        self.test_map.reset()

    def configure_optimizers(self):

        if self.cfg.optimizer.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )

        # Define the warmup scheduler
        total_warmup_steps = (
            self.cfg.optimizer.lr_scheduler.warmup_num_epochs_ratio
            * self.trainer.max_epochs
            * self.hparams.num_step_per_epoch
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
                T_max=self.cfg.optimizer.lr_scheduler.T_max
                * self.hparams.num_step_per_epoch,  # unit: step
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
