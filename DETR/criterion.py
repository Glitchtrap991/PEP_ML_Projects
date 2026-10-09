import torch
from torch import nn
from torchvision.ops import generalized_box_iou

from box_ops import cxcywh_to_xyxy


class DETRLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        eos_coef=0.1,
        lambda_bbox=5.0,
        lambda_giou=2.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher

        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou

        # Weight for each class + no-object
        class_weights = torch.ones(num_classes + 1)

        # Last class = no-object (eos)
        class_weights[-1] = eos_coef

        self.register_buffer(
            "class_weights",
            class_weights
        )

    def loss_labels(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]
        B, Q, _ = pred_logits.shape

        # Initially EVERYTHING is no-object
        target_classes = torch.full(
            (B, Q),
            self.num_classes,
            dtype=torch.long,
            device=pred_logits.device
        )

        # Replace matched queries with actual classes
        for b, (pred_idx, target_idx) in enumerate(indices):
            pred_idx = pred_idx.to(pred_logits.device)
            target_idx = target_idx.to(pred_logits.device)

            target_classes[b, pred_idx] = (
                targets[b]["labels"][target_idx]
            )

        loss_ce = nn.functional.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            weight=self.class_weights
        )

        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        pred_boxes_list = []
        target_boxes_list = []

        for b, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_idx = pred_idx.to(
                outputs["pred_boxes"].device
            )

            target_idx = target_idx.to(
                targets[b]["boxes"].device
            )

            pred_boxes_list.append(
                outputs["pred_boxes"][b, pred_idx]
            )

            target_boxes_list.append(
                targets[b]["boxes"][target_idx]
            )

        # Entire batch contains no objects
        if len(pred_boxes_list) == 0:
            zero = outputs["pred_boxes"].sum() * 0
            return zero, zero

        pred_boxes = torch.cat(pred_boxes_list)
        target_boxes = torch.cat(target_boxes_list)

        # ----------------------------------
        # L1 loss
        # ----------------------------------
        loss_bbox = nn.functional.l1_loss(
            pred_boxes,
            target_boxes,
            reduction="sum"
        )

        num_boxes = target_boxes.shape[0]
        loss_bbox = loss_bbox / max(num_boxes, 1)

        # ----------------------------------
        # GIoU loss
        # ----------------------------------
        pred_xyxy = cxcywh_to_xyxy(pred_boxes)
        target_xyxy = cxcywh_to_xyxy(target_boxes)

        giou_matrix = generalized_box_iou(
            pred_xyxy,
            target_xyxy
        )

        giou = torch.diag(giou_matrix)

        loss_giou = (1 - giou).sum()
        loss_giou = loss_giou / max(num_boxes, 1)

        return loss_bbox, loss_giou

    def forward(self, outputs, targets):
        indices = self.matcher(
            outputs,
            targets
        )

        loss_ce = self.loss_labels(
            outputs,
            targets,
            indices
        )

        loss_bbox, loss_giou = self.loss_boxes(
            outputs,
            targets,
            indices
        )

        total_loss = (
            loss_ce
            + self.lambda_bbox * loss_bbox
            + self.lambda_giou * loss_giou
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }
