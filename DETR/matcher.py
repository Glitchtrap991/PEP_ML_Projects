import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from box_ops import cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert (
            cost_class != 0 or
            cost_bbox != 0 or
            cost_giou != 0
        ), "At least one cost weight must be non-zero"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs Hungarian matching between predicted queries and ground truth boxes.

        Args:
            outputs: Dict containing:
                "pred_logits": Tensor of shape [B, Q, num_classes + 1]
                "pred_boxes": Tensor of shape [B, Q, 4]
            targets: List of dicts of length B, each containing:
                "labels": Tensor of shape [N]
                "boxes": Tensor of shape [N, 4]

        Returns:
            List of size B, containing tuples of (pred_indices, target_indices)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        matches = []

        for b in range(batch_size):
            pred_logits = outputs["pred_logits"][b]
            pred_boxes = outputs["pred_boxes"][b]

            target_labels = targets[b]["labels"]
            target_boxes = targets[b]["boxes"]

            # Handle images with no objects
            if len(target_labels) == 0:
                matches.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long)
                ))
                continue

            # -----------------------------------------
            # 1. Classification cost
            # -----------------------------------------
            pred_probs = pred_logits.softmax(-1)
            # Shape [Q, N]: For every query and every GT object, get probability assigned to GT class
            cost_class = -pred_probs[:, target_labels]

            # -----------------------------------------
            # 2. L1 bbox cost
            # -----------------------------------------
            # Shape [Q, N]
            cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)

            # -----------------------------------------
            # 3. GIoU cost
            # -----------------------------------------
            pred_xyxy = cxcywh_to_xyxy(pred_boxes)
            target_xyxy = cxcywh_to_xyxy(target_boxes)

            # generalized_box_iou gives higher = better. Hungarian minimizes, therefore negate it.
            cost_giou = -generalized_box_iou(pred_xyxy, target_xyxy)

            # -----------------------------------------
            # Final matching cost
            # -----------------------------------------
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            # scipy works on CPU / NumPy
            C = C.cpu()
            pred_indices, target_indices = linear_sum_assignment(C)

            matches.append((
                torch.as_tensor(pred_indices, dtype=torch.long),
                torch.as_tensor(target_indices, dtype=torch.long)
            ))

        return matches
