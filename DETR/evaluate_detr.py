import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import IEDXRayDataset, collate_fn, transforms
from detr_impl import DETR, num_classes


# ============================================================
# IEDXRAY CLASS DEFINITIONS
# ============================================================
CLASS_NAMES = {
    0: "Explosive",
    1: "Battery",
    2: "Modified laptop",
    3: "Modified parts",
    4: "Modified Mobile phone",
    5: "Modified Pager",
    6: "Modified Walkie Talkie",
    7: "Laptop",
    8: "Pager",
    9: "Mobile Phone",
    10: "Walkie-Talkie"
}


# ============================================================
# 1. MODEL LOADING
# ============================================================
def load_detr_checkpoint(checkpoint_path: Path, device: torch.device):
    """Loads trained DETR weights from checkpoint in eval mode."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path.resolve()}\n"
            f"Please ensure best.pt has been created by training before evaluating."
        )

    model = DETR(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "val_loss" in checkpoint:
        val_l = checkpoint.get("val_loss", "N/A")
        val_map = checkpoint.get("val_map", "N/A")
        print(f"Checkpoint info -> Val Loss: {val_l} | Val mAP50-95: {val_map}")

    return model


# ============================================================
# 2 & 3. PREDICTION COLLECTION (ONE-PASS INFERENCE)
# ============================================================
@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Runs single-pass inference over dataloader.

    Converts coordinates from normalized cxcywh to absolute pixel xyxy.
    Returns cached predictions and ground-truth boxes on CPU.
    """
    model.eval()
    cached_predictions = []
    cached_targets = []

    print(f"Running inference over {len(dataloader.dataset)} images...")

    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)

        image_height = images.shape[2]
        image_width = images.shape[3]

        # DETR output logits -> probabilities
        probs = outputs["pred_logits"].softmax(-1)
        # Exclude final no-object class
        scores, labels = probs[..., :-1].max(-1)

        batch_size = images.shape[0]
        for b in range(batch_size):
            pred_boxes_b = outputs["pred_boxes"][b]

            # Convert predicted boxes: normalized cxcywh -> absolute pixel xyxy
            cx, cy, w, h = pred_boxes_b.unbind(-1)
            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height
            pred_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

            cached_predictions.append({
                "boxes": pred_xyxy.detach().cpu(),
                "scores": scores[b].detach().cpu(),
                "labels": labels[b].detach().cpu()
            })

            # Convert ground-truth boxes: normalized cxcywh -> absolute pixel xyxy
            gt_boxes_b = targets[b]["boxes"].to(device)
            gt_labels_b = targets[b]["labels"]

            if gt_boxes_b.numel() > 0:
                gt_cx, gt_cy, gt_w, gt_h = gt_boxes_b.unbind(-1)
                gt_x1 = (gt_cx - gt_w / 2) * image_width
                gt_y1 = (gt_cy - gt_h / 2) * image_height
                gt_x2 = (gt_cx + gt_w / 2) * image_width
                gt_y2 = (gt_cy + gt_h / 2) * image_height
                gt_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)
            else:
                gt_xyxy = torch.empty((0, 4), dtype=torch.float32)

            cached_targets.append({
                "boxes": gt_xyxy.detach().cpu(),
                "labels": gt_labels_b.detach().cpu()
            })

    print(f"Collected predictions for {len(cached_predictions)} images.")
    return cached_predictions, cached_targets


# ============================================================
# 4. IOU CALCULATION & ONE-TO-ONE MATCHING
# ============================================================
def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU calculation for boxes in absolute xyxy format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def match_image_class(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float = 0.50
):
    """Performs class-aware 1-to-1 matching for a single image and class.

    1. Sorts predictions by score descending.
    2. Matches each prediction to the unmatched GT box with the highest IoU.
    3. TP when IoU >= iou_thresh; once matched, GT cannot be matched again.
    4. Unmatched predictions = FP, unmatched GTs = FN.
    """
    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]

    if num_preds == 0:
        return 0, 0, num_gts
    if num_gts == 0:
        return 0, num_preds, 0

    sort_order = torch.argsort(pred_scores, descending=True)
    sorted_pred_boxes = pred_boxes[sort_order]

    ious = box_iou_xyxy(sorted_pred_boxes, gt_boxes)  # [num_preds, num_gts]

    matched_gt = set()
    tp = 0
    fp = 0

    for i in range(num_preds):
        best_iou = -1.0
        best_gt_idx = -1

        for j in range(num_gts):
            if j not in matched_gt:
                iou_val = ious[i, j].item()
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j

        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = num_gts - len(matched_gt)
    return tp, fp, fn


# ============================================================
# 5. PRECISION / RECALL / F1
# ============================================================
def compute_prf1(tp: int, fp: int, fn: int):
    """Calculates Precision, Recall, and F1 safely handling zero denominators."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================
# 6. CONFIDENCE SWEEP ON VALIDATION SET
# ============================================================
def sweep_confidence_thresholds(
    cached_predictions,
    cached_targets,
    num_classes=num_classes,
    iou_thresh=0.50,
    steps=201
):
    """Sweeps confidence thresholds from 0.0 to 1.0 (step 0.005) on validation cache.

    Finds the operating point maximizing F1 at IoU=0.50.
    """
    thresholds = torch.linspace(0.0, 1.0, steps)
    precision_curve = []
    recall_curve = []
    f1_curve = []

    print(f"Sweeping {steps} confidence thresholds across validation predictions...")

    for conf in thresholds:
        conf_val = conf.item()
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, tgt in zip(cached_predictions, cached_targets):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            keep = pred_scores >= conf_val
            filt_boxes = pred_boxes[keep]
            filt_scores = pred_scores[keep]
            filt_labels = pred_labels[keep]

            for c in range(num_classes):
                c_pred_mask = (filt_labels == c)
                c_gt_mask = (gt_labels == c)

                c_pred_boxes = filt_boxes[c_pred_mask]
                c_pred_scores = filt_scores[c_pred_mask]
                c_gt_boxes = gt_boxes[c_gt_mask]

                tp, fp, fn = match_image_class(
                    c_pred_boxes, c_pred_scores, c_gt_boxes, iou_thresh=iou_thresh
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

        p, r, f1 = compute_prf1(total_tp, total_fp, total_fn)
        precision_curve.append(p)
        recall_curve.append(r)
        f1_curve.append(f1)

    precision_curve = torch.tensor(precision_curve)
    recall_curve = torch.tensor(recall_curve)
    f1_curve = torch.tensor(f1_curve)

    best_index = torch.argmax(f1_curve).item()
    best_confidence = thresholds[best_index].item()
    best_precision = precision_curve[best_index].item()
    best_recall = recall_curve[best_index].item()
    best_f1 = f1_curve[best_index].item()

    print("\n" + "=" * 60)
    print("VALIDATION CONFIDENCE SWEEP RESULTS")
    print("=" * 60)
    print(f"Best validation confidence threshold: {best_confidence:.4f}")
    print(f"Validation Precision:                 {best_precision:.4f}")
    print(f"Validation Recall:                    {best_recall:.4f}")
    print(f"Validation F1:                        {best_f1:.4f}")
    print(f"IoU threshold:                        {iou_thresh:.2f}")
    print("=" * 60 + "\n")

    return {
        "thresholds": thresholds,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "f1_curve": f1_curve,
        "best_index": best_index,
        "best_confidence": best_confidence,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
    }


# ============================================================
# 7 & 8. PLOTTING PR & F1-CONFIDENCE CURVES
# ============================================================
def plot_pr_curve(recall_curve: torch.Tensor, precision_curve: torch.Tensor, output_path: str = "detr_validation_pr_curve.png"):
    """Plots and saves the Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve.numpy(), precision_curve.numpy(), color="#1f77b4", lw=2.5, label="DETR PR Curve")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("DETR Validation Precision-Recall Curve", fontsize=14, pad=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_f1_confidence(
    thresholds: torch.Tensor,
    f1_curve: torch.Tensor,
    best_confidence: float,
    best_f1: float,
    output_path: str = "detr_validation_f1_confidence.png"
):
    """Plots and saves the F1-Confidence curve with marked best operating point."""
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds.numpy(), f1_curve.numpy(), color="#2ca02c", lw=2.5, label="F1 Curve")
    plt.scatter([best_confidence], [best_f1], color="red", s=90, zorder=5,
                label=f"Best F1={best_f1:.4f} @ conf={best_confidence:.3f}")
    plt.axvline(x=best_confidence, color="red", linestyle="--", alpha=0.6)
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.ylabel("F1", fontsize=12)
    plt.title("DETR Validation F1-Confidence Curve", fontsize=14, pad=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.legend(loc="lower center", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# 9. COCO-STYLE MAP EVALUATION (TORCHMETRICS)
# ============================================================
def compute_map(cached_predictions, cached_targets, min_filtering_conf=0.001):
    """Calculates COCO-style mAP@0.5 and mAP@0.5:0.95 using TorchMetrics.

    Uses full confidence ranking (excluding only near-zero predictions).
    """
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    formatted_preds = []
    formatted_targets = []

    for pred, tgt in zip(cached_predictions, cached_targets):
        keep = pred["scores"] >= min_filtering_conf
        formatted_preds.append({
            "boxes": pred["boxes"][keep],
            "scores": pred["scores"][keep],
            "labels": pred["labels"][keep]
        })
        formatted_targets.append({
            "boxes": tgt["boxes"],
            "labels": tgt["labels"]
        })

    metric.update(formatted_preds, formatted_targets)
    results = metric.compute()

    map_50 = results["map_50"].item()
    map_50_95 = results["map"].item()
    map_per_class = results.get("map_per_class", None)
    map_50_per_class = results.get("map_50_per_class", None)

    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "map_per_class": map_per_class,
        "map_50_per_class": map_50_per_class,
    }


# ============================================================
# 10 & 11. DATASET EVALUATION AT FROZEN CONFIDENCE THRESHOLD
# ============================================================
def evaluate_dataset(
    cached_predictions,
    cached_targets,
    conf_thresh: float,
    num_classes: int = num_classes,
    iou_thresh: float = 0.50
):
    """Evaluates dataset using frozen confidence threshold.

    Computes global and per-class TP, FP, FN, Precision, Recall, and F1.
    """
    per_class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(num_classes)}

    for pred, tgt in zip(cached_predictions, cached_targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]

        keep = pred_scores >= conf_thresh
        filt_boxes = pred_boxes[keep]
        filt_scores = pred_scores[keep]
        filt_labels = pred_labels[keep]

        for c in range(num_classes):
            c_pred_mask = (filt_labels == c)
            c_gt_mask = (gt_labels == c)

            c_pred_boxes = filt_boxes[c_pred_mask]
            c_pred_scores = filt_scores[c_pred_mask]
            c_gt_boxes = gt_boxes[c_gt_mask]

            tp, fp, fn = match_image_class(
                c_pred_boxes, c_pred_scores, c_gt_boxes, iou_thresh=iou_thresh
            )
            per_class_stats[c]["tp"] += tp
            per_class_stats[c]["fp"] += fp
            per_class_stats[c]["fn"] += fn

    total_tp = sum(per_class_stats[c]["tp"] for c in range(num_classes))
    total_fp = sum(per_class_stats[c]["fp"] for c in range(num_classes))
    total_fn = sum(per_class_stats[c]["fn"] for c in range(num_classes))

    total_p, total_r, total_f1 = compute_prf1(total_tp, total_fp, total_fn)

    per_class_results = {}
    for c in range(num_classes):
        tp = per_class_stats[c]["tp"]
        fp = per_class_stats[c]["fp"]
        fn = per_class_stats[c]["fn"]
        p, r, f1 = compute_prf1(tp, fp, fn)
        per_class_results[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1
        }

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": total_p,
        "recall": total_r,
        "f1": total_f1,
        "per_class": per_class_results
    }


# ============================================================
# 12. OBJECT DETECTION CONFUSION MATRIX
# ============================================================
def generate_confusion_matrix(
    cached_predictions,
    cached_targets,
    conf_thresh: float,
    num_classes: int = num_classes,
    iou_thresh: float = 0.50
) -> np.ndarray:
    """Builds an object detection confusion matrix of shape [num_classes + 1, num_classes + 1].

    Rows: Ground Truth class (0..10) and Background (11).
    Cols: Predicted class (0..10) and Background (11).

    Distinguishes:
    - Correct class detections: matrix[c, c]
    - Wrong-class detections: matrix[c_gt, c_pred] (IoU >= 0.50, but c_gt != c_pred)
    - Missed GT / False Negatives: matrix[c_gt, Background]
    - Unmatched predictions / False Positives: matrix[Background, c_pred]
    """
    bg_idx = num_classes  # 11
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for pred, tgt in zip(cached_predictions, cached_targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]

        keep = pred_scores >= conf_thresh
        filt_boxes = pred_boxes[keep]
        filt_scores = pred_scores[keep]
        filt_labels = pred_labels[keep]

        num_preds = filt_boxes.shape[0]
        num_gts = gt_boxes.shape[0]

        if num_preds == 0 and num_gts == 0:
            continue

        if num_preds == 0:
            for gt_lbl in gt_labels:
                matrix[gt_lbl.item(), bg_idx] += 1
            continue

        if num_gts == 0:
            for pred_lbl in filt_labels:
                matrix[bg_idx, pred_lbl.item()] += 1
            continue

        # Sort predictions by confidence descending
        sort_order = torch.argsort(filt_scores, descending=True)
        sorted_boxes = filt_boxes[sort_order]
        sorted_labels = filt_labels[sort_order]

        ious = box_iou_xyxy(sorted_boxes, gt_boxes)  # [num_preds, num_gts]

        matched_gts = set()

        for i in range(num_preds):
            pred_lbl = sorted_labels[i].item()

            best_iou = -1.0
            best_gt_idx = -1

            # Match to unmatched GT with highest IoU
            for j in range(num_gts):
                if j not in matched_gts:
                    iou_val = ious[i, j].item()
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = j

            if best_gt_idx >= 0 and best_iou >= iou_thresh:
                gt_lbl = gt_labels[best_gt_idx].item()
                matrix[gt_lbl, pred_lbl] += 1
                matched_gts.add(best_gt_idx)
            else:
                # Unmatched prediction -> FP (Background row, pred_lbl column)
                matrix[bg_idx, pred_lbl] += 1

        # Missed GT objects -> FN (gt_lbl row, Background column)
        for j in range(num_gts):
            if j not in matched_gts:
                matrix[gt_labels[j].item(), bg_idx] += 1

    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: dict,
    output_path: str = "detr_test_confusion_matrix.png"
):
    """Plots and saves the object detection confusion matrix."""
    labels = [class_names.get(i, f"Class {i}") for i in range(len(class_names))] + ["Background"]
    n = len(labels)

    plt.figure(figsize=(13, 11))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("DETR Test Confusion Matrix (IoU=0.50)", fontsize=15, pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, labels, fontsize=9)

    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)

    # Annotate cell values
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if matrix[i, j] > thresh else "black"
                plt.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# 13. MAIN EVALUATION PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DETR Final Evaluation Pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to best.pt checkpoint"
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=r"C:\Users\Aparajith\Downloads\PEP AML Project Sem 5\IEDXray\val.txt",
        help="Path to validation manifest"
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default=r"C:\Users\Aparajith\Downloads\PEP AML Project Sem 5\IEDXray\test.txt",
        help="Path to test manifest"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback check inside DETR directory
        alt_path = Path(__file__).parent / args.checkpoint
        if alt_path.exists():
            ckpt_path = alt_path

    # 1. Load best.pt model
    model = load_detr_checkpoint(ckpt_path, device)

    # 2 & 3. Validation inference (collected once)
    print("\n--- Step 1: Loading Validation Dataset ---")
    val_ds = IEDXRayDataset(manifest_path=args.val_manifest, transform=transforms)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    val_preds, val_targets = collect_predictions(model, val_dl, device)

    # 6. Confidence sweep on validation set
    print("\n--- Step 2: Validation Confidence Sweep ---")
    sweep_results = sweep_confidence_thresholds(
        val_preds,
        val_targets,
        num_classes=num_classes,
        iou_thresh=0.50,
        steps=201
    )

    best_confidence = sweep_results["best_confidence"]

    # 7 & 8. Generate validation plots
    print("\n--- Step 3: Saving Validation Plots ---")
    plot_pr_curve(
        sweep_results["recall_curve"],
        sweep_results["precision_curve"],
        output_path="detr_validation_pr_curve.png"
    )
    plot_f1_confidence(
        sweep_results["thresholds"],
        sweep_results["f1_curve"],
        best_confidence,
        sweep_results["best_f1"],
        output_path="detr_validation_f1_confidence.png"
    )

    # 9. Validation mAP
    print("\n--- Step 4: Computing Validation mAP ---")
    val_map_results = compute_map(val_preds, val_targets, min_filtering_conf=0.001)
    print(f"Validation mAP@0.5:      {val_map_results['map_50']:.4f}")
    print(f"Validation mAP@0.5:0.95: {val_map_results['map_50_95']:.4f}")

    # 10. Official Test set evaluation (using FROZEN best_confidence)
    print("\n--- Step 5: Official Test Set Evaluation ---")
    print(f"Using frozen validation-derived confidence threshold: {best_confidence:.4f}")

    test_ds = IEDXRayDataset(manifest_path=args.test_manifest, transform=transforms)
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    test_preds, test_targets = collect_predictions(model, test_dl, device)

    test_results = evaluate_dataset(
        test_preds,
        test_targets,
        conf_thresh=best_confidence,
        num_classes=num_classes,
        iou_thresh=0.50
    )

    test_map_results = compute_map(test_preds, test_targets, min_filtering_conf=0.001)

    # 12. Confusion Matrix on official test set
    print("\n--- Step 6: Generating Confusion Matrix ---")
    cm = generate_confusion_matrix(
        test_preds,
        test_targets,
        conf_thresh=best_confidence,
        num_classes=num_classes,
        iou_thresh=0.50
    )
    plot_confusion_matrix(cm, CLASS_NAMES, output_path="detr_test_confusion_matrix.png")

    # 13. Final output summary
    print("\n" + "=" * 60)
    print("DETR OFFICIAL TEST RESULTS")
    print("=" * 60)
    print(f"Confidence threshold: {best_confidence:.4f}")
    print("IoU threshold:        0.50\n")
    print(f"Precision:            {test_results['precision']:.4f}")
    print(f"Recall:               {test_results['recall']:.4f}")
    print(f"F1:                   {test_results['f1']:.4f}")
    print(f"mAP@0.5:              {test_map_results['map_50']:.4f}")
    print(f"mAP@0.5:0.95:         {test_map_results['map_50_95']:.4f}\n")
    print(f"TP: {test_results['tp']}")
    print(f"FP: {test_results['fp']}")
    print(f"FN: {test_results['fn']}\n")

    print("=" * 85)
    print(f"{'Class':<26} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 85)
    for c in range(num_classes):
        c_name = CLASS_NAMES[c]
        st = test_results["per_class"][c]
        print(f"{c_name:<26} | {st['precision']:<9.4f} | {st['recall']:<9.4f} | {st['f1']:<9.4f} | {st['tp']:<6} | {st['fp']:<6} | {st['fn']:<6}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
