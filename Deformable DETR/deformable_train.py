import os
from pathlib import Path
import json
import csv
from torchvision.ops import box_convert

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from deformable_dataset import IEDXRayDeformableDataset, collate_fn
from deformable_pretrained import model, processor


# ============================================================
# CONFIG
# ============================================================

NUM_EPOCHS = 10
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
NUM_WORKERS = 12

LEARNING_RATE = 2e-5
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 0.1

# Confidence threshold for mAP calculation.
# Filtering queries below 0.15 cuts CPU box sorting in TorchMetrics by >70%
MAP_SCORE_THRESHOLD = 0.15

# Evaluate full mAP every N epochs (and always on the last epoch) to avoid burning GPU time on CPU mAP compute
MAP_EVAL_INTERVAL = 2

# Local Windows path OR SageMaker dataset channel
DATASET_ROOT = Path(
    os.environ.get(
        "SM_CHANNEL_DATASET",
        r"C:\Users\Aparajith\Downloads\PEP AML Project Sem 5\IEDXray"
    )
)

# SageMaker will persist /opt/ml/model as model.tar.gz.
# Locally this just becomes ./checkpoints

if "SM_MODEL_DIR" in os.environ:
    CHECKPOINT_DIR = Path(os.environ["SM_MODEL_DIR"])
else:
    CHECKPOINT_DIR = Path("checkpoints")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_deformable_detr.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_deformable_detr.pt"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEFORMABLE DETR TRAINING")
print("=" * 70)

print(f"Device       : {device}")
print(f"Dataset root : {DATASET_ROOT}")
print(f"Checkpoint   : {CHECKPOINT_DIR}")

if device.type == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM         : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# DATASETS
# ============================================================

train_dataset = IEDXRayDeformableDataset(
    DATASET_ROOT / "train.txt",
    processor,
)

val_dataset = IEDXRayDeformableDataset(
    DATASET_ROOT / "val.txt",
    processor,
)

print(f"\nTrain images : {len(train_dataset)}")
print(f"Val images   : {len(val_dataset)}")


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_fn,
    persistent_workers=(NUM_WORKERS > 0),
)


# ============================================================
# MODEL
# ============================================================

model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

# Give the pretrained EfficientNet backbone a smaller LR.
backbone_params = []
other_params = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if name.startswith("model.backbone"):
        backbone_params.append(parameter)
    else:
        other_params.append(parameter)


optimizer = torch.optim.AdamW(
    [
        {
            "params": other_params,
            "lr": LEARNING_RATE,
        },
        {
            "params": backbone_params,
            "lr": BACKBONE_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,
    gamma=0.1,
)


# ============================================================
# AMP
# ============================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# ============================================================
# HELPERS
# ============================================================

def move_labels_to_device(labels, device):
    """
    Move every tensor in each target dictionary to the GPU.
    """

    moved_labels = []

    for target in labels:

        moved_target = {}

        for key, value in target.items():

            if torch.is_tensor(value):
                moved_target[key] = value.to(
                    device,
                    non_blocking=True,
                )
            else:
                moved_target[key] = value

        moved_labels.append(moved_target)

    return moved_labels


def cxcywh_to_xyxy(boxes):
    """
    Convert normalized:

        [cx, cy, width, height]

    into normalized:

        [x1, y1, x2, y2]
    """

    cx, cy, w, h = boxes.unbind(-1)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )


def prepare_predictions_for_map(outputs, labels):
    """
    Convert Deformable DETR outputs into the format expected
    by TorchMetrics MeanAveragePrecision.

    Predictions:
        boxes  -> absolute xyxy
        scores -> confidence
        labels -> class ID

    Targets:
        boxes  -> absolute xyxy
        labels -> class ID
    """

    probabilities = outputs.logits.sigmoid()

    predictions = []
    targets = []

    batch_size = outputs.logits.shape[0]

    for i in range(batch_size):

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        # Deformable DETR uses sigmoid class predictions.
        #
        # For each query choose the highest scoring class.
        scores, pred_labels = probabilities[i].max(dim=-1)

        keep = scores > MAP_SCORE_THRESHOLD

        scores = scores[keep]
        pred_labels = pred_labels[keep]

        pred_boxes = outputs.pred_boxes[i][keep]

        # Normalized cxcywh -> normalized xyxy
        pred_boxes = cxcywh_to_xyxy(pred_boxes)

        # Processor stores resized image size as [height, width]
        image_size = labels[i]["size"]

        height = image_size[0]
        width = image_size[1]

        scale = torch.stack(
            [width, height, width, height]
        ).to(
            device=pred_boxes.device,
            dtype=pred_boxes.dtype,
        )

        pred_boxes = pred_boxes * scale

        predictions.append(
            {
                "boxes": pred_boxes.detach().cpu(),
                "scores": scores.detach().cpu(),
                "labels": pred_labels.detach().cpu(),
            }
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        target_boxes = labels[i]["boxes"]

        # Processor gives normalized cxcywh targets.
        target_boxes = cxcywh_to_xyxy(target_boxes)

        target_scale = torch.stack(
            [width, height, width, height]
        ).to(
            device=target_boxes.device,
            dtype=target_boxes.dtype,
        )

        target_boxes = target_boxes * target_scale

        targets.append(
            {
                "boxes": target_boxes.detach().cpu(),
                "labels": labels[i][
                    "class_labels"
                ].detach().cpu(),
            }
        )

    return predictions, targets


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    val_map,
    val_map50,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_map": val_map,
        "val_map50": val_map50,
    }

    torch.save(checkpoint, path)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    num_epochs,
    checkpoint_dir,
):
    best_map = -1.0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "map50": [],
        "map50_95": [],
        "learning_rate": [],
    }

    for epoch in range(num_epochs):

        # ==============================
        # TRAIN
        # ==============================
        model.train()

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        total_batches = len(train_loader)

        for step, batch in enumerate(train_loader):

            pixel_values = batch["pixel_values"].to(
                device,
                non_blocking=True,
            )

            labels = [
                {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in target.items()
                }
                for target in batch["labels"]
            ]

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels,
                )

                loss = outputs.loss
                scaled_loss = loss / ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == total_batches:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == total_batches:
                current_avg = running_loss / (step + 1)
                print(
                    f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] "
                    f"Step [{step + 1:04d}/{total_batches:04d}] | "
                    f"Loss: {loss.item():.4f} (Avg: {current_avg:.4f})"
                )

        train_loss = running_loss / total_batches

        # ==============================
        # VALIDATION
        # ==============================
        model.eval()

        val_running_loss = 0.0

        should_compute_map = (
            ((epoch + 1) % MAP_EVAL_INTERVAL == 0)
            or ((epoch + 1) == num_epochs)
        )

        metric = None
        if should_compute_map:
            metric = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
            )

        with torch.no_grad():

            for batch in val_loader:

                pixel_values = batch["pixel_values"].to(
                    device,
                    non_blocking=True,
                )

                labels = [
                    {
                        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                        for k, v in target.items()
                    }
                    for target in batch["labels"]
                ]

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    outputs = model(
                        pixel_values=pixel_values,
                        labels=labels,
                    )

                val_running_loss += outputs.loss.item()

                if should_compute_map:
                    # ----------------------
                    # Predictions
                    # ----------------------
                    probabilities = outputs.logits.sigmoid()

                    for i in range(pixel_values.shape[0]):

                        scores, pred_labels = probabilities[i].max(dim=-1)

                        keep = scores > MAP_SCORE_THRESHOLD

                        scores = scores[keep]
                        pred_labels = pred_labels[keep]
                        pred_boxes = outputs.pred_boxes[i][keep]

                        # normalized cxcywh -> normalized xyxy
                        pred_boxes = box_convert(
                            pred_boxes,
                            in_fmt="cxcywh",
                            out_fmt="xyxy",
                        )

                        # labels["size"] is [height, width] tensor on device
                        # Vectorized scale avoiding CPU sync
                        h = labels[i]["size"][0]
                        w = labels[i]["size"][1]
                        scale = torch.stack([w, h, w, h]).to(
                            device=device,
                            dtype=pred_boxes.dtype,
                        )

                        pred_boxes = pred_boxes * scale

                        # ----------------------
                        # Ground truth
                        # ----------------------
                        gt_boxes = box_convert(
                            labels[i]["boxes"],
                            in_fmt="cxcywh",
                            out_fmt="xyxy",
                        )

                        gt_boxes = gt_boxes * scale

                        metric.update(
                            [
                                {
                                    "boxes": pred_boxes.cpu(),
                                    "scores": scores.cpu(),
                                    "labels": pred_labels.cpu(),
                                }
                            ],
                            [
                                {
                                    "boxes": gt_boxes.cpu(),
                                    "labels": labels[i]["class_labels"].cpu(),
                                }
                            ],
                        )

        val_loss = val_running_loss / len(val_loader)

        if should_compute_map:
            print("  -> Computing validation mAP on CPU...")
            results = metric.compute()
            map50 = results["map_50"].item()
            map50_95 = results["map"].item()
            map_str = f"mAP50 {map50:.4f} | mAP50-95 {map50_95:.4f}"
        else:
            map50 = float("nan")
            map50_95 = float("nan")
            map_str = f"mAP: skipped (eval every {MAP_EVAL_INTERVAL} ep)"

        # ==============================
        # HISTORY
        # ==============================

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["map50"].append(map50)
        history["map50_95"].append(map50_95)
        history["learning_rate"].append(current_lr)

        print(
            f"\nEpoch {epoch + 1:03d}/{num_epochs:03d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"{map_str} | "
            f"LR {current_lr:.2e}"
        )

        # ==============================
        # SAVE HISTORY
        # ==============================

        with open(
            checkpoint_dir / "training_history.json",
            "w",
        ) as f:
            json.dump(history, f, indent=4)

        with open(
            checkpoint_dir / "training_history.csv",
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "map50",
                "map50_95",
                "learning_rate",
            ])

            for j in range(len(history["epoch"])):
                writer.writerow([
                    history["epoch"][j],
                    history["train_loss"][j],
                    history["val_loss"][j],
                    history["map50"][j],
                    history["map50_95"][j],
                    history["learning_rate"][j],
                ])

        # ==============================
        # BEST MODEL
        # ==============================

        if should_compute_map and (map50_95 > best_map):

            best_map = map50_95

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "map50": map50,
                    "map50_95": map50_95,
                    "history": history,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"  -> NEW BEST: "
                f"mAP50-95={best_map:.4f}"
            )

        # ==============================
        # LAST MODEL
        # ==============================

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "map50": map50,
                "map50_95": map50_95,
                "history": history,
            },
            LAST_MODEL_PATH,
        )

        scheduler.step()

    return history

# ============================================================
# ENTRY POINT
# ============================================================
def main():

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    print("\nTraining complete.")
    print(f"Artifacts saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()