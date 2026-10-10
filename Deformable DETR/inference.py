from pathlib import Path
import json
import random
import timm
import torch
from PIL import Image
from timm.layers import freeze_batch_norm_2d
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    TimmBackboneConfig,
)

from xai.gradcam import DeformableDETRGradCAM


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = (Path(__file__).resolve().parent / "../IEDXray").resolve()

# "test" = untouched official test; "val" = training validation manifest.
EVAL_SPLIT = "test"

TEST_IMAGES = DATASET_ROOT / "images" / "test"
TEST_LABELS = DATASET_ROOT / "labels" / "test"

CHECKPOINT_PATH = Path(
    "deformable_model/best_deformable_detr.pt"
)

OUTPUT_DIR = Path("inference_outputs") / EVAL_SPLIT
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

BACKBONE_NAME = "tf_efficientnet_b7"
BACKBONE_OUT_INDICES = (2, 3, 4)

NUM_CLASSES = 11

# Detection confidence threshold used for P/R and visualizations.
# mAP itself should use a low threshold so the PR curve is preserved.
CONF_THRESHOLD = 0.25
MAP_SCORE_THRESHOLD = 0.001

# IoU threshold used for the reported Precision / Recall.
PR_IOU_THRESHOLD = 0.50

# Limit Grad-CAM generation because backward() per detection is expensive.
MAX_GRADCAM_IMAGES = 20
RANDOM_GRADCAM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


id2label = {
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
    10: "Walkie-Talkie",
}

label2id = {v: k for k, v in id2label.items()}


# ============================================================
# BUILD EXACT TRAINED ARCHITECTURE
# ============================================================

def build_model():

    processor = AutoImageProcessor.from_pretrained(
        "SenseTime/deformable-detr"
    )

    processor.size = {
        "height": 640,
        "width": 640,
    }

    processor.do_resize = True

    backbone_config = TimmBackboneConfig(
        backbone=BACKBONE_NAME
    )

    backbone_config.out_indices = list(
        BACKBONE_OUT_INDICES
    )

    config = DeformableDetrConfig.from_pretrained(
        "SenseTime/deformable-detr",
        backbone_config=backbone_config,
        num_feature_levels=4,
        id2label=id2label,
        label2id=label2id,
    )

    model = DeformableDetrForObjectDetection(
        config
    )

    # IMPORTANT:
    # pretrained=False because the final trained checkpoint
    # will overwrite ALL model weights anyway.
    backbone = timm.create_model(
        BACKBONE_NAME,
        pretrained=False,
        features_only=True,
        out_indices=BACKBONE_OUT_INDICES,
    )

    backbone = freeze_batch_norm_2d(
        backbone
    )

    assert (
        backbone.feature_info.channels()
        == model.model.backbone.intermediate_channel_sizes
    )

    model.model.backbone.model = backbone

    return model, processor


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(model):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    # Our training code saved a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch", "unknown")
        )

        print(
            "Checkpoint best mAP50-95:",
            checkpoint.get(
                "best_map",
                checkpoint.get(
                    "map_50_95",
                    "unknown"
                )
            )
        )

    else:
        # Supports a plain state_dict too.
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# YOLO TEST LABELS -> ABSOLUTE XYXY
# ============================================================

def load_ground_truth(
    label_path,
    image_width,
    image_height,
):

    boxes = []
    labels = []

    if not label_path.exists():
        return (
            torch.empty((0, 4)),
            torch.empty(
                (0,),
                dtype=torch.long,
            ),
        )

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            class_id, cx, cy, bw, bh = map(
                float,
                line.split(),
            )

            class_id = int(class_id)

            x1 = (cx - bw / 2) * image_width
            y1 = (cy - bh / 2) * image_height

            x2 = (cx + bw / 2) * image_width
            y2 = (cy + bh / 2) * image_height

            x1 = max(
                0.0,
                min(x1, image_width)
            )

            y1 = max(
                0.0,
                min(y1, image_height)
            )

            x2 = max(
                0.0,
                min(x2, image_width)
            )

            y2 = max(
                0.0,
                min(y2, image_height)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([
                x1,
                y1,
                x2,
                y2,
            ])

            labels.append(class_id)

    if boxes:

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32,
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    else:

        boxes = torch.empty(
            (0, 4),
            dtype=torch.float32,
        )

        labels = torch.empty(
            (0,),
            dtype=torch.long,
        )

    return boxes, labels


# ============================================================
# IOU
# ============================================================

def box_iou(box1, box2):

    x1 = max(
        box1[0].item(),
        box2[0].item(),
    )

    y1 = max(
        box1[1].item(),
        box2[1].item(),
    )

    x2 = min(
        box1[2].item(),
        box2[2].item(),
    )

    y2 = min(
        box1[3].item(),
        box2[3].item(),
    )

    intersection = max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )

    area1 = max(
        0.0,
        box1[2].item() - box1[0].item(),
    ) * max(
        0.0,
        box1[3].item() - box1[1].item(),
    )

    area2 = max(
        0.0,
        box2[2].item() - box2[0].item(),
    ) * max(
        0.0,
        box2[3].item() - box2[1].item(),
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# PRECISION / RECALL MATCHING
# ============================================================

def match_predictions(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
):

    """
    One-to-one class-aware matching at IoU >= 0.50.

    Predictions are matched highest-confidence first.

    Returns per-class TP / FP / FN.
    """

    stats = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for class_id in range(NUM_CLASSES)
    }

    for class_id in range(NUM_CLASSES):

        pred_mask = (
            pred_labels == class_id
        )

        gt_mask = (
            gt_labels == class_id
        )

        class_pred_boxes = pred_boxes[
            pred_mask
        ]

        class_pred_scores = pred_scores[
            pred_mask
        ]

        class_gt_boxes = gt_boxes[
            gt_mask
        ]

        if len(class_pred_scores) > 0:

            order = torch.argsort(
                class_pred_scores,
                descending=True,
            )

            class_pred_boxes = (
                class_pred_boxes[order]
            )

        matched_gt = set()

        for pred_box in class_pred_boxes:

            best_iou = 0.0
            best_gt = None

            for gt_index, gt_box in enumerate(
                class_gt_boxes
            ):

                if gt_index in matched_gt:
                    continue

                iou = box_iou(
                    pred_box,
                    gt_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_index

            if (
                best_gt is not None
                and best_iou >= PR_IOU_THRESHOLD
            ):

                stats[class_id]["tp"] += 1
                matched_gt.add(best_gt)

            else:

                stats[class_id]["fp"] += 1

        stats[class_id]["fn"] += (
            len(class_gt_boxes)
            - len(matched_gt)
        )

    return stats


# ============================================================
# TARGET LAYER FOR GRAD-CAM
# ============================================================

def get_gradcam_target_layer(model):

    """
    Use the last native EfficientNet feature stage.

    features_only EfficientNet exposes the underlying blocks.
    We hook the final block group so the tensor remains spatial.
    """

    backbone = model.model.backbone.model

    return backbone.blocks[-1]


# ============================================================
# SAVE GRADCAM OVERLAY
# ============================================================

def save_gradcam_overlay(
    image,
    cam,
    box,
    class_name,
    confidence,
    output_path,
):

    import numpy as np
    import matplotlib.pyplot as plt

    image_np = np.asarray(
        image.convert("RGB")
    )

    cam_np = cam.numpy()

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    ax.imshow(image_np)

    ax.imshow(
        cam_np,
        cmap="jet",
        alpha=0.40,
    )

    x1, y1, x2, y2 = box

    rectangle = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2,
    )

    ax.add_patch(rectangle)

    ax.text(
        x1,
        max(0, y1 - 5),
        f"{class_name} {confidence:.3f}",
        fontsize=10,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Device:", DEVICE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRADCAM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, processor = build_model()

    model = load_checkpoint(model)

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # TorchMetrics mAP
    # --------------------------------------------------------

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        # Standard COCO maxDets: [1, 10, 100].
        # Changing this to 300 can make the standard mAP summary -1.
    )

    # --------------------------------------------------------
    # P/R accumulators
    # --------------------------------------------------------

    total_stats = {
        class_id: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for class_id in range(NUM_CLASSES)
    }

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    target_layer = get_gradcam_target_layer(
        model
    )

    gradcam = DeformableDETRGradCAM(
        model=model,
        target_layer=target_layer,
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    if EVAL_SPLIT == "test":
        image_paths = sorted(
            p for p in TEST_IMAGES.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        print(f"Official test images: {len(image_paths)}")
        if len(image_paths) != 5136:
            print("WARNING: Expected 5,136 images in the official test split.")
    elif EVAL_SPLIT == "val":
        manifest = DATASET_ROOT / "val.txt"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing validation manifest: {manifest}")
        image_paths = [
            (DATASET_ROOT / line.strip().replace("\\", "/").removeprefix("./")).resolve()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"Validation images: {len(image_paths)}")
    else:
        raise ValueError("EVAL_SPLIT must be 'test' or 'val'.")

    if not image_paths:
        raise RuntimeError(f"No images found for EVAL_SPLIT={EVAL_SPLIT}.")

    for p in image_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing image: {p}")

    all_detections = []
    gradcam_candidates = []
    gradcam_images_saved = 0

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        image_paths
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        # images/test/x.jpg -> labels/test/x.txt
        # images/train/x.jpg -> labels/train/x.txt (validation manifest)
        relative_path = image_path.relative_to(DATASET_ROOT)
        parts = list(relative_path.parts)
        if parts[0].lower() != "images":
            raise ValueError(f"Expected image under images/: {image_path}")
        parts[0] = "labels"
        label_path = DATASET_ROOT.joinpath(*parts).with_suffix(".txt")

        gt_boxes, gt_labels = (
            load_ground_truth(
                label_path,
                width,
                height,
            )
        )

        encoding = processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = encoding[
            "pixel_values"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                pixel_values=pixel_values
            )

        # HF Deformable DETR postprocessing ranks query-class PAIRS,
        # not only the single highest class per query. Its standard
        # postprocessor returns the top 100 scored pairs per image.
        postprocessed = processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=torch.tensor([[height, width]], device=DEVICE),
        )[0]

        scores = postprocessed["scores"]
        labels = postprocessed["labels"]
        boxes_xyxy = postprocessed["boxes"]

        # Recover each returned detection's original query index for
        # detection-specific Grad-CAM. Same flatten/top-k order as HF.
        num_queries, num_classes = outputs.logits[0].shape
        flat_scores = outputs.logits[0].sigmoid().flatten()
        topk = min(100, flat_scores.numel())
        top_scores, top_indices = flat_scores.topk(topk)
        query_indices = top_indices // num_classes
        top_labels = top_indices % num_classes
        if (
            scores.numel() != topk
            or not torch.allclose(scores, top_scores, rtol=1e-4, atol=1e-6)
            or not torch.equal(labels, top_labels)
        ):
            raise RuntimeError(
                "HF postprocessor top-k does not match query-index recovery. "
                "Check the installed transformers version."
            )

        # ----------------------------------------------------
        # mAP predictions
        #
        # Keep almost everything for AP curve.
        # ----------------------------------------------------

        map_keep = (
            scores >= MAP_SCORE_THRESHOLD
        )

        map_prediction = {
            "boxes": (
                boxes_xyxy[map_keep]
                .detach()
                .cpu()
            ),
            "scores": (
                scores[map_keep]
                .detach()
                .cpu()
            ),
            "labels": (
                labels[map_keep]
                .detach()
                .cpu()
            ),
        }

        map_target = {
            "boxes": gt_boxes,
            "labels": gt_labels,
        }

        metric.update(
            [map_prediction],
            [map_target],
        )

        # ----------------------------------------------------
        # P/R predictions
        # ----------------------------------------------------

        keep = (
            scores >= CONF_THRESHOLD
        )

        pred_boxes = (
            boxes_xyxy[keep]
            .detach()
            .cpu()
        )

        pred_scores = (
            scores[keep]
            .detach()
            .cpu()
        )

        pred_labels = (
            labels[keep]
            .detach()
            .cpu()
        )

        pred_queries = (
            query_indices[keep]
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Match for P/R
        # ----------------------------------------------------

        image_stats = match_predictions(
            pred_boxes,
            pred_scores,
            pred_labels,
            gt_boxes,
            gt_labels,
        )

        for class_id in range(
            NUM_CLASSES
        ):

            for key in (
                "tp",
                "fp",
                "fn",
            ):

                total_stats[
                    class_id
                ][key] += image_stats[
                    class_id
                ][key]

        # ----------------------------------------------------
        # Save detections
        # ----------------------------------------------------

        image_detection_data = {
            "image": image_path.name,
            "detections": [],
        }

        for detection_index in range(
            len(pred_scores)
        ):

            class_id = int(
                pred_labels[
                    detection_index
                ].item()
            )

            score = float(
                pred_scores[
                    detection_index
                ].item()
            )

            query_index = int(
                pred_queries[
                    detection_index
                ].item()
            )

            box = (
                pred_boxes[
                    detection_index
                ].tolist()
            )

            image_detection_data[
                "detections"
            ].append({
                "query_index": query_index,
                "class_id": class_id,
                "class_name": id2label[
                    class_id
                ],
                "confidence": score,
                "bbox_xyxy": box,
            })

        all_detections.append(
            image_detection_data
        )

        # ----------------------------------------------------
        # Grad-CAM candidates
        # Collect detected cases to randomly sample from.
        # ----------------------------------------------------

        if len(pred_scores) > 0:
            best_index = int(
                torch.argmax(
                    pred_scores
                ).item()
            )
            gradcam_candidates.append({
                "image_path": image_path,
                "query_index": int(
                    pred_queries[
                        best_index
                    ].item()
                ),
                "class_id": int(
                    pred_labels[
                        best_index
                    ].item()
                ),
                "confidence": float(
                    pred_scores[
                        best_index
                    ].item()
                ),
                "box": pred_boxes[
                    best_index
                ].tolist(),
            })

        if (
            (image_index + 1) % 100
            == 0
        ):

            print(
                f"Processed "
                f"{image_index + 1}"
                f"/{len(image_paths)}"
            )

    # ========================================================
    # METRICS
    # ========================================================

    results = metric.compute()

    map_50 = float(
        results["map_50"]
    )

    map_50_95 = float(
        results["map"]
    )

    # --------------------------------------------------------
    # Overall micro Precision / Recall
    # --------------------------------------------------------

    total_tp = sum(
        x["tp"]
        for x in total_stats.values()
    )

    total_fp = sum(
        x["fp"]
        for x in total_stats.values()
    )

    total_fn = sum(
        x["fn"]
        for x in total_stats.values()
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 72)
    print(f"IEDXRAY {EVAL_SPLIT.upper()} RESULTS")
    print("=" * 72)

    print(
        f"Confidence threshold : "
        f"{CONF_THRESHOLD:.2f}"
    )

    print(
        f"P/R IoU threshold    : "
        f"{PR_IOU_THRESHOLD:.2f}"
    )

    print(
        f"Precision            : "
        f"{precision:.4f}"
    )

    print(
        f"Recall               : "
        f"{recall:.4f}"
    )

    print(
        f"mAP50                : "
        f"{map_50:.4f}"
    )

    print(
        f"mAP50-95             : "
        f"{map_50_95:.4f}"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Per-class P/R
    # --------------------------------------------------------

    print("\nPER-CLASS PRECISION / RECALL\n")

    print(
        f"{'ID':<4}"
        f"{'Class':<28}"
        f"{'P':>10}"
        f"{'R':>10}"
        f"{'TP':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
    )

    print("-" * 76)

    per_class_results = {}

    for class_id in range(
        NUM_CLASSES
    ):

        stats = total_stats[
            class_id
        ]

        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        class_precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        class_recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        print(
            f"{class_id:<4}"
            f"{id2label[class_id]:<28}"
            f"{class_precision:>10.4f}"
            f"{class_recall:>10.4f}"
            f"{tp:>8}"
            f"{fp:>8}"
            f"{fn:>8}"
        )

        per_class_results[
            str(class_id)
        ] = {
            "class_name": id2label[
                class_id
            ],
            "precision": class_precision,
            "recall": class_recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # --------------------------------------------------------
    # TorchMetrics per-class AP
    # --------------------------------------------------------

    if (
        "classes" in results
        and "map_per_class" in results
    ):

        print(
            "\nPER-CLASS mAP50-95\n"
        )

        for class_tensor, ap_tensor in zip(
            results["classes"],
            results["map_per_class"],
        ):

            class_id = int(
                class_tensor.item()
            )

            ap = float(
                ap_tensor.item()
            )

            print(
                f"{class_id:2d} "
                f"{id2label[class_id]:<28} "
                f"{ap:.4f}"
            )

            per_class_results.setdefault(
                str(class_id),
                {},
            )["map_50_95"] = ap

    # ========================================================
    # SAVE JSON
    # ========================================================

    summary = {
        "confidence_threshold": (
            CONF_THRESHOLD
        ),
        "pr_iou_threshold": (
            PR_IOU_THRESHOLD
        ),
        "precision": precision,
        "recall": recall,
        "map50": map_50,
        "map50_95": map_50_95,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class_results,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    with open(
        OUTPUT_DIR / "detections.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_detections,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Grad-CAM on randomly sampled cases
    # --------------------------------------------------------

    if MAX_GRADCAM_IMAGES > 0 and gradcam_candidates:
        num_samples = min(
            MAX_GRADCAM_IMAGES,
            len(gradcam_candidates),
        )
        rng = random.Random(RANDOM_GRADCAM_SEED)
        sampled_candidates = rng.sample(
            gradcam_candidates,
            num_samples,
        )

        print(
            f"\nGenerating Grad-CAM overlays for {num_samples} "
            f"randomly selected cases..."
        )

        for candidate in sampled_candidates:
            cand_image = Image.open(
                candidate["image_path"]
            ).convert("RGB")
            cand_w, cand_h = cand_image.size

            encoding = processor(
                images=cand_image,
                return_tensors="pt",
            )
            cand_pixel_values = encoding[
                "pixel_values"
            ].to(DEVICE)

            cam, _, _ = gradcam.generate(
                pixel_values=cand_pixel_values,
                query_index=candidate["query_index"],
                class_id=candidate["class_id"],
                output_size=(cand_h, cand_w),
            )

            output_path = (
                GRADCAM_DIR
                / (
                    f"{candidate['image_path'].stem}"
                    f"_q{candidate['query_index']}"
                    f"_c{candidate['class_id']}.png"
                )
            )

            save_gradcam_overlay(
                image=cand_image,
                cam=cam,
                box=candidate["box"],
                class_name=id2label[
                    candidate["class_id"]
                ],
                confidence=candidate["confidence"],
                output_path=output_path,
            )

            gradcam_images_saved += 1

    gradcam.remove_hooks()

    print(
        "\nSaved metrics to:",
        OUTPUT_DIR / "metrics.json"
    )

    print(
        "Saved detections to:",
        OUTPUT_DIR / "detections.json"
    )

    if gradcam_images_saved > 0:
        print(
            f"Saved {gradcam_images_saved} Grad-CAM images to:",
            GRADCAM_DIR
        )
    else:
        print(
            "No Grad-CAM images saved."
        )


if __name__ == "__main__":
    main()