"""
=============================================================================
Model Comparison with Grad-CAM: Deformable DETR vs YOLO_v26n vs YOLO_v26m
=============================================================================
This script loads three trained models:
  1. Deformable DETR (EfficientNet-B7 backbone)
  2. YOLO_v26n (Nano baseline)
  3. YOLO_v26m (Medium 960px variant)

For selected test images from the IEDXray test split, it produces a 1x4 side-by-side
comparison grid:
  [ Ground Truth | Deformable DETR | YOLO_v26n (Nano) | YOLO_v26m (Med 960px) ]
with Grad-CAM heatmap overlays, detected bounding boxes, class labels, and confidences.

Usage:
  python compare_models_gradcam.py --samples 10 --seed 42 --conf 0.25
  python compare_models_gradcam.py --image Test000001.jpg
=============================================================================
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from timm.layers import freeze_batch_norm_2d
from transformers import (
    AutoImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    TimmBackboneConfig,
)
from ultralytics import YOLO

# Add Deformable DETR directory to path if needed for local modules
REPO_ROOT = Path(__file__).resolve().parent
DEFORMABLE_DIR = REPO_ROOT / "Deformable DETR"
if str(DEFORMABLE_DIR) not in sys.path:
    sys.path.insert(0, str(DEFORMABLE_DIR))

# =============================================================================
# CONSTANTS & LABEL MAP
# =============================================================================

ID2LABEL = {
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
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES = 11

COLOR_MAP = {
    0: "#E63946",   # Explosive - Red
    1: "#F4A261",   # Battery - Orange
    2: "#2A9D8F",   # Modified laptop - Teal
    3: "#E76F51",   # Modified parts - Coral
    4: "#264653",   # Modified Mobile - Dark Slate
    5: "#9B5DE5",   # Modified Pager - Purple
    6: "#F15BB5",   # Modified Walkie - Pink
    7: "#00BBF9",   # Laptop - Cyan
    8: "#00F5D4",   # Pager - Mint
    9: "#FEE440",   # Mobile Phone - Yellow
    10: "#3A86FF",  # Walkie-Talkie - Blue
}


# =============================================================================
# DEFORMABLE DETR HELPERS & GRAD-CAM
# =============================================================================

def build_deformable_detr(checkpoint_path: Path, device: torch.device):
    """Build and load trained Deformable DETR model with EfficientNet-B7 backbone."""
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    processor.size = {"height": 640, "width": 640}
    processor.do_resize = True

    backbone_name = "tf_efficientnet_b7"
    out_indices = (2, 3, 4)

    backbone_config = TimmBackboneConfig(backbone=backbone_name)
    backbone_config.out_indices = list(out_indices)

    config = DeformableDetrConfig.from_pretrained(
        "SenseTime/deformable-detr",
        backbone_config=backbone_config,
        num_feature_levels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model = DeformableDetrForObjectDetection(config)

    backbone = timm.create_model(
        backbone_name,
        pretrained=False,
        features_only=True,
        out_indices=out_indices,
    )
    backbone = freeze_batch_norm_2d(backbone)
    assert backbone.feature_info.channels() == model.model.backbone.intermediate_channel_sizes

    model.model.backbone.model = backbone

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Deformable DETR checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, processor


class DeformableDETRGradCAM:
    """Grad-CAM for EfficientNet-B7 Deformable DETR backbone."""

    def __init__(self, model):
        self.model = model
        self.target_layer = model.model.backbone.model.blocks[-1]
        self.activations = None
        self.gradients = None
        self.handle = self.target_layer.register_forward_hook(self._capture)

    def _capture(self, module, inp, output):
        if isinstance(output, (tuple, list)):
            output = output[-1]
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, pixel_values, query_index, class_id, output_size):
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        with torch.inference_mode(False), torch.enable_grad():
            cam_input = pixel_values.detach().clone().requires_grad_(True)
            outputs = self.model(pixel_values=cam_input)
            logits = outputs.logits

            target_score = logits[0, query_index, class_id]
            target_score.backward()

            if self.activations is None or self.gradients is None:
                return None

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam, size=tuple(output_size), mode="bilinear", align_corners=False
            )[0, 0]

            cam = cam - cam.min()
            cam = cam / cam.max().clamp_min(1e-8)
            result = cam.detach().cpu().numpy()

        self.activations = None
        self.gradients = None
        return result

    def remove_hooks(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =============================================================================
# YOLO GRAD-CAM
# =============================================================================

class YOLOGradCAM:
    """Grad-CAM implementation for Ultralytics YOLO models."""

    def __init__(self, yolo_model, target_layer_idx=22):
        self.model = yolo_model.model
        self.model.eval()
        if target_layer_idx < len(self.model.model):
            self.target_layer = self.model.model[target_layer_idx]
        else:
            self.target_layer = self.model.model[-2]

        self.activations = None
        self.gradients = None
        self.handle = self.target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, output):
        if isinstance(output, (tuple, list)):
            output = output[-1]
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, img_tensor, class_id=None, output_size=None):
        """
        img_tensor: [1, 3, H, W] float tensor on model device with requires_grad=True
        output_size: (H, W) tuple for resizing the resulting CAM
        """
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        model_device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(model_device)

        with torch.enable_grad():
            preds = self.model(img_tensor)
            output_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds

            if class_id is not None and output_tensor.shape[1] > 4 + class_id:
                score = output_tensor[0, 4 + class_id, :].max()
            else:
                score = output_tensor[0, 4:, :].max()

            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                return None

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            if output_size is not None:
                cam = F.interpolate(
                    cam, size=tuple(output_size), mode="bilinear", align_corners=False
                )[0, 0]
            else:
                cam = cam[0, 0]

            cam = cam - cam.min()
            cam = cam / cam.max().clamp_min(1e-8)
            result = cam.detach().cpu().numpy()

        self.activations = None
        self.gradients = None
        return result

    def remove_hooks(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =============================================================================
# GROUND TRUTH & DATASET UTILS
# =============================================================================

def load_ground_truth(label_path: Path, img_w: int, img_h: int):
    """Load ground truth boxes and labels from YOLO format .txt file."""
    if not label_path.is_file():
        return []

    gt_items = []
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = max(0.0, (cx - bw / 2.0) * img_w)
            y1 = max(0.0, (cy - bh / 2.0) * img_h)
            x2 = min(float(img_w), (cx + bw / 2.0) * img_w)
            y2 = min(float(img_h), (cy + bh / 2.0) * img_h)
            gt_items.append({
                "class_id": cid,
                "class_name": ID2LABEL.get(cid, str(cid)),
                "box": [x1, y1, x2, y2],
            })
    return gt_items


# =============================================================================
# INFERENCE RUNNERS
# =============================================================================

def run_deformable_detr_inference(
    model, processor, gradcam, image: Image.Image, conf_threshold: float, device: torch.device
):
    """Run Deformable DETR forward pass and generate Grad-CAM for best detection."""
    w, h = image.size
    encoding = processor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    postprocessed = processor.post_process_object_detection(
        outputs,
        threshold=0.0,
        target_sizes=torch.tensor([[h, w]], device=device),
    )[0]

    scores = postprocessed["scores"]
    labels = postprocessed["labels"]
    boxes = postprocessed["boxes"]

    num_queries, num_classes = outputs.logits[0].shape
    flat_scores = outputs.logits[0].sigmoid().flatten()
    topk = min(100, flat_scores.numel())
    _, top_indices = flat_scores.topk(topk)
    query_indices = top_indices // num_classes

    keep = scores >= conf_threshold
    pred_scores = scores[keep]
    pred_labels = labels[keep]
    pred_boxes = boxes[keep]
    pred_queries = query_indices[keep]

    detections = []
    for s, l, b, q in zip(pred_scores, pred_labels, pred_boxes, pred_queries):
        detections.append({
            "class_id": int(l.item()),
            "class_name": ID2LABEL.get(int(l.item()), str(l.item())),
            "confidence": float(s.item()),
            "box": b.tolist(),
            "query_index": int(q.item()),
        })

    cam = None
    if len(detections) > 0:
        best_det = detections[0]
        cam = gradcam.generate(
            pixel_values=pixel_values,
            query_index=best_det["query_index"],
            class_id=best_det["class_id"],
            output_size=(h, w),
        )

    return detections, cam


def run_yolo_inference(
    yolo_model, gradcam, image: Image.Image, conf_threshold: float, device: torch.device, imgsz: int = 640
):
    """Run YOLO inference and generate Grad-CAM for best detection."""
    w, h = image.size
    results = yolo_model.predict(
        image,
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    boxes = results.boxes
    detections = []
    if len(boxes) > 0:
        for box in boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class_id": cid,
                "class_name": ID2LABEL.get(cid, str(cid)),
                "confidence": conf,
                "box": xyxy,
            })

    cam = None
    if len(detections) > 0:
        best_det = detections[0]
        # Prepare resized tensor for backward pass
        resized_img = image.resize((imgsz, imgsz))
        img_np = np.array(resized_img).astype(np.float32) / 255.0
        model_device = next(yolo_model.model.parameters()).device
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(model_device)
        img_tensor.requires_grad_(True)

        cam = gradcam.generate(
            img_tensor=img_tensor,
            class_id=best_det["class_id"],
            output_size=(h, w),
        )

    return detections, cam


# =============================================================================
# DRAWING & GRID GENERATION
# =============================================================================

def draw_annotations(
    image: Image.Image,
    detections: list,
    cam: np.ndarray = None,
    title: str = "",
    is_gt: bool = False,
):
    """Create a matplotlib visual panel with bounding boxes and optional CAM overlay."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    img_np = np.asarray(image.convert("RGB"))
    ax.imshow(img_np)

    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.45)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cid = det["class_id"]
        color = COLOR_MAP.get(cid, "#00FF00" if is_gt else "#FF0000")

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=2.5,
        )
        ax.add_patch(rect)

        label_text = det["class_name"]
        if not is_gt and "confidence" in det:
            label_text += f" {det['confidence']:.2f}"

        ax.text(
            x1,
            max(0, y1 - 4),
            label_text,
            fontsize=8.5,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
                pad=1.5,
            ),
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def create_comparison_grid(
    image: Image.Image,
    gt_items: list,
    det_detr: list,
    cam_detr: np.ndarray,
    det_nano: list,
    cam_nano: np.ndarray,
    det_med: list,
    cam_med: np.ndarray,
    image_name: str,
    output_path: Path,
):
    """Combine 4 panels into a high-resolution 1x4 comparison grid."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=160)
    img_np = np.asarray(image.convert("RGB"))

    panels = [
        {
            "ax": axes[0],
            "title": f"Ground Truth ({len(gt_items)} objects)",
            "cam": None,
            "items": gt_items,
            "is_gt": True,
        },
        {
            "ax": axes[1],
            "title": (
                f"Deformable DETR\n"
                + (f"Top: {det_detr[0]['class_name']} ({det_detr[0]['confidence']:.2f})" if det_detr else "No detection")
            ),
            "cam": cam_detr,
            "items": det_detr,
            "is_gt": False,
        },
        {
            "ax": axes[2],
            "title": (
                f"YOLO_v26n (Nano 640px)\n"
                + (f"Top: {det_nano[0]['class_name']} ({det_nano[0]['confidence']:.2f})" if det_nano else "No detection")
            ),
            "cam": cam_nano,
            "items": det_nano,
            "is_gt": False,
        },
        {
            "ax": axes[3],
            "title": (
                f"YOLO_v26m (Medium 960px)\n"
                + (f"Top: {det_med[0]['class_name']} ({det_med[0]['confidence']:.2f})" if det_med else "No detection")
            ),
            "cam": cam_med,
            "items": det_med,
            "is_gt": False,
        },
    ]

    for p in panels:
        ax = p["ax"]
        ax.imshow(img_np)
        if p["cam"] is not None:
            ax.imshow(p["cam"], cmap="jet", alpha=0.45)

        for item in p["items"]:
            x1, y1, x2, y2 = item["box"]
            cid = item["class_id"]
            color = COLOR_MAP.get(cid, "#00FF00" if p["is_gt"] else "#FF3366")

            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.5,
            )
            ax.add_patch(rect)

            label_text = item["class_name"]
            if not p["is_gt"] and "confidence" in item:
                label_text += f" {item['confidence']:.2f}"

            ax.text(
                x1,
                max(0, y1 - 4),
                label_text,
                fontsize=8.5,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.85,
                    pad=1.5,
                ),
            )

        ax.set_title(p["title"], fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(f"Model Comparison & Grad-CAM Heatmaps: {image_name}", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Side-by-side Grad-CAM Comparison for 3 Models")
    parser.add_argument("--samples", type=int, default=10, help="Number of random test cases to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sample selection")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--output-dir", type=str, default="comparison_outputs", help="Output directory")
    parser.add_argument("--image", type=str, default=None, help="Specific test image name to evaluate (e.g. Test000001.jpg)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Paths
    dataset_root = REPO_ROOT / "IEDXray"
    test_images_dir = dataset_root / "images" / "test"
    test_labels_dir = dataset_root / "labels" / "test"

    output_dir = REPO_ROOT / args.output_dir
    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    detr_ckpt = DEFORMABLE_DIR / "deformable_model" / "best_deformable_detr.pt"
    yolo_nano_ckpt = REPO_ROOT / "YOLO_v26n" / "best_nano.pt"
    if not yolo_nano_ckpt.is_file():
        fallback = Path(r"C:\Users\Aparajith\runs\detect\IEDyolo\yolo_checkpoints\weights\best.pt")
        if fallback.is_file():
            yolo_nano_ckpt = fallback

    yolo_med_ckpt = REPO_ROOT / "yolov26_960_px" / "best.pt"

    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    # 1. Deformable DETR
    print(f"Loading Deformable DETR from: {detr_ckpt}")
    detr_model, detr_processor = build_deformable_detr(detr_ckpt, device)
    detr_gradcam = DeformableDETRGradCAM(detr_model)

    # 2. YOLO Nano
    print(f"Loading YOLO_v26n (Nano) from: {yolo_nano_ckpt}")
    yolo_nano = YOLO(str(yolo_nano_ckpt))
    yolo_nano.to(device)
    yolo_nano_gradcam = YOLOGradCAM(yolo_nano, target_layer_idx=22)

    # 3. YOLO Medium 960px
    print(f"Loading YOLO_v26m (Medium 960px) from: {yolo_med_ckpt}")
    yolo_med = YOLO(str(yolo_med_ckpt))
    yolo_med.to(device)
    yolo_med_gradcam = YOLOGradCAM(yolo_med, target_layer_idx=22)

    # Image Selection
    all_images = sorted([
        p for p in test_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    if args.image:
        selected_images = [test_images_dir / args.image]
        if not selected_images[0].is_file():
            raise FileNotFoundError(f"Specified image not found: {selected_images[0]}")
    else:
        rng = random.Random(args.seed)
        num_samples = min(args.samples, len(all_images))
        selected_images = rng.sample(all_images, num_samples)

    print("\n" + "=" * 70)
    print(f"PROCESSING {len(selected_images)} COMPARISON SAMPLES")
    print("=" * 70)

    summary_records = []

    for idx, img_path in enumerate(selected_images, start=1):
        print(f"[{idx}/{len(selected_images)}] Processing {img_path.name}...")
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Ground Truth
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_items = load_ground_truth(label_path, w, h)

        # 1. Deformable DETR
        det_detr, cam_detr = run_deformable_detr_inference(
            model=detr_model,
            processor=detr_processor,
            gradcam=detr_gradcam,
            image=image,
            conf_threshold=args.conf,
            device=device,
        )

        # 2. YOLO Nano (640px)
        det_nano, cam_nano = run_yolo_inference(
            yolo_model=yolo_nano,
            gradcam=yolo_nano_gradcam,
            image=image,
            conf_threshold=args.conf,
            device=device,
            imgsz=640,
        )

        # 3. YOLO Med (960px)
        det_med, cam_med = run_yolo_inference(
            yolo_model=yolo_med,
            gradcam=yolo_med_gradcam,
            image=image,
            conf_threshold=args.conf,
            device=device,
            imgsz=960,
        )

        # Save Grid Image
        grid_output_path = grids_dir / f"{img_path.stem}_comparison.png"
        create_comparison_grid(
            image=image,
            gt_items=gt_items,
            det_detr=det_detr,
            cam_detr=cam_detr,
            det_nano=det_nano,
            cam_nano=cam_nano,
            det_med=det_med,
            cam_med=cam_med,
            image_name=img_path.name,
            output_path=grid_output_path,
        )

        summary_records.append({
            "image": img_path.name,
            "grid_image": str(grid_output_path.relative_to(REPO_ROOT)),
            "ground_truth_count": len(gt_items),
            "deformable_detr_detections": len(det_detr),
            "yolo_nano_detections": len(det_nano),
            "yolo_med_detections": len(det_med),
        })

    # Cleanup hooks
    detr_gradcam.remove_hooks()
    yolo_nano_gradcam.remove_hooks()
    yolo_med_gradcam.remove_hooks()

    # Save summary JSON
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "confidence_threshold": args.conf,
            "samples_processed": len(selected_images),
            "device": str(device),
            "models": {
                "deformable_detr": str(detr_ckpt.relative_to(REPO_ROOT)),
                "yolo_nano": str(yolo_nano_ckpt.name),
                "yolo_med_960px": str(yolo_med_ckpt.relative_to(REPO_ROOT)),
            },
            "records": summary_records,
        }, f, indent=4)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Comparison grids saved to: {grids_dir}")
    print(f"Summary JSON saved to:     {summary_path}")


if __name__ == "__main__":
    main()
