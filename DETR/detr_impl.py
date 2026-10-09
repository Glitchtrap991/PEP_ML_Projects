from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os

# Modular imports
from dataset import IEDXRayDataset, transforms, collate_fn
from matcher import HungarianMatcher
from criterion import DETRLoss
from plots import plot_training_results



# ============================================================
# ARCHITECTURE HYPERPARAMETERS
# ============================================================
img_size = 640
num_classes = 11  # All XRay classes from IEDXRay
attention_heads = 8
encoder_blocks = 6
decoder_blocks = 6
embed_dim = 256  # Hidden dimension size for transformer projection
mlp_nodes = embed_dim * 4  # Used in Encoder & Decoder MLP layers


def build_backbone():
    """Initializes and freezes EfficientNet-B7 feature extractor backbone."""
    try:
        backbone = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
    except Exception:
        backbone = models.efficientnet_b7(pretrained=True)

    for param in backbone.parameters():
        param.requires_grad = False

    backbone.classifier = nn.Identity()
    return backbone


# ============================================================
# TRANSFORMER MODULES
# ============================================================
class Encoder(nn.Module):
    def __init__(self, d_model=embed_dim, n_heads=attention_heads, mlp_dim=mlp_nodes):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.MHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
        self.MLP_Enc = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model)
        )

    def forward(self, x, pos):
        residual1 = x
        x = self.layer_norm1(x)

        q = x + pos
        k = x + pos
        v = x

        x, _ = self.MHA(q, k, v)
        x = x + residual1

        residual2 = x
        x = self.layer_norm2(x)
        x = self.MLP_Enc(x)
        x = x + residual2
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=embed_dim, n_heads=attention_heads, mlp_dim=mlp_nodes):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.MHA1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)  # self-attention
        self.MHA2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)  # cross-attention
        self.MLP_Dec = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model)
        )

    def forward(self, x, enc_out, query_pos, pos):
        # Self-Attention
        residual1 = x
        x = self.layer_norm1(x)

        q = x + query_pos
        k = x + query_pos
        v = x

        x, _ = self.MHA1(q, k, v)
        x = x + residual1

        # Cross-Attention
        residual2 = x
        x = self.layer_norm2(x)

        q = x + query_pos  # object queries
        k = enc_out + pos  # encoder features with positional encodings
        v = enc_out

        x, _ = self.MHA2(q, k, v)
        x = x + residual2

        # Feed Forward
        residual3 = x
        x = self.layer_norm3(x)
        x = self.MLP_Dec(x)
        x = x + residual3
        return x


class DETR(nn.Module):
    def __init__(
        self,
        num_classes=num_classes,
        d_model=embed_dim,
        n_heads=attention_heads,
        num_encoder_layers=encoder_blocks,
        num_decoder_layers=decoder_blocks,
        num_queries=100
    ):
        super().__init__()
        self.backbone = build_backbone()
        # EfficientNet-B7 output channels = 2560
        self.projection = nn.Conv2d(2560, d_model, kernel_size=1)
        self.encoders = nn.ModuleList(
            Encoder(d_model=d_model, n_heads=n_heads) for _ in range(num_encoder_layers)
        )
        self.decoders = nn.ModuleList(
            Decoder(d_model=d_model, n_heads=n_heads) for _ in range(num_decoder_layers)
        )
        self.classification_head = nn.Linear(d_model, num_classes + 1)
        self.bbox = nn.Linear(d_model, 4)

        self.query_pos_embed = nn.Parameter(torch.randn(num_queries, d_model))
        self.row_embed = nn.Parameter(torch.randn(50, d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(50, d_model // 2))

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.projection(x)  # [B, D, H, W]

        H, W = x.shape[-2:]
        pos_embed = torch.cat([
            self.row_embed[:H].unsqueeze(0).repeat(H, 1, 1),
            self.col_embed[:W].unsqueeze(1).repeat(1, W, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        B = x.shape[0]
        x = x.flatten(2).permute(2, 0, 1)  # [HW, B, D]
        pos_embed = pos_embed.repeat(1, B, 1)

        # Encoder pass
        for encoder in self.encoders:
            x = encoder(x, pos_embed)

        # Decoder pass
        query_pos = self.query_pos_embed.unsqueeze(1).repeat(1, B, 1)
        tgt = torch.zeros_like(query_pos)

        for decoder in self.decoders:
            tgt = decoder(tgt, x, query_pos, pos_embed)

        tgt = tgt.transpose(0, 1)  # [Q, B, D] -> [B, Q, D]

        return {
            "pred_logits": self.classification_head(tgt),
            "pred_boxes": self.bbox(tgt).sigmoid()
        }


# ============================================================
# MAIN TRAINING & VALIDATION PIPELINE
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    import os

    dataset_root = Path(
        os.environ.get(
            "SM_CHANNEL_DATASET",
            r"C:\Users\Aparajith\Downloads\PEP AML Project Sem 5\IEDXray"
        )
    )

    # Datasets
    train_ds = IEDXRayDataset(
        manifest_path=dataset_root / "train.txt",
        transform=transforms
    )
    val_ds = IEDXRayDataset(
        manifest_path=dataset_root / "val.txt",
        transform=transforms
    )

    BATCH_SIZE = 16
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Model, Matcher, Criterion
    model = DETR(num_classes=num_classes).to(device)
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = DETRLoss(
        num_classes=num_classes,
        matcher=matcher,
        eos_coef=0.1,
        lambda_bbox=5.0,
        lambda_giou=2.0
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW([
        {
            "params": model.backbone.parameters(),
            "lr": 1e-5
        },
        {
            "params": [
                p for name, p in model.named_parameters()
                if not name.startswith("backbone.")
            ],
            "lr": 1e-4
        }
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=40,
        gamma=0.1
    )

    # Training Setup
    num_epochs = 50
    checkpoint_dir = Path(
        os.environ.get("SM_CHECKPOINT_DIR", "checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_map = -1.0

    # Loss History
    train_total_history = []
    train_ce_history = []
    train_bbox_history = []
    train_giou_history = []

    val_total_history = []
    val_ce_history = []
    val_bbox_history = []
    val_giou_history = []

    val_map50_history = []
    val_map_history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================
    scaler = torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))
    for epoch in range(num_epochs):
        # ----------------- Train -----------------
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_bbox = 0.0
        running_giou = 0.0

        for images, targets in train_dl:
            images = images.to(device)
            targets = [
                {
                    "labels": t["labels"].to(device),
                    "boxes": t["boxes"].to(device)
                }
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = losses["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {loss.item()}")

            scaler.scale(loss).backward()

            # Required before gradient clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.1
            )

            scaler.step(optimizer)
            scaler.update()

            running_loss += losses["loss"].item()
            running_ce += losses["loss_ce"].item()
            running_bbox += losses["loss_bbox"].item()
            running_giou += losses["loss_giou"].item()

        train_loss = running_loss / len(train_dl)
        train_ce = running_ce / len(train_dl)
        train_bbox = running_bbox / len(train_dl)
        train_giou = running_giou / len(train_dl)

        train_total_history.append(train_loss)
        train_ce_history.append(train_ce)
        train_bbox_history.append(train_bbox)
        train_giou_history.append(train_giou)

        # ----------------- Validation -----------------
        model.eval()

        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True
        )
        val_running_loss = 0.0
        val_running_ce = 0.0
        val_running_bbox = 0.0
        val_running_giou = 0.0

        with torch.no_grad():
            for images, targets in val_dl:
                images = images.to(device)

                targets = [
                    {
                        "labels": t["labels"].to(device),
                        "boxes": t["boxes"].to(device)
                    }
                    for t in targets
                ]

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device.type == "cuda")
                ):
                    outputs = model(images)
                    losses = criterion(outputs, targets)

                # ========================================================
                # DETECTION METRICS
                # ========================================================
                probs = outputs["pred_logits"].softmax(-1)
                scores, labels = probs[..., :-1].max(-1)

                image_height = images.shape[2]
                image_width = images.shape[3]

                predictions = []
                metric_targets = []

                for b in range(images.shape[0]):
                    keep = scores[b] > 0.05
                    pred_b = outputs["pred_boxes"][b][keep]

                    cx, cy, w, h = pred_b.unbind(-1)
                    x1 = (cx - w / 2) * image_width
                    y1 = (cy - h / 2) * image_height
                    x2 = (cx + w / 2) * image_width
                    y2 = (cy + h / 2) * image_height
                    pred_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                    predictions.append({
                        "boxes": pred_xyxy.detach(),
                        "scores": scores[b][keep].detach(),
                        "labels": labels[b][keep].detach()
                    })

                    gt_boxes = targets[b]["boxes"]
                    gt_cx, gt_cy, gt_w, gt_h = gt_boxes.unbind(-1)
                    gt_x1 = (gt_cx - gt_w / 2) * image_width
                    gt_y1 = (gt_cy - gt_h / 2) * image_height
                    gt_x2 = (gt_cx + gt_w / 2) * image_width
                    gt_y2 = (gt_cy + gt_h / 2) * image_height
                    gt_xyxy = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)

                    metric_targets.append({
                        "boxes": gt_xyxy.detach(),
                        "labels": targets[b]["labels"].detach()
                    })

                metric.update(predictions, metric_targets)

                val_running_loss += losses["loss"].item()
                val_running_ce += losses["loss_ce"].item()
                val_running_bbox += losses["loss_bbox"].item()
                val_running_giou += losses["loss_giou"].item()

        val_loss = val_running_loss / len(val_dl)
        val_ce = val_running_ce / len(val_dl)
        val_bbox = val_running_bbox / len(val_dl)
        val_giou = val_running_giou / len(val_dl)

        metric_results = metric.compute()

        val_map = metric_results["map"].item()
        val_map50 = metric_results["map_50"].item()

        val_map50_history.append(val_map50)
        val_map_history.append(val_map)

        val_total_history.append(val_loss)
        val_ce_history.append(val_ce)
        val_bbox_history.append(val_bbox)
        val_giou_history.append(val_giou)

        scheduler.step()

        # ----------------- Logging -----------------
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )
        print(
            f"    Train -> CE: {train_ce:.4f} | L1: {train_bbox:.4f} | GIoU: {train_giou:.4f}"
        )
        print(
            f"    Val   -> CE: {val_ce:.4f} | L1: {val_bbox:.4f} | GIoU: {val_giou:.4f}"
        )
        print(
            f"    Metrics -> "
            f"mAP50: {val_map50:.4f} | "
            f"mAP50-95: {val_map:.4f}"
        )

        # ----------------- Checkpointing -----------------
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_map50": val_map50,
                "val_map": val_map,
                "train_total_history": train_total_history,
                "val_total_history": val_total_history,
                "train_ce_history": train_ce_history,
                "val_ce_history": val_ce_history,
                "train_bbox_history": train_bbox_history,
                "val_bbox_history": val_bbox_history,
                "train_giou_history": train_giou_history,
                "val_giou_history": val_giou_history,
                "val_map50_history": val_map50_history,
                "val_map_history": val_map_history,
            },
            checkpoint_dir / "last.pt"
        )

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_map50": val_map50,
                    "val_map": val_map,
                    "val_map50_history": val_map50_history,
                    "val_map_history": val_map_history,
                },
                checkpoint_dir / "best.pt"
            )
            print(f"    Best checkpoint saved (val mAP50-95 = {val_map:.4f})")

    # ========================================================
    # PLOT AFTER TRAINING
    # ========================================================
    plot_training_results(
        num_epochs=num_epochs,
        train_total_history=train_total_history,
        val_total_history=val_total_history,
        train_ce_history=train_ce_history,
        val_ce_history=val_ce_history,
        train_bbox_history=train_bbox_history,
        val_bbox_history=val_bbox_history,
        train_giou_history=train_giou_history,
        val_giou_history=val_giou_history,
        output_dir="."
    )

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        val_map50_history,
        label="mAP@0.5"
    )

    plt.plot(
        epochs,
        val_map_history,
        label="mAP@0.5:0.95"
    )

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("DETR Validation mAP")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "detr_validation_map.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()