import box_ops
from pathlib import Path
import matplotlib.pyplot as plt
import os

def plot_training_results(
    num_epochs,
    train_total_history,
    val_total_history,
    train_ce_history,
    val_ce_history,
    train_bbox_history,
    val_bbox_history,
    train_giou_history,
    val_giou_history,
    output_dir="."
):
    """Generates and saves total loss and loss components plots."""
    output_dir = Path(
        os.environ.get("SM_MODEL_DIR", ".")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, num_epochs + 1)

    # ------------------------------------------------------------
    # 1. Total Loss Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_total_history, label="Training Loss")
    plt.plot(epochs, val_total_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DETR Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "detr_total_loss.png", dpi=300)
    plt.show()

    # ------------------------------------------------------------
    # 2. Individual Loss Components Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ce_history, label="Train Classification")
    plt.plot(epochs, val_ce_history, label="Val Classification")
    plt.plot(epochs, train_bbox_history, label="Train L1")
    plt.plot(epochs, val_bbox_history, label="Val L1")
    plt.plot(epochs, train_giou_history, label="Train GIoU")
    plt.plot(epochs, val_giou_history, label="Val GIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DETR Loss Components")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "detr_loss_components.png", dpi=300)
    plt.show()
