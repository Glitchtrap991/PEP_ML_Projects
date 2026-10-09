import os
import json
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


# ============================================================
# SAGEMAKER PATHS
# ============================================================

DATA_DIR = Path(os.environ["SM_CHANNEL_TRAINING"])

LOCAL_DATA_DIR = Path("/tmp/iedxray")
shutil.copytree(DATA_DIR, LOCAL_DATA_DIR, dirs_exist_ok=True)
DATA_DIR = LOCAL_DATA_DIR

# Convert train/val manifests to absolute paths
for manifest_name in ("train.txt", "val.txt"):
    manifest = DATA_DIR / manifest_name
    paths = manifest.read_text().splitlines()

    absolute_paths = [
        str((DATA_DIR / p.lstrip("./")).resolve())
        for p in paths
        if p.strip()
    ]

    manifest.write_text("\n".join(absolute_paths) + "\n")


# Final model artifacts
MODEL_DIR = Path(os.environ["SM_MODEL_DIR"])

# Non-model artifacts
OUTPUT_DIR = Path(
    os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("IEDXRay YOLO 960px (v26-M) experiment")
print("=" * 70)

print("Dataset:    ", DATA_DIR)
print("Model dir:  ", MODEL_DIR)
print("Output dir: ", OUTPUT_DIR)

print("PyTorch:    ", torch.__version__)
print("CUDA:       ", torch.cuda.is_available())

gpu_count = torch.cuda.device_count()

if gpu_count == 0:
    raise RuntimeError("No CUDA GPUs detected.")

TRAIN_DEVICE = 0 if gpu_count == 1 else list(range(gpu_count))

print(f"Detected {gpu_count} GPU(s)")
print(f"Training device(s): {TRAIN_DEVICE}")


# ============================================================
# DATASET YAML
# ============================================================

dataset_yaml = Path("/tmp/dataset.yaml")

config = {
    "path": str(DATA_DIR),

    "train": "train.txt",
    "val": "val.txt",
    "test": "images/test",

    "names": {
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
    },
}


with open(dataset_yaml, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)


print("\nDataset YAML:")
print(dataset_yaml.read_text())


# ============================================================
# BASIC DATASET CHECK
# ============================================================

train_manifest = DATA_DIR / "train.txt"
val_manifest = DATA_DIR / "val.txt"

if not train_manifest.exists():
    raise FileNotFoundError(train_manifest)

if not val_manifest.exists():
    raise FileNotFoundError(val_manifest)

train_count = len(
    train_manifest.read_text().strip().splitlines()
)

val_count = len(
    val_manifest.read_text().strip().splitlines()
)

test_dir = DATA_DIR / "images" / "test"

test_count = len([
    p for p in test_dir.iterdir()
    if p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff"
    }
])

print("\nDataset counts")
print("Train:", train_count)
print("Val:  ", val_count)
print("Test: ", test_count)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


# IMPORTANT:
# Replace with the EXACT model used for your 640px baseline.
model = YOLO("yolo26m.pt")


train_results = model.train(
    data=str(dataset_yaml),

    epochs=50,

    # Main experimental variable
    imgsz=960,

    # Keep this identical to your baseline if possible.
    
    batch=16 * gpu_count, #variable then keep -1

    workers=8,

    device=TRAIN_DEVICE,
    amp=True,

    # Validation happens after each epoch.
    val=True,

    # Keep everything under SageMaker's model directory.
    project=str(MODEL_DIR),
    name="iedxray_960_medium",

    save=True,
    save_period=5,

    plots=True,
)


RUN_DIR = MODEL_DIR / "iedxray_960_medium"
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
LAST_WEIGHTS = RUN_DIR / "weights" / "last.pt"


if not BEST_WEIGHTS.exists():
    raise FileNotFoundError(
        f"best.pt was not produced: {BEST_WEIGHTS}"
    )


print("\nTraining complete.")
print("Best:", BEST_WEIGHTS)
print("Last:", LAST_WEIGHTS)


# ============================================================
# FINAL VALIDATION USING BEST.PT
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

best_model = YOLO(str(BEST_WEIGHTS))


val_metrics = best_model.val(
    data=str(dataset_yaml),

    split="val",
    imgsz=960,

    device=0,

    plots=True,
    save_json=True,

    project=str(OUTPUT_DIR),
    name="validation_best",
)


# ============================================================
# OFFICIAL TEST SET
# ============================================================

print("\n" + "=" * 70)
print("OFFICIAL TEST EVALUATION")
print("=" * 70)


test_metrics = best_model.val(
    data=str(dataset_yaml),

    split="test",
    imgsz=960,

    device=0,

    plots=True,
    save_json=True,

    project=str(OUTPUT_DIR),
    name="official_test",
)


# ============================================================
# SAVE SUMMARY AS JSON
# ============================================================

summary = {

    "experiment": {
        "model": "yolo26n.pt",
        "imgsz": 960,
        "epochs": 50,
        "batch": 16,

        "train_images": train_count,
        "val_images": val_count,
        "test_images": test_count,
    },

    "validation": {
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "map50": float(val_metrics.box.map50),
        "map75": float(val_metrics.box.map75),
        "map50_95": float(val_metrics.box.map),
    },

    "official_test": {
        "precision": float(test_metrics.box.mp),
        "recall": float(test_metrics.box.mr),
        "map50": float(test_metrics.box.map50),
        "map75": float(test_metrics.box.map75),
        "map50_95": float(test_metrics.box.map),
    },
}


summary_path = OUTPUT_DIR / "experiment_summary.json"

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)


print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(json.dumps(summary, indent=4))


# ============================================================
# SAVE AN EASY-TO-DOWNLOAD COPY OF BEST.PT
# ============================================================

shutil.copy2(
    BEST_WEIGHTS,
    MODEL_DIR / "best.pt"
)

if LAST_WEIGHTS.exists():
    shutil.copy2(
        LAST_WEIGHTS,
        MODEL_DIR / "last.pt"
    )


print("\nEverything finished successfully.")