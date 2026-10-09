from pathlib import Path
import random
import os

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"C:\Users\Aparajith\Downloads\PEP AML Project Sem 5\IEDXray")

TRAIN_IMAGE_DIR = DATASET_ROOT / "images" / "train"

VAL_FRACTION = 0.20
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# ============================================================
# FIND ALL TRAINING IMAGES
# ============================================================

images = sorted([
    p for p in TRAIN_IMAGE_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
])

print(f"Total official training images: {len(images):,}")

if not images:
    raise RuntimeError(
        f"No images found in {TRAIN_IMAGE_DIR}"
    )


# ============================================================
# REPRODUCIBLE SHUFFLE
# ============================================================

rng = random.Random(SEED)
rng.shuffle(images)


# ============================================================
# SPLIT
# ============================================================

num_val = round(len(images) * VAL_FRACTION)

val_images = images[:num_val]
train_images = images[num_val:]


print(f"Training:   {len(train_images):,}")
print(f"Validation: {len(val_images):,}")


# ============================================================
# VERIFY NO OVERLAP
# ============================================================

train_set = set(train_images)
val_set = set(val_images)

assert train_set.isdisjoint(val_set)
assert len(train_set | val_set) == len(images)

print("Train/validation overlap: 0")


# ============================================================
# WRITE MANIFEST FILES
# ============================================================

train_relative = [
    p.relative_to(DATASET_ROOT).as_posix()
    for p in train_images
]

val_relative = [
    p.relative_to(DATASET_ROOT).as_posix()
    for p in val_images
]

train_txt = DATASET_ROOT / "train.txt"
val_txt = DATASET_ROOT / "val.txt"


with open(train_txt, "w") as f:
    for image in train_relative:
        f.write(image + "\n")


with open(val_txt, "w") as f:
    for image in val_relative:
        f.write(image + "\n")


print(f"\nCreated:")
print(train_txt)
print(val_txt)