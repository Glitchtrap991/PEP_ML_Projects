import json
from pathlib import Path
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_JSON = Path(r"IEDXray\annotations\complete_train.json")
TEST_JSON  = Path(r"IEDXray\annotations\complete_test.json")

OUTPUT_ROOT = Path(r"IEDXray\labels")

# COCO category ID -> YOLO class ID
CATEGORY_MAP = {
    1: 0,   # Explosive
    2: 1,   # Battery
    3: 2,   # Modified laptop
    4: 3,   # Modified parts
    5: 4,   # Modified Mobile phone
    6: 5,   # Modified Pager
    7: 6,   # Modified Walkie Talkie
    8: 7,   # Laptop
    9: 8,   # Pager
    10: 9,  # Mobile Phone
    11: 10, # Walkie-Talkie
}

CLASS_NAMES = [
    "Explosive",
    "Battery",
    "Modified laptop",
    "Modified parts",
    "Modified Mobile phone",
    "Modified Pager",
    "Modified Walkie Talkie",
    "Laptop",
    "Pager",
    "Mobile Phone",
    "Walkie-Talkie",
]


# ============================================================
# CONVERTER
# ============================================================

def convert_coco_to_yolo(json_path: Path, split: str):

    print(f"\n{'=' * 70}")
    print(f"Processing: {json_path}")
    print(f"Split:      {split}")
    print(f"{'=' * 70}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print(f"Images:      {len(images):,}")
    print(f"Annotations: {len(annotations):,}")
    print(f"Categories:  {len(categories):,}")

    # --------------------------------------------------------
    # Validate categories
    # --------------------------------------------------------

    print("\nCategories in JSON:")

    for cat in categories:
        print(f"  COCO {cat['id']:>2} -> {cat['name']}")

        if cat["id"] not in CATEGORY_MAP:
            raise ValueError(
                f"Unknown category ID {cat['id']} ({cat['name']})"
            )

    # --------------------------------------------------------
    # Image lookup
    # --------------------------------------------------------

    image_lookup = {}

    for image in images:

        image_id = image["id"]

        if image_id in image_lookup:
            raise ValueError(
                f"Duplicate image ID detected: {image_id}"
            )

        image_lookup[image_id] = image

    # --------------------------------------------------------
    # Group annotations by image
    # --------------------------------------------------------

    annotations_by_image = defaultdict(list)

    seen_annotation_ids = set()

    for ann in annotations:

        ann_id = ann["id"]

        if ann_id in seen_annotation_ids:
            raise ValueError(
                f"Duplicate annotation ID: {ann_id}"
            )

        seen_annotation_ids.add(ann_id)

        image_id = ann["image_id"]

        if image_id not in image_lookup:
            raise ValueError(
                f"Annotation {ann_id} refers to missing "
                f"image_id={image_id}"
            )

        category_id = ann["category_id"]

        if category_id not in CATEGORY_MAP:
            raise ValueError(
                f"Unknown category ID {category_id}"
            )

        annotations_by_image[image_id].append(ann)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    labels_dir = OUTPUT_ROOT / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()

    invalid_boxes = 0
    clipped_boxes = 0
    written_boxes = 0
    empty_images = 0

    # --------------------------------------------------------
    # Convert each image
    # --------------------------------------------------------

    for image_id, image in image_lookup.items():

        file_name = image["file_name"]

        img_width = float(image["width"])
        img_height = float(image["height"])

        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Invalid dimensions for {file_name}: "
                f"{img_width} x {img_height}"
            )

        yolo_lines = []

        for ann in annotations_by_image.get(image_id, []):

            bbox = ann.get("bbox")

            if bbox is None or len(bbox) != 4:
                print(
                    f"WARNING: Invalid bbox in annotation "
                    f"{ann['id']}"
                )
                invalid_boxes += 1
                continue

            # COCO:
            # x = left edge
            # y = top edge
            # w = box width
            # h = box height

            x, y, w, h = map(float, bbox)

            if w <= 0 or h <= 0:
                print(
                    f"WARNING: Non-positive bbox "
                    f"in annotation {ann['id']}: {bbox}"
                )
                invalid_boxes += 1
                continue

            # ------------------------------------------------
            # Convert xywh -> xyxy
            # ------------------------------------------------

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # ------------------------------------------------
            # Clip to image boundaries
            # ------------------------------------------------

            old_box = (x1, y1, x2, y2)

            x1 = max(0.0, min(x1, img_width))
            y1 = max(0.0, min(y1, img_height))

            x2 = max(0.0, min(x2, img_width))
            y2 = max(0.0, min(y2, img_height))

            new_box = (x1, y1, x2, y2)

            if old_box != new_box:
                clipped_boxes += 1

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                invalid_boxes += 1
                continue

            # ------------------------------------------------
            # YOLO normalized xywh
            # ------------------------------------------------

            x_center = ((x1 + x2) / 2.0) / img_width
            y_center = ((y1 + y2) / 2.0) / img_height

            norm_width = box_width / img_width
            norm_height = box_height / img_height

            # Safety check
            values = [
                x_center,
                y_center,
                norm_width,
                norm_height
            ]

            if not all(0.0 <= v <= 1.0 for v in values):
                print(
                    f"WARNING: Normalization error "
                    f"in annotation {ann['id']}: {values}"
                )
                invalid_boxes += 1
                continue

            # ------------------------------------------------
            # COCO class -> YOLO class
            # ------------------------------------------------

            yolo_class = CATEGORY_MAP[ann["category_id"]]

            yolo_lines.append(
                f"{yolo_class} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{norm_width:.6f} "
                f"{norm_height:.6f}"
            )

            class_counts[yolo_class] += 1
            written_boxes += 1

        # ----------------------------------------------------
        # Create label file
        # ----------------------------------------------------

        label_name = Path(file_name).stem + ".txt"
        label_path = labels_dir / label_name

        if yolo_lines:

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
                f.write("\n")

        else:
            # Explicit empty label file.
            #
            # YOLO can operate without this file, but creating
            # it makes dataset auditing easier.
            label_path.touch()

            empty_images += 1

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(f"\nConversion complete: {split}")

    print(f"Images processed: {len(images):,}")
    print(f"Boxes written:    {written_boxes:,}")
    print(f"Empty images:     {empty_images:,}")
    print(f"Invalid boxes:    {invalid_boxes:,}")
    print(f"Clipped boxes:    {clipped_boxes:,}")

    print("\nClass distribution:")

    for class_id, name in enumerate(CLASS_NAMES):

        count = class_counts[class_id]

        print(
            f"  {class_id:>2} "
            f"{name:<28} "
            f"{count:>8,}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    convert_coco_to_yolo(
        TRAIN_JSON,
        "train"
    )

    convert_coco_to_yolo(
        TEST_JSON,
        "test"
    )