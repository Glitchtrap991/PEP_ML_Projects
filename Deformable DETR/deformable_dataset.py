from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class IEDXRayDeformableDataset(Dataset):
    def __init__(self, manifest_path, processor):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.processor = processor

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.image_paths = []

        for line in lines:
            line = line.replace("\\", "/")

            if line.startswith("./"):
                line = line[2:]

            image_path = self.root / Path(line)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image does not exist: {image_path}"
                )

            self.image_paths.append(image_path)

        print(
            f"Loaded {len(self.image_paths)} images "
            f"from {self.manifest_path.name}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        image_width, image_height = image.size

        # images/train/foo.jpg -> labels/train/foo.txt
        relative_path = image_path.relative_to(self.root)

        parts = list(relative_path.parts)

        if parts[0].lower() != "images":
            raise ValueError(
                f"Expected path under images/, got: {relative_path}"
            )

        parts[0] = "labels"

        label_path = self.root.joinpath(*parts).with_suffix(".txt")

        annotations = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    class_id, cx, cy, bw, bh = map(
                        float, line.split()
                    )

                    class_id = int(class_id)

                    # YOLO normalized cxcywh
                    # ->
                    # absolute COCO xywh

                    box_width = bw * image_width
                    box_height = bh * image_height

                    x_min = (cx - bw / 2) * image_width
                    y_min = (cy - bh / 2) * image_height

                    # Numerical safety
                    x_min = max(0.0, x_min)
                    y_min = max(0.0, y_min)

                    box_width = min(
                        box_width,
                        image_width - x_min
                    )

                    box_height = min(
                        box_height,
                        image_height - y_min
                    )

                    if box_width <= 0 or box_height <= 0:
                        continue

                    annotations.append({
                        "bbox": [
                            x_min,
                            y_min,
                            box_width,
                            box_height,
                        ],
                        "category_id": class_id,
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    })

        target = {
            "image_id": idx,
            "annotations": annotations,
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels

def collate_fn(batch):
    pixel_values = torch.stack(
        [item[0] for item in batch]
    )

    labels = [
        item[1] for item in batch
    ]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }