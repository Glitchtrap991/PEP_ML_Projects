from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class IEDXRayDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.transform = transform
        self.image_paths = []

        with open(self.manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                line = line.replace("\\", "/")
                if line.startswith("./"):
                    line = line[2:]

                image_path = self.root / Path(line)
                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image listed in manifest does not exist:\n"
                        f"  Manifest entry: {line}\n"
                        f"  Resolved path: {image_path}"
                    )

                self.image_paths.append(image_path)

        print(
            f"Loaded {len(self.image_paths)} images "
            f"from {self.manifest_path}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Example:
        # IEDXray/images/train/Train000001.jpg
        #       ↓
        # labels/train/Train000001.txt
        relative_image_path = image_path.relative_to(self.root)
        parts = list(relative_image_path.parts)

        if parts[0].lower() != "images":
            raise ValueError(
                f"Expected path beginning with images/, got: {relative_image_path}"
            )

        parts[0] = "labels"
        label_path = self.root.joinpath(*parts).with_suffix(".txt")

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file missing:\n"
                f"  Image: {image_path}\n"
                f"  Expected label: {label_path}"
            )

        labels = []
        boxes = []

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                if len(values) != 5:
                    raise ValueError(
                        f"Invalid YOLO annotation in {label_path}:\n{line}"
                    )

                class_id = int(values[0])
                cx, cy, w, h = map(float, values[1:])

                labels.append(class_id)
                boxes.append([cx, cy, w, h])

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target


transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)
