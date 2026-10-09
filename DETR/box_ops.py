import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Converts bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    Args:
        boxes: Tensor of shape [N, 4] with (cx, cy, w, h)

    Returns:
        Tensor of shape [N, 4] with (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def rescale_bbox(boxes: torch.Tensor, size: tuple) -> torch.Tensor:
    """Rescales normalized bounding boxes to absolute pixel coordinates.

    Args:
        boxes: Tensor of shape [N, 4] in (cx, cy, w, h) normalized coordinates
        size: Tuple of (img_w, img_h)

    Returns:
        Tensor of shape [N, 4] in (x1, y1, x2, y2) pixel coordinates
    """
    img_w, img_h = size
    b = cxcywh_to_xyxy(boxes)
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device, dtype=boxes.dtype)
    return b * scale
