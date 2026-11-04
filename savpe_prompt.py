import cv2
import numpy as np


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the smallest axis-aligned xyxy box enclosing the binary mask."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return x, y, x + w, y + h


def build_visual_prompts(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Convert the mask's enclosing bbox into the bboxes/cls arrays YOLOE expects."""
    return dict(
        bboxes=np.array([mask_to_bbox(mask)], dtype=np.float32),
        cls=np.array([0]),
    )
