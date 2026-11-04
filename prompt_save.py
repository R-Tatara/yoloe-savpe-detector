import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROMPT_DIR = Path("prompt")
PROMPT_JSON_NAME = "prompt.json"
REFER_IMAGE_NAME = "refer.png"
MASK_IMAGE_NAME = "mask.png"
SCHEMA_VERSION = 1
PROMPT_SOURCE = "sam2-click"
DEFAULT_CLASS_NAME = ""
MASK_PIXEL_VALUE = 255  # binary mask is stored as 0/255 for viewability


def validate_bbox(bbox: list[float], width: int, height: int) -> None:
    """Raise ValueError unless bbox is a non-empty xyxy box inside the image."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2>x1 and y2>y1: {bbox}")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"bbox is outside the {width}x{height} image: {bbox}")


def save_visual_prompt(
    frame: np.ndarray,
    visual_prompts: dict[str, np.ndarray],
    mask: np.ndarray,
    directory: Path = PROMPT_DIR,
) -> None:
    """Write the reference image, mask and prompt metadata, overwriting existing files."""
    height, width = frame.shape[:2]
    bbox = [float(value) for value in visual_prompts["bboxes"][0]]
    validate_bbox(bbox, width, height)

    directory.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(directory / REFER_IMAGE_NAME), frame):
        raise OSError(f"failed to write {REFER_IMAGE_NAME}")
    mask_image = (mask.astype(bool) * MASK_PIXEL_VALUE).astype(np.uint8)
    if not cv2.imwrite(str(directory / MASK_IMAGE_NAME), mask_image):
        raise OSError(f"failed to write {MASK_IMAGE_NAME}")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "source": PROMPT_SOURCE,
        "image": {"file": REFER_IMAGE_NAME, "width": width, "height": height},
        "mask_file": MASK_IMAGE_NAME,
        "class_name": DEFAULT_CLASS_NAME,
        "bbox": bbox,
    }
    json_path = directory / PROMPT_JSON_NAME
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
