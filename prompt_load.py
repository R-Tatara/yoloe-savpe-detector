import json
from pathlib import Path

import cv2
import numpy as np

PROMPT_DIR = Path("prompt")
PROMPT_JSON_NAME = "prompt.json"
SCHEMA_VERSION = 1
DEFAULT_CLASS_ID = 0


def validate_bbox(bbox: list[float], width: int, height: int) -> None:
    """Raise ValueError unless bbox is a non-empty xyxy box inside the image."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2>x1 and y2>y1: {bbox}")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"bbox is outside the {width}x{height} image: {bbox}")


def load_visual_prompt(
    directory: Path = PROMPT_DIR,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read and validate a saved prompt, returning (refer_image, visual_prompts)."""
    json_path = directory / PROMPT_JSON_NAME
    if not json_path.is_file():
        raise FileNotFoundError(f"prompt metadata not found: {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))

    version = metadata.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version}")

    image_path = directory / metadata["image"]["file"]
    if not image_path.is_file():
        raise FileNotFoundError(f"reference image not found: {image_path}")
    refer_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if refer_image is None:
        raise ValueError(f"failed to decode reference image: {image_path}")

    height, width = refer_image.shape[:2]
    if (width, height) != (metadata["image"]["width"], metadata["image"]["height"]):
        raise ValueError("reference image size does not match prompt.json")

    bbox = [float(value) for value in metadata["bbox"]]
    validate_bbox(bbox, width, height)
    visual_prompts = dict(
        bboxes=np.array([bbox], dtype=np.float32),
        cls=np.array([DEFAULT_CLASS_ID]),
    )
    return refer_image, visual_prompts
