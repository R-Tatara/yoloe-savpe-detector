from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from clustering import RotatedBox, largest_cluster_xywhr, xywhr_to_corners

DEVICE = 0  # 0 for GPU (CUDA device 0), 'cpu' for CPU

# MODEL_NAME = Path("yoloe-11l-seg.pt")
MODEL_NAME = Path("yoloe-26x-seg.pt")
CONF_THRESHOLD = 0.05  # lower than default since this is a cross-image prompt
IMGSZ = 1280  # higher than the 640 default so individual objects stay separable
IOU_THRESHOLD = 0.5
TRACKER_CONFIG = "botsort.yaml"  # Ultralytics built-in tracker config (ByteTrack-based with appearance matching)
ROTATED_BOX_COLOR = (0, 255, 255)  # BGR yellow, distinct from the box colors
ROTATED_BOX_THICKNESS = 2
KEY_ESC = 27  # no previous step here, so Esc quits like q
KEY_QUIT = ord("q")


def load_yoloe_model() -> YOLOE:
    """Load the YOLOE model used for SAVPE-based detection and tracking."""
    return YOLOE(str(MODEL_NAME))


def initialize_vpe_classes(
    model: YOLOE,
    refer_image: np.ndarray,
    visual_prompts: dict[str, np.ndarray],
) -> list:
    # Make the visual prompt embedding into the model's classes
    return model.predict(
        refer_image,
        refer_image=refer_image,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        device=DEVICE,
        # quantize=16,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        retina_masks=True,  # keep mask resolution matched to the input frame
        verbose=False,
    )


def clear_class_names(model: YOLOE) -> None:
    # Clear class names for simple appearance
    model.predictor.model.names = {i: "" for i in model.names}


def run_detection_loop(
    model: YOLOE,
    pipeline: rs.pipeline,
    get_frame: Callable[[rs.pipeline], np.ndarray | None],
) -> None:
    # Keep pulling RealSense frames and running the VPE-baked model
    window_name = "YOLOE Realtime Detection (press q/Esc to quit)"
    while True:
        frame = get_frame(pipeline)
        if frame is None:
            continue

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            device=DEVICE,
            # quantize=16,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMGSZ,
            retina_masks=True,  # keep mask resolution matched to the input frame
            verbose=False,
        )

        rotated_boxes = extract_rotated_boxes(results)
        report_results(results, rotated_boxes)
        plotted = results[0].plot()
        plotted = draw_rotated_boxes(plotted, rotated_boxes)
        cv2.imshow(window_name, plotted)

        if cv2.waitKey(1) & 0xFF in (KEY_ESC, KEY_QUIT):
            break


def extract_rotated_boxes(results: list) -> list[RotatedBox | None]:
    # Compute an xywhr box per mask from its largest pixel cluster; keeps one entry per detection (None if empty)
    if results[0].masks is None:
        return [None] * len(results[0].boxes)
    return [largest_cluster_xywhr(mask) for mask in results[0].masks.data.cpu().numpy()]


def draw_rotated_boxes(
    image: np.ndarray, rotated_boxes: list[RotatedBox | None]
) -> np.ndarray:
    # Draw rotated bounding box outlines onto the given image, skipping empty entries
    for xywhr in rotated_boxes:
        if xywhr is None:
            continue
        cv2.polylines(
            image,
            [xywhr_to_corners(xywhr)],
            isClosed=True,
            color=ROTATED_BOX_COLOR,
            thickness=ROTATED_BOX_THICKNESS,
        )
    return image


def report_results(results: list, rotated_boxes: list[RotatedBox | None]) -> None:
    # Print tracking ID, confidence and xywhr per detection, excluding ones without a rotated box
    reported = [
        (box, xywhr)
        for box, xywhr in zip(results[0].boxes, rotated_boxes, strict=True)
        if xywhr is not None
    ]
    print(f"Detections: {len(reported)}")
    for box, (center_x, center_y, width, height, angle_rad) in reported:
        track_id = int(box.id.item()) if box.id is not None else None
        print(
            f"id={track_id}, conf={box.conf.item():.2f}, "
            f"xywhr=[{center_x:.2f}, {center_y:.2f}, {width:.2f}, {height:.2f}, {angle_rad:.3f}]"
        )
