import cv2
import numpy as np
import pyrealsense2 as rs

from prompt_load import load_visual_prompt
from prompt_save import save_visual_prompt
from savpe import (
    clear_class_names,
    initialize_vpe_classes,
    load_yoloe_model,
    run_detection_loop,
)
from savpe_prompt import build_visual_prompts
from segmentation import load_sam_model, select_reference_mask

WIDTH = 640
HEIGHT = 480
FPS = 30


def setup_realsense() -> rs.pipeline:
    # Start the RealSense color stream and return the running pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipeline.start(config)
    return pipeline


def get_frame(pipeline: rs.pipeline) -> np.ndarray | None:
    # Block until the next color frame arrives and return it as a numpy array
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())


def create_visual_prompt(
    pipeline: rs.pipeline,
) -> bool:
    """Let the user pick the object with SAM2 and save the prompt to disk."""
    frame = get_frame(pipeline)
    if frame is None:
        print("Error: could not capture the initial color frame.")
        return False

    # Load SAM2 and let the user click the object to segment
    sam_model = load_sam_model()
    mask = select_reference_mask(sam_model, frame)
    if mask is None:
        print("Canceled: no object was selected.")
        return False

    save_visual_prompt(frame, build_visual_prompts(mask), mask)
    return True


def capture_and_save_visual_prompt() -> bool:
    """Capture a frame, let the user pick the object, and save the visual prompt."""
    pipeline = setup_realsense()

    try:
        return create_visual_prompt(pipeline)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def detect_with_saved_visual_prompt() -> None:
    """Load the saved visual prompt into YOLOE and run real-time detection."""
    pipeline = setup_realsense()

    try:
        # Load YOLOE and the saved prompt, then bake the prompt into the model's classes
        model = load_yoloe_model()
        refer_image, visual_prompts = load_visual_prompt()
        initialize_vpe_classes(model, refer_image, visual_prompts)

        clear_class_names(model)

        run_detection_loop(model, pipeline, get_frame)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    """Create and save a visual prompt, then run detection with it."""
    if not capture_and_save_visual_prompt():
        return

    detect_with_saved_visual_prompt()


if __name__ == "__main__":
    main()
