from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

WIDTH = 640
HEIGHT = 480
FPS = 30

MODEL_NAME = Path("yoloe-11s-seg.pt")
CONF_THRESHOLD = 0.05  # lower than default since this is a cross-image prompt
IMGSZ = 1280  # higher than the 640 default so individual objects stay separable
IOU_THRESHOLD = 0.5
DEVICE = 0  # 0 for GPU (CUDA device 0), 'cpu' for CPU
TRACKER_CONFIG = "botsort.yaml"  # Ultralytics built-in tracker config (ByteTrack-based with appearance matching)


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


def select_reference_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    # Let the user drag a box around one in-scene object (Enter to confirm, Esc to cancel)
    x, y, w, h = cv2.selectROI(
        "Select object and press Enter", frame, showCrosshair=False
    )
    cv2.destroyAllWindows()
    return x, y, w, h


def build_visual_prompts(bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    # Convert a single xywh bbox into the bboxes/cls arrays YOLOE expects
    x, y, w, h = bbox
    return dict(
        bboxes=np.array([[x, y, x + w, y + h]], dtype=np.float32),
        cls=np.array([0]),
    )


def initialize_vpe_classes(
    model: YOLOE,
    frame: np.ndarray,
    visual_prompts: dict[str, np.ndarray],
) -> list:
    # Make the visual prompt embedding into the model's classes
    return model.predict(
        frame,
        refer_image=frame,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        device=DEVICE,
        quantize=16,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        verbose=False,
    )


def run_detection_loop(model: YOLOE, pipeline: rs.pipeline) -> None:
    # Keep pulling RealSense frames and running the VPE-baked model
    window_name = "YOLOE Realtime Detection (press q to quit)"
    while True:
        frame = get_frame(pipeline)
        if frame is None:
            continue

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            device=DEVICE,
            quantize=16,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMGSZ,
            verbose=False,
        )
        report_results(results)
        cv2.imshow(window_name, results[0].plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def report_results(results: list) -> None:
    # Print detection details including tracking ID
    print(f"Detections: {len(results[0].boxes)}")
    for box in results[0].boxes:
        track_id = int(box.id.item()) if box.id is not None else None
        xyxy = [round(v, 2) for v in box.xyxy.tolist()[0]]
        print(f"id={track_id}, conf={box.conf.item():.2f}, xyxy={xyxy}")


def main() -> None:
    # Set up the RealSense camera
    try:
        pipeline = setup_realsense()
    except Exception as e:
        print(f"Error: RealSense not found. {e}")
        return

    try:
        # Capture single RealSense frame and let the user select the object
        frame = get_frame(pipeline)
        bbox = select_reference_bbox(frame)
        visual_prompts = build_visual_prompts(bbox)

        # Initialize the YOLOE model and prepare the visual prompt
        model = YOLOE(str(MODEL_NAME))
        results = initialize_vpe_classes(model, frame, visual_prompts)
        model.predictor.model.names = {i: "" for i in model.names}  # Clear class names for simple appearance

        run_detection_loop(model, pipeline)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
