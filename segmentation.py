from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.engine.results import Results

DEVICE = 0  # 0 for GPU (CUDA device 0), 'cpu' for CPU

SAM_MODEL_NAME = Path("sam2.1_b.pt")  # auto-downloaded by Ultralytics if missing
SAM_FOREGROUND_LABEL = 1  # Ultralytics SAM point label for "include this object"
MASK_BINARY_THRESHOLD = 0.5  # cutoff applied to the SAM mask logits/probabilities
POINT_WINDOW_NAME = "Click the object, then press Enter (q/Esc to quit)"
MASK_WINDOW_NAME = "Check the mask: Enter to continue, Esc to re-select, q to quit"
POINT_MARKER_COLOR = (0, 0, 255)  # BGR red marker for the clicked point
POINT_MARKER_RADIUS = 5
MASK_OVERLAY_COLOR = (0, 255, 0)  # BGR green tint over the SAM mask area
MASK_OVERLAY_ALPHA = 0.5
KEY_WAIT_MS = 20
KEY_ESC = 27  # go back one step (quits when there is no previous step)
KEY_QUIT = ord("q")  # quit the whole program from any window
KEY_ENTER = (13, 10)  # Enter is 13 on most builds and 10 on some Linux builds


def select_reference_point(frame: np.ndarray) -> tuple[int, int] | None:
    """Let the user click one object point (Enter to confirm, q/Esc to quit)."""
    height, width = frame.shape[:2]
    clicked_point: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)

    cv2.namedWindow(POINT_WINDOW_NAME)
    cv2.setMouseCallback(POINT_WINDOW_NAME, on_mouse)
    try:
        while True:
            preview = frame.copy()
            if clicked_point is not None:
                cv2.circle(
                    preview,
                    clicked_point,
                    POINT_MARKER_RADIUS,
                    POINT_MARKER_COLOR,
                    thickness=-1,
                )
            cv2.imshow(POINT_WINDOW_NAME, preview)
            key = cv2.waitKey(KEY_WAIT_MS) & 0xFF

            if key in (KEY_ESC, KEY_QUIT):
                return None
            if cv2.getWindowProperty(POINT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in KEY_ENTER:
                if clicked_point is None:
                    print("Error: click the object before pressing Enter.")
                    continue
                x, y = clicked_point
                if not (0 <= x < width and 0 <= y < height):
                    print("Error: the clicked point is outside the frame.")
                    clicked_point = None
                    continue
                return clicked_point
    finally:
        cv2.destroyWindow(POINT_WINDOW_NAME)


def load_sam_model() -> SAM:
    """Load the SAM2 model used to turn the clicked point into an object mask."""
    return SAM(str(SAM_MODEL_NAME))


def generate_object_mask(
    sam_model: SAM,
    frame: np.ndarray,
    point: tuple[int, int],
) -> np.ndarray | None:
    """Run SAM2 with the clicked foreground point and return a binary HxW mask."""
    results = sam_model.predict(
        frame,
        points=[list(point)],
        labels=[SAM_FOREGROUND_LABEL],
        device=DEVICE,
        verbose=False,
    )
    masks = results[0].masks
    if masks is None or len(masks.data) == 0:
        return None

    best_index = select_best_mask_index(results[0])
    mask = masks.data[best_index].cpu().numpy()
    binary_mask = (mask > MASK_BINARY_THRESHOLD).astype(np.uint8)

    height, width = frame.shape[:2]
    if binary_mask.shape != (height, width):
        binary_mask = cv2.resize(
            binary_mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
    if not binary_mask.any():
        return None
    return binary_mask


def select_best_mask_index(result: Results) -> int:
    """Pick the highest-scoring SAM2 candidate, falling back to the first one."""
    boxes = result.boxes
    mask_count = len(result.masks.data)
    if boxes is None or boxes.conf is None or len(boxes.conf) != mask_count:
        return 0
    return int(boxes.conf.argmax())


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a translucent color over the masked area for visual inspection."""
    tinted = frame.copy()
    tinted[mask.astype(bool)] = MASK_OVERLAY_COLOR
    return cv2.addWeighted(
        tinted, MASK_OVERLAY_ALPHA, frame, 1.0 - MASK_OVERLAY_ALPHA, 0.0
    )


def confirm_mask(frame: np.ndarray, mask: np.ndarray) -> bool | None:
    """Show the mask overlay: True on Enter, False on Esc (re-select), None on q/window close."""
    preview = overlay_mask(frame, mask)
    cv2.namedWindow(MASK_WINDOW_NAME)
    try:
        while True:
            cv2.imshow(MASK_WINDOW_NAME, preview)
            key = cv2.waitKey(KEY_WAIT_MS) & 0xFF

            if key == KEY_ESC:
                return False
            if key == KEY_QUIT:
                return None
            if cv2.getWindowProperty(MASK_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in KEY_ENTER:
                return True
    finally:
        cv2.destroyWindow(MASK_WINDOW_NAME)


def select_reference_mask(sam_model: SAM, frame: np.ndarray) -> np.ndarray | None:
    """Repeat click -> SAM2 -> mask check until confirmed, or None if canceled."""
    while True:
        point = select_reference_point(frame)
        if point is None:
            return None

        mask = generate_object_mask(sam_model, frame, point)
        if mask is None:
            print("Error: SAM2 returned no mask for that point. Click again.")
            continue

        confirmation = confirm_mask(frame, mask)
        if confirmation is None:
            return None
        if confirmation:
            return mask
