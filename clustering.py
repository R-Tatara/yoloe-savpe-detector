import math

import cv2
import numpy as np

# (cx, cy, w, h, angle in rad), matching the ultralytics xywhr convention
RotatedBox = tuple[float, float, float, float, float]

# max pixel gap treated as same cluster; kept small to separate touching objects
CLUSTER_DISTANCE_PX = 2
OPENING_KERNEL_SIZE = 3  # removes thin noise bridges before clustering; 0 disables
# clusters with fewer mask pixels than this are discarded as noise
MIN_CLUSTER_AREA_PX = 50
CONNECTIVITY = 8


def _to_binary(mask: np.ndarray) -> np.ndarray:
    """Normalize an input mask (0/1 or 0/255) to a uint8 0/1 binary mask."""
    return (mask > 0).astype(np.uint8)


def denoise_mask(mask: np.ndarray) -> np.ndarray:
    """Remove thin noise connections from a binary mask via morphological opening."""
    binary = _to_binary(mask)
    if OPENING_KERNEL_SIZE <= 0:
        return binary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def label_clusters(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label mask pixels within CLUSTER_DISTANCE_PX of each other as the same cluster."""
    binary = _to_binary(mask)
    # dilating both sides of a gap by radius merges points at most CLUSTER_DISTANCE_PX apart
    radius = max(1, round(CLUSTER_DISTANCE_PX / 2))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(binary, kernel)
    num_labels, labels = cv2.connectedComponentsWithStats(
        dilated, connectivity=CONNECTIVITY
    )[:2]
    # map labels back onto the original (non-dilated) mask pixels
    labels = labels * binary
    return labels, num_labels


def extract_largest_cluster(mask: np.ndarray) -> np.ndarray | None:
    """Return a binary mask containing only the largest spatial cluster, or None if too small."""
    denoised = denoise_mask(mask)
    if not denoised.any():
        return None

    labels, num_labels = label_clusters(denoised)
    if num_labels <= 1:
        return None

    counts = np.bincount(labels.ravel(), minlength=num_labels)
    counts[0] = 0  # ignore background label
    largest_label = int(counts.argmax())
    if counts[largest_label] < MIN_CLUSTER_AREA_PX:
        return None
    return (labels == largest_label).astype(np.uint8)


def largest_cluster_xywhr(mask: np.ndarray) -> RotatedBox | None:
    """Compute the minimum-area rotated box (cx, cy, w, h, angle in rad) of the largest mask cluster."""
    largest_cluster = extract_largest_cluster(mask)
    if largest_cluster is None:
        return None
    points = cv2.findNonZero(largest_cluster)
    (center_x, center_y), (width, height), angle_deg = cv2.minAreaRect(points)
    return (center_x, center_y, width, height, math.radians(angle_deg))


def xywhr_to_corners(xywhr: RotatedBox) -> np.ndarray:
    """Convert (cx, cy, w, h, angle in rad) to 4 integer corner points for drawing."""
    center_x, center_y, width, height, angle_rad = xywhr
    rect = ((center_x, center_y), (width, height), math.degrees(angle_rad))
    return cv2.boxPoints(rect).astype(int)
