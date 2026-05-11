"""CLAHE preprocessing — mandatory at train AND val AND inference time.

Per the methodology brief: CLAHE with `clip_limit=40.0, tile_grid_size=(8,8)`
is the only image-enhancement step in the keypoint-detection pipeline. It is
load-bearing for dim panoramic / periapical radiographs; removing it
materially degrades downstream accuracy. CLAHE is applied deterministically
(p=1.0) at every stage — it is NOT a randomized augmentation.

The same `apply_clahe` function is used by:

- the keypoint R-CNN dataset transform (training + validation)
- the inference path (`pipeline/infer.py`, written by another subagent)

YOLO detection / segmentation models in this pipeline consume raw RGB
without CLAHE per the upstream methodology; if a later experiment wants
CLAHE on the YOLO side, route through this same function for consistency.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

# Constants from the methodology brief (§1.2 Keypoint Detection — Augmentations).
# Aggressive clip_limit (40.0 vs OpenCV default 4.0) is intentional for dim
# DenPAR contrast.
CLAHE_CLIP_LIMIT: Final[float] = 40.0
CLAHE_TILE_GRID: Final[tuple[int, int]] = (8, 8)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to a radiograph.

    Accepts grayscale (H, W) or 3-channel RGB (H, W, 3) uint8 images.
    Returns the same shape and dtype.

    For 3-channel input the image is converted to LAB; CLAHE is applied to
    the L channel; the image is converted back to RGB. This matches the
    common Albumentations CLAHE behavior used in the upstream methodology.

    Args:
        image: Input radiograph as numpy array. Either 2D grayscale or
            3D RGB uint8. Float inputs are coerced to uint8 in [0, 255].

    Returns:
        CLAHE-enhanced image with the same shape and dtype as input.
    """
    if image.dtype != np.uint8:
        # Coerce to uint8 in [0, 255] without clipping surprises.
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    if img.ndim == 2:
        return clahe.apply(img)

    if img.ndim == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    raise ValueError(
        f"apply_clahe expects (H,W) or (H,W,3); got shape {image.shape}"
    )
