"""Horizontal vs angular/vertical bone-loss pattern classification.

Methodology brief §3.2. Pure functions, no I/O, no global state.

Algorithm sketch (rewritten from the brief — no upstream names):

1. Build a bone *centerline* polyline from the bone mask polygon. The
   polygon's vertices form a closed loop; walk both halves between the
   leftmost and rightmost vertices and average paired walks. The output
   is an ordered list of points running roughly along the bone ridge.
2. For each endpoint of the centerline:
   a. Find the nearest tooth-mask vertex.
   b. Walk a fixed number of indices around the tooth polygon (the
      sign of the walk is chosen so the second vertex moves toward the
      apex side of the tooth; with a single jaw classification per image
      the sign is parameterized by jaw direction).
   c. The two tooth-mask vertices define a tooth-tangent vector
      (extended by a constant factor — used only for sign/scale).
   d. Compute the bone-tangent vector from the centerline endpoint to a
      point a fraction of the way inward.
   e. Take the angle between the two tangent vectors via the dot
      product.
3. If any endpoint produces ``angle ≤ 55°`` → angular/vertical. If all
   endpoints exceed 55° → horizontal. Endpoints far from any tooth-mask
   vertex (further than SKIP_THRESHOLD) are skipped.

Constants (§3.2): LINE_EXTENSION=20, MASK_SEARCH_DISTANCE=20,
SKIP_THRESHOLD=20, ANGULAR_ANGLE=55°, POINTS_AWAY=len(centerline)//4.

The bone mask is provided as a binary 2-D `np.ndarray` (1=bone, 0=bg);
the function extracts the largest external contour. The tooth mask is
also a binary 2-D array. We do not depend on cv2's specific contour
return shape — a small pure-NumPy contour-trace would also work — but
since opencv-python is already a project dependency, this module uses
``cv2.findContours``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from dental_rad_cli.schema import BoneLossPattern, Jaw

# Constants from methodology brief §3.2.
ANGULAR_ANGLE_DEG: float = 55.0
LINE_EXTENSION: float = 20.0
MASK_SEARCH_DISTANCE: int = 20
SKIP_THRESHOLD_PX: float = 20.0
POINTS_AWAY_DIVISOR: int = 4  # bone[len // 4]


def _contour_largest(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour of a binary mask as an
    ``(N, 2)`` float array of (x, y) pixel coordinates. Returns None if
    no contour found.

    Pure NumPy fallback would re-implement Moore-neighbor tracing; we use
    cv2 here because opencv-python is already a project dep. Import is
    deferred so unit tests for trivial paths don't pay the import cost.
    """
    import cv2  # local import to keep top-level light

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    # cv2 returns (N, 1, 2) int32; reshape to (N, 2) float.
    return largest.reshape(-1, 2).astype(np.float64)


def build_centerline(bone_mask: np.ndarray) -> Optional[np.ndarray]:
    """Reduce a bone polygon to a polyline running roughly along its
    ridge.

    Approach (methodology brief §1.4): treat the polygon vertices as a
    circular doubly-linked list; locate the leftmost and rightmost
    vertices; walk both halves in opposite directions and average paired
    points. The output polyline has length min(half_a, half_b).
    """
    contour = _contour_largest(bone_mask)
    if contour is None or len(contour) < 4:
        return None

    n = len(contour)
    xs = contour[:, 0]
    left_idx = int(np.argmin(xs))
    right_idx = int(np.argmax(xs))
    if left_idx == right_idx:
        return None

    # Two halves of the loop between left and right ends.
    # Half A: walk forward from left_idx to right_idx.
    # Half B: walk forward from right_idx to left_idx (i.e. the other
    # side of the loop).
    half_a: List[np.ndarray] = []
    i = left_idx
    while i != right_idx:
        half_a.append(contour[i])
        i = (i + 1) % n
    half_a.append(contour[right_idx])

    half_b: List[np.ndarray] = []
    j = left_idx
    while j != right_idx:
        half_b.append(contour[j])
        j = (j - 1) % n
    half_b.append(contour[right_idx])

    m = min(len(half_a), len(half_b))
    if m < 2:
        return None
    # Resample to length m by index lookup (uniform — cheaper than
    # arclength resample, faithful to the brief's "paired walk").
    a_idx = np.linspace(0, len(half_a) - 1, m).astype(int)
    b_idx = np.linspace(0, len(half_b) - 1, m).astype(int)
    averaged = np.stack(
        [(half_a[ai] + half_b[bi]) / 2.0 for ai, bi in zip(a_idx, b_idx)]
    )
    return averaged


def _angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> Optional[float]:
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos_t = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    # numerical clamp
    cos_t = max(-1.0, min(1.0, cos_t))
    # We care about the *acute* angle between two undirected line
    # segments — the tooth tangent and bone tangent have no inherent
    # direction. Fold to [0, 90].
    theta = math.degrees(math.acos(cos_t))
    if theta > 90.0:
        theta = 180.0 - theta
    return theta


def _tooth_tangent_at(
    tooth_contour: np.ndarray,
    centerline_end: np.ndarray,
    jaw: Optional[Jaw],
) -> Optional[Tuple[float, float]]:
    """Return the (extended) tooth-tangent vector near the centerline
    endpoint, or None if the endpoint is too far from any tooth vertex.
    """
    n = len(tooth_contour)
    if n < 2:
        return None
    diffs = tooth_contour - centerline_end[None, :]
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    nearest = int(np.argmin(dists))
    if dists[nearest] >= SKIP_THRESHOLD_PX:
        return None

    delta = MASK_SEARCH_DISTANCE
    q = tooth_contour[nearest]
    q2_fwd = tooth_contour[(nearest + delta) % n]
    q2_bwd = tooth_contour[(nearest - delta) % n]

    # Sign rule from §3.2: pick the q2 that moves *toward* the apex side
    # of the tooth. For mandibular IOPAs the apex is at smaller y
    # (higher in image — mandibular roots point up in the image plane);
    # for maxillary IOPAs the apex is at larger y. We pick the q2 that
    # disagrees with the bone endpoint's y-direction.
    if jaw == "mandibular":
        q2 = q2_bwd if q2_fwd[1] < centerline_end[1] else q2_fwd
    elif jaw == "maxillary":
        q2 = q2_bwd if q2_fwd[1] > centerline_end[1] else q2_fwd
    else:
        q2 = q2_fwd  # silent fallback when jaw unknown

    dx = q[0] - q2[0]
    dy = q[1] - q2[1]
    # Extend symmetrically — only direction matters for the angle calc,
    # but we keep the extension to match the methodology shape.
    return (2.0 * LINE_EXTENSION * dx, 2.0 * LINE_EXTENSION * dy)


def _endpoint_pattern(
    centerline: np.ndarray,
    end_idx: int,
    tooth_contours: List[np.ndarray],
    jaw: Optional[Jaw],
) -> Optional[str]:
    """Return ``"angular_vertical"``, ``"horizontal"``, or None for one
    centerline endpoint.
    """
    p = centerline[end_idx]

    # Nearest tooth-mask vertex across ALL tooth contours.
    best_tooth: Optional[np.ndarray] = None
    best_dist = math.inf
    for tc in tooth_contours:
        if len(tc) == 0:
            continue
        diffs = tc - p[None, :]
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        idx = int(np.argmin(dists))
        if dists[idx] < best_dist:
            best_dist = float(dists[idx])
            best_tooth = tc
    if best_tooth is None or best_dist >= SKIP_THRESHOLD_PX:
        return None

    mask_vec = _tooth_tangent_at(best_tooth, p, jaw)
    if mask_vec is None:
        return None

    k = max(1, len(centerline) // POINTS_AWAY_DIVISOR)
    if end_idx == 0:
        bp = centerline[k]
    else:
        bp = centerline[-1 - k]
    bone_vec = (float(bp[0] - p[0]), float(bp[1] - p[1]))

    theta = _angle_deg(bone_vec, mask_vec)
    if theta is None:
        return None
    return "angular_vertical" if theta <= ANGULAR_ANGLE_DEG else "horizontal"


def classify_pattern(
    tooth_mask: np.ndarray,
    bone_mask: np.ndarray,
    cej_landmarks: List[Tuple[float, float]],
    bone_crest_landmarks: List[Tuple[float, float]],
    jaw: Optional[Jaw] = None,
) -> BoneLossPattern:
    """Classify bone-loss pattern as horizontal vs angular/vertical.

    ``cej_landmarks`` and ``bone_crest_landmarks`` are not required by
    the angle math itself (the algorithm operates on bone-centerline
    endpoints and tooth-mask tangents) but their presence is the
    gate-condition: if no CEJ or bone-crest landmarks are available, the
    rule layer has no business calling the classifier — returns
    ``"unknown"``.

    Returns ``"unknown"`` when inputs are insufficient, when the bone
    polygon cannot be reduced to a centerline, or when no endpoint
    produced a usable angle.
    """
    if (
        tooth_mask is None
        or bone_mask is None
        or tooth_mask.size == 0
        or bone_mask.size == 0
        or not cej_landmarks
        or not bone_crest_landmarks
    ):
        return "unknown"

    centerline = build_centerline(bone_mask)
    if centerline is None or len(centerline) < 2:
        return "unknown"

    tooth_contour = _contour_largest(tooth_mask)
    if tooth_contour is None or len(tooth_contour) < 2:
        return "unknown"

    tooth_contours = [tooth_contour]

    saw_horizontal = False
    saw_any = False
    for end_idx in (0, len(centerline) - 1):
        result = _endpoint_pattern(centerline, end_idx, tooth_contours, jaw)
        if result is None:
            continue
        saw_any = True
        if result == "angular_vertical":
            # Any angular endpoint wins (consistent with §3.2 rule:
            # angular is the "named defect" — its presence dominates).
            return "angular_vertical"
        if result == "horizontal":
            saw_horizontal = True

    if not saw_any:
        return "unknown"
    if saw_horizontal:
        return "horizontal"
    return "unknown"
