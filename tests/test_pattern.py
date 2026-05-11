"""Tests for `dental_rad_cli.pipeline.pattern`.

Numeric fixtures only. We construct binary masks for tooth and bone
geometries and verify the classifier returns the expected pattern.
The 55° boundary is what matters — we build geometries above and below
it and check the sign of the decision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dental_rad_cli.pipeline.pattern import (
    ANGULAR_ANGLE_DEG,
    build_centerline,
    classify_pattern,
)


# ---------------------------------------------------------------------------
# Mask construction helpers
# ---------------------------------------------------------------------------


def _rect_mask(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Filled rectangle in a binary uint8 mask."""
    m = np.zeros(shape, dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def _triangle_mask(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    """Filled triangle/polygon in a binary uint8 mask."""
    import cv2

    m = np.zeros(shape, dtype=np.uint8)
    poly = np.array(pts, dtype=np.int32)
    cv2.fillPoly(m, [poly], color=1)
    return m


# ---------------------------------------------------------------------------
# build_centerline — sanity
# ---------------------------------------------------------------------------


def test_build_centerline_returns_none_for_empty_mask() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    assert build_centerline(m) is None


def test_build_centerline_returns_polyline_for_rect() -> None:
    m = _rect_mask((200, 400), 50, 90, 350, 110)
    line = build_centerline(m)
    assert line is not None
    assert line.shape[1] == 2
    assert len(line) >= 2


# ---------------------------------------------------------------------------
# classify_pattern — insufficient inputs
# ---------------------------------------------------------------------------


def test_classify_pattern_no_tooth_mask() -> None:
    bone = _rect_mask((100, 200), 20, 40, 180, 60)
    tooth = np.zeros((100, 200), dtype=np.uint8)  # empty
    assert classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[(50.0, 30.0)],
        bone_crest_landmarks=[(50.0, 50.0)],
    ) == "unknown"


def test_classify_pattern_no_bone_mask() -> None:
    tooth = _rect_mask((100, 200), 80, 10, 120, 90)
    bone = np.zeros((100, 200), dtype=np.uint8)
    assert classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[(100.0, 30.0)],
        bone_crest_landmarks=[(100.0, 50.0)],
    ) == "unknown"


def test_classify_pattern_no_landmarks() -> None:
    tooth = _rect_mask((100, 200), 80, 10, 120, 90)
    bone = _rect_mask((100, 200), 20, 40, 180, 60)
    assert classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[],
        bone_crest_landmarks=[(100.0, 50.0)],
    ) == "unknown"
    assert classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[(100.0, 30.0)],
        bone_crest_landmarks=[],
    ) == "unknown"


# ---------------------------------------------------------------------------
# classify_pattern — geometric cases
#
# The pattern algorithm operates on bone-centerline endpoints and the
# nearest tooth-mask tangent. To produce a deterministic outcome we
# construct a tooth that sits adjacent to one end of a bone strip:
#
#   - Horizontal case: bone is a horizontal strip; tooth is a vertical
#     rectangle whose left side runs perpendicular to the bone. The
#     angle between the bone tangent (horizontal) and the tooth tangent
#     (vertical) is ~90° → > 55° → horizontal.
#
#   - Angular case: bone is a strip that slopes down sharply at one
#     end (a notch); the bone tangent at the notch end is nearly
#     parallel to the tooth's vertical side → angle < 55° → angular.
# ---------------------------------------------------------------------------


def test_classify_pattern_horizontal() -> None:
    # Image is 200x400 (H x W).
    H, W = 200, 400
    # Bone: thin horizontal strip across the middle.
    bone = _rect_mask((H, W), x0=20, y0=95, x1=380, y1=105)
    # Tooth: vertical rectangle sitting on the right end of the bone.
    tooth = _rect_mask((H, W), x0=370, y0=20, x1=390, y1=180)
    pattern = classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[(380.0, 50.0)],
        bone_crest_landmarks=[(380.0, 100.0)],
    )
    # The bone-tangent (horizontal) vs tooth-tangent (vertical) should
    # be ~90° → horizontal.
    assert pattern == "horizontal"


def test_classify_pattern_angular() -> None:
    # Image is 200x400.
    H, W = 200, 400
    # Bone: a strip that slopes steeply at the right end. Build as a
    # polygon with the right end dipping down to y=180 while the rest
    # stays around y=100. The centerline endpoint on the right will run
    # nearly parallel to a vertical tooth.
    bone = _triangle_mask(
        (H, W),
        pts=[
            (20, 95),
            (350, 95),
            (390, 170),  # steep drop
            (390, 180),
            (350, 105),
            (20, 105),
        ],
    )
    # Tooth: vertical rectangle to the right of the bone end.
    tooth = _rect_mask((H, W), x0=380, y0=20, x1=395, y1=185)
    pattern = classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[(388.0, 50.0)],
        bone_crest_landmarks=[(388.0, 165.0)],
        jaw="maxillary",
    )
    # Steep bone end (nearly vertical) parallel to vertical tooth side
    # → small angle → angular_vertical.
    assert pattern in {"angular_vertical", "horizontal"}
    # The stronger property — at least one endpoint should classify
    # as angular_vertical given the steep geometry. If the test ever
    # flips to "horizontal" the geometry needs tightening; pin the
    # angular expectation as the canonical assertion.
    assert pattern == "angular_vertical"


def test_classify_pattern_boundary_constant_is_55() -> None:
    # Sanity-check the named constant matches the brief — guards against
    # accidental edits to the threshold.
    assert math.isclose(ANGULAR_ANGLE_DEG, 55.0)


def test_classify_pattern_far_from_tooth_returns_unknown() -> None:
    # Bone strip on left of image, tooth on right of image — endpoints
    # are >> SKIP_THRESHOLD from any tooth-mask vertex → unknown.
    H, W = 200, 400
    bone = _rect_mask((H, W), 10, 95, 60, 105)
    tooth = _rect_mask((H, W), 350, 20, 380, 180)
    pattern = classify_pattern(
        tooth_mask=tooth,
        bone_mask=bone,
        cej_landmarks=[(365.0, 50.0)],
        bone_crest_landmarks=[(365.0, 100.0)],
    )
    assert pattern == "unknown"
