"""Bone-loss percentage and AAP severity tier (rule-layer math).

Pure functions. No file I/O. No global state.

Formula (clinical / paper-internal, methodology brief §3.3)::

    axis    = apex - CEJ                 # tooth long-axis direction
    L_total = ||axis||                   # root length
    L_loss  = (bone_crest - CEJ) · axis_unit  # signed projection onto axis
    pct     = 100 * L_loss / L_total

Bone loss is conceptually a measurement *along the tooth's long axis*
(CEJ → apex direction). The earlier 2D-Euclidean formula
(``||bone_crest - CEJ|| / ||apex - CEJ||``) was geometrically wrong
when bone-crest and CEJ were laterally offset (e.g. a bone-crest
keypoint interpolated at the bbox edge while CEJ landed at the tooth
center): the numerator inflated by the lateral component and produced
clinically impossible values >100%. Projecting bone-crest onto the
CEJ→apex axis is the correct "fraction of root length below the CEJ"
semantic the AAP staging downstream expects.

AAP stage thresholds (methodology brief §3.3)::

    <15%   → Stage I  (mild)
    15-33% → Stage II (moderate)
    >33%   → Stage III (severe)

Anatomically-impossible cases are rejected:

- Bone crest projects *coronal* to CEJ (negative offset along the
  CEJ→apex axis beyond tolerance) → return None. Caller records
  ``reason="bone_crest_above_cej"``. Return-None-with-reason chosen
  over silent clamp-to-zero because (a) it is more often a
  keypoint-detection error than a real measurement, and (b) a
  downstream zero would be indistinguishable from a healthy tooth.

- Bone crest projects *beyond* the apex (offset > root length →
  pct > 100%). Anatomically this would mean bone is gone past the
  root tip; in practice it is always a keypoint error. We clamp at
  100 rather than returning None because the tooth IS in severe
  bone loss territory and the staging downstream still wants to see
  "severe", not "incomputable".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from dental_rad_cli.schema import Point, SeverityTier

# AAP stage thresholds — percent values (inclusive lower bound on
# moderate, exclusive upper bound on moderate).
_MILD_MAX_EXCLUSIVE: float = 15.0
_MODERATE_MAX_INCLUSIVE: float = 33.0

# Tolerance for "denominator effectively zero" (apex and CEJ coincide).
# 1e-9 pixels is well below any real image resolution.
_ZERO_DIST_ABS_TOL: float = 1e-9

# Tolerance for "bone-crest projects coronal to CEJ". Small negative
# offsets along the axis (in pixels) are tolerated as keypoint noise;
# only larger negative offsets are rejected as anatomically impossible.
# 2 px is empirically the noise floor for the keypoint R-CNN heads.
_NEGATIVE_PROJECTION_PX_TOL: float = 2.0


def _dist(a: Point, b: Point) -> float:
    """2-D Euclidean distance between two pixel coordinates."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def compute_bone_loss_pct(
    cej: Optional[Point],
    bone_crest: Optional[Point],
    apex: Optional[Point],
) -> Optional[float]:
    """Return bone-loss percentage (projected onto the tooth long axis).

    The percentage is the signed projection of ``(bone_crest - CEJ)``
    onto the unit CEJ→apex axis, divided by the axis length. This
    answers "what fraction of the root length is between the CEJ and
    the bone crest, along the tooth's long axis?" — the semantic the
    AAP staging downstream expects.

    Returns None when:

    - Any keypoint is None.
    - CEJ and apex coincide (denominator ≤ tolerance — degenerate
      root direction).
    - Bone crest projects coronally past CEJ (signed offset along
      axis < -tolerance — anatomically impossible / keypoint error).

    Clamps to [0, 100] when the projection is within tolerance of
    zero (returns 0) or exceeds the root length (returns 100). Both
    clamps are documented in the module docstring.
    """
    if cej is None or bone_crest is None or apex is None:
        return None

    # Tooth long axis = CEJ → apex.
    axis_x = apex[0] - cej[0]
    axis_y = apex[1] - cej[1]
    axis_length = math.hypot(axis_x, axis_y)
    if axis_length <= _ZERO_DIST_ABS_TOL:
        return None

    # Unit axis vector.
    axis_unit = (axis_x / axis_length, axis_y / axis_length)

    # Signed scalar projection of (bone_crest - CEJ) onto the axis.
    crest_vec = (bone_crest[0] - cej[0], bone_crest[1] - cej[1])
    bone_offset = _dot(crest_vec, axis_unit)

    # Direction check: meaningfully negative offset means bone crest is
    # on the crown side of CEJ along the tooth axis — anatomically
    # impossible.
    if bone_offset < -_NEGATIVE_PROJECTION_PX_TOL:
        return None

    pct = 100.0 * bone_offset / axis_length

    # Clamp at the physical bounds. Negative noise within tolerance →
    # 0; over-projection past the apex → 100 (still "severe" downstream).
    if pct < 0.0:
        pct = 0.0
    elif pct > 100.0:
        pct = 100.0

    return pct


def severity_tier(pct: Optional[float]) -> Optional[SeverityTier]:
    """Map a bone-loss percentage to AAP mild / moderate / severe.

    None in → None out (no measurement available).
    """
    if pct is None:
        return None
    if pct < _MILD_MAX_EXCLUSIVE:
        return "mild"
    if pct <= _MODERATE_MAX_INCLUSIVE:
        return "moderate"
    return "severe"
