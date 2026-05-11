"""Bone-loss percentage and AAP severity tier (rule-layer math).

Pure functions. No file I/O. No global state.

Formula (clinical / paper-internal, methodology brief §3.3)::

    L_total = ||apex - CEJ||           # CEJ-to-apex distance (root length)
    L_loss  = ||bone_crest - CEJ||     # CEJ-to-bone-crest (lost portion)
    pct     = 100 * L_loss / L_total

AAP stage thresholds (methodology brief §3.3)::

    <15%   → Stage I  (mild)
    15-33% → Stage II (moderate)
    >33%   → Stage III (severe)

Anatomically-impossible case (bone crest "above" CEJ in the direction of
the apex — i.e. the projection of (bone_crest - CEJ) onto (apex - CEJ) is
negative) is rejected: `compute_bone_loss_pct` returns None and the
caller's BoneLossSite records ``reason="bone_crest_above_cej"``. We
chose return-None-with-reason over silent clamp-to-zero because (a) it
is more often a keypoint-detection error than a real measurement, and
(b) a downstream zero would be indistinguishable from a healthy tooth.
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

# Tolerance for "anatomical direction check". Projection of bone-crest
# vector onto apex vector must be > -tol; small negative numerical noise
# is tolerated.
_DIRECTION_ABS_TOL: float = 1e-6


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
    """Return bone-loss percentage, or None if it cannot be computed.

    Returns None when:
    - Any keypoint is None.
    - CEJ and apex coincide (denominator ≤ tolerance — degenerate root).
    - Bone crest projects on the opposite side of CEJ from apex
      (anatomically impossible / keypoint error).
    """
    if cej is None or bone_crest is None or apex is None:
        return None

    l_total = _dist(apex, cej)
    if l_total <= _ZERO_DIST_ABS_TOL:
        return None

    # Direction check: project (bone_crest - CEJ) onto (apex - CEJ).
    # If projection is meaningfully negative, the bone crest is on the
    # crown side of the CEJ — anatomically impossible.
    apex_vec = (apex[0] - cej[0], apex[1] - cej[1])
    crest_vec = (bone_crest[0] - cej[0], bone_crest[1] - cej[1])
    projection = _dot(crest_vec, apex_vec)
    if projection < -_DIRECTION_ABS_TOL:
        return None

    l_loss = _dist(bone_crest, cej)
    return 100.0 * l_loss / l_total


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
