"""Tests for `dental_rad_cli.pipeline.severity`.

The bone-loss math projects the bone-crest vector onto the tooth's
long axis (CEJ → apex) — see ``severity.py`` module docstring. Tests
below pin both the geometric semantics and the AAP tier boundaries
(<15% mild, 15-33% moderate, >33% severe).
"""

from __future__ import annotations

import math

import pytest

from dental_rad_cli.pipeline.severity import (
    compute_bone_loss_pct,
    severity_tier,
)


# ---------------------------------------------------------------------------
# compute_bone_loss_pct — happy path
# ---------------------------------------------------------------------------


def test_compute_bone_loss_pct_simple_vertical() -> None:
    # CEJ at (0,0), bone crest at (0,10), apex at (0,100).
    # Axis = (0,100). Projection of (0,10) onto axis = 10. pct = 10%.
    pct = compute_bone_loss_pct((0.0, 0.0), (0.0, 10.0), (0.0, 100.0))
    assert pct is not None
    assert math.isclose(pct, 10.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_diagonal_on_axis() -> None:
    # CEJ=(0,0), apex=(10,10), bone_crest=(5,5) lies exactly on the
    # axis at its midpoint. Projection = ||(5,5)|| along axis direction,
    # ratio is 0.5 → 50%.
    pct = compute_bone_loss_pct((0.0, 0.0), (5.0, 5.0), (10.0, 10.0))
    assert pct is not None
    assert math.isclose(pct, 50.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_axis_30_degrees_off_vertical() -> None:
    # Tooth with axis 30 degrees off the vertical. Root length 100.
    # Apex offset by (50, 50*sqrt(3)) from CEJ.
    # Bone crest at 25% root length along the axis = (12.5, 12.5*sqrt(3)).
    # Projection should produce 25% bone loss.
    s3 = math.sqrt(3.0)
    cej = (0.0, 0.0)
    apex = (50.0, 50.0 * s3)
    bone_crest = (12.5, 12.5 * s3)
    pct = compute_bone_loss_pct(cej, bone_crest, apex)
    assert pct is not None
    assert math.isclose(pct, 25.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_bbox_edge_crest_no_inflation() -> None:
    # Regression test for the 124%/140%/109% bug observed at hour-9
    # (2026-05-12 smoke test, bw03 multiple teeth). When the bone-crest
    # keypoint lands at a bbox edge laterally far from CEJ but slightly
    # apical of it, the OLD 2D-Euclidean formula reported >100% because
    # the lateral distance dominated the numerator. The new projection
    # formula correctly returns a small percentage proportional to the
    # axial component only.
    #
    # Setup mirrors bw03 tooth #2 (transposed to a clean origin): tooth
    # axis is vertical (~300 px root length). Bone crest is ~300 px
    # laterally offset but only 30 px below CEJ along the axis. The
    # axial bone loss is 30/300 = 10%; lateral pixels contribute nothing.
    cej = (300.0, 100.0)
    bone_crest = (0.0, 130.0)       # 300 px left, 30 px below along axis
    apex = (300.0, 400.0)            # 300 px straight down from CEJ
    pct = compute_bone_loss_pct(cej, bone_crest, apex)
    assert pct is not None
    # OLD formula: 100 * ||(-300, 30)|| / 300 = 100 * sqrt(90900)/300 ≈ 100.5%
    # NEW formula: 100 * 30 / 300 = 10.0%
    assert math.isclose(pct, 10.0, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_bone_loss_pct — edge cases
# ---------------------------------------------------------------------------


def test_compute_bone_loss_pct_none_cej() -> None:
    assert compute_bone_loss_pct(None, (0.0, 10.0), (0.0, 100.0)) is None


def test_compute_bone_loss_pct_none_bone_crest() -> None:
    assert compute_bone_loss_pct((0.0, 0.0), None, (0.0, 100.0)) is None


def test_compute_bone_loss_pct_none_apex() -> None:
    assert compute_bone_loss_pct((0.0, 0.0), (0.0, 10.0), None) is None


def test_compute_bone_loss_pct_zero_root_length() -> None:
    # CEJ == apex → axis-length zero → None.
    assert compute_bone_loss_pct((5.0, 5.0), (5.0, 6.0), (5.0, 5.0)) is None


def test_compute_bone_loss_pct_bone_crest_above_cej_returns_none() -> None:
    # CEJ at origin, apex straight down. Bone crest 5 px straight UP
    # from CEJ — bone-crest projects coronal to CEJ beyond the 2 px
    # noise tolerance. Anatomically impossible; function returns None.
    pct = compute_bone_loss_pct((0.0, 0.0), (0.0, -5.0), (0.0, 100.0))
    assert pct is None


def test_compute_bone_loss_pct_bone_crest_slightly_above_cej_tolerated() -> None:
    # 1 px above CEJ along axis — within the 2 px keypoint-noise
    # tolerance. Clamped to 0.0 rather than rejected.
    pct = compute_bone_loss_pct((0.0, 0.0), (0.0, -1.0), (0.0, 100.0))
    assert pct is not None
    assert math.isclose(pct, 0.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_bone_crest_lateral_perpendicular() -> None:
    # Bone crest perpendicular to the axis: projection = 0. Should NOT
    # reject (perpendicular is not "above"); should return 0% (no
    # measurable bone loss along the axis).
    pct = compute_bone_loss_pct((0.0, 0.0), (5.0, 0.0), (0.0, 100.0))
    assert pct is not None
    assert math.isclose(pct, 0.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_exact_overlap_crest_on_cej() -> None:
    # Bone crest coincides with CEJ → zero bone loss.
    pct = compute_bone_loss_pct((1.0, 1.0), (1.0, 1.0), (1.0, 50.0))
    assert pct is not None
    assert math.isclose(pct, 0.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_crest_past_apex_clamped_to_100() -> None:
    # Bone crest projects past the apex along the axis. Anatomically
    # impossible (bone gone past the root tip), but the staging
    # downstream still wants "severe". Clamp at 100%.
    pct = compute_bone_loss_pct((0.0, 0.0), (0.0, 150.0), (0.0, 100.0))
    assert pct is not None
    assert math.isclose(pct, 100.0, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# severity_tier — boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pct,expected",
    [
        (0.0, "mild"),
        (5.0, "mild"),
        (14.0, "mild"),
        (14.9999, "mild"),
        # Boundary at 15% — exactly 15 is moderate (inclusive lower).
        (15.0, "moderate"),
        (20.0, "moderate"),
        (32.0, "moderate"),
        (33.0, "moderate"),  # exactly 33 stays moderate (inclusive upper)
        (33.0001, "severe"),
        (34.0, "severe"),
        (75.0, "severe"),
        (100.0, "severe"),
    ],
)
def test_severity_tier_boundaries(pct: float, expected: str) -> None:
    assert severity_tier(pct) == expected


def test_severity_tier_none_input() -> None:
    assert severity_tier(None) is None
