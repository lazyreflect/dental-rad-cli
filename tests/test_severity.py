"""Tests for `dental_rad_cli.pipeline.severity`.

Each numeric tolerance is explicit. Tier-boundary tests pin the
AAP thresholds: <15% mild, 15-33% moderate, >33% severe.
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
    # L_loss = 10, L_total = 100 → 10%.
    pct = compute_bone_loss_pct((0.0, 0.0), (0.0, 10.0), (0.0, 100.0))
    assert pct is not None
    assert math.isclose(pct, 10.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_diagonal_axis() -> None:
    # Diagonal: CEJ→apex is sqrt(200), CEJ→crest is sqrt(50). Ratio 0.5.
    pct = compute_bone_loss_pct((0.0, 0.0), (5.0, 5.0), (10.0, 10.0))
    assert pct is not None
    assert math.isclose(pct, 50.0, rel_tol=1e-9, abs_tol=1e-9)


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
    # CEJ == apex → denominator zero → None.
    assert compute_bone_loss_pct((5.0, 5.0), (5.0, 6.0), (5.0, 5.0)) is None


def test_compute_bone_loss_pct_bone_crest_above_cej_returns_none() -> None:
    # CEJ at origin, apex straight down. Bone crest is straight UP from
    # CEJ — anatomically impossible (the bone crest is on the crown side
    # of the CEJ). The function rejects with None rather than clamping.
    # Documented in severity.py module docstring.
    pct = compute_bone_loss_pct((0.0, 0.0), (0.0, -5.0), (0.0, 100.0))
    assert pct is None


def test_compute_bone_loss_pct_bone_crest_lateral_is_kept() -> None:
    # Bone crest perpendicular to apex direction → projection is zero;
    # tolerance allows zero. Should NOT reject; should return the raw
    # Euclidean ratio.
    pct = compute_bone_loss_pct((0.0, 0.0), (5.0, 0.0), (0.0, 100.0))
    assert pct is not None
    # L_loss = 5, L_total = 100 → 5%.
    assert math.isclose(pct, 5.0, rel_tol=1e-9, abs_tol=1e-9)


def test_compute_bone_loss_pct_exact_overlap_crest_on_cej() -> None:
    # Bone crest coincides with CEJ → zero bone loss.
    pct = compute_bone_loss_pct((1.0, 1.0), (1.0, 1.0), (1.0, 50.0))
    assert pct is not None
    assert math.isclose(pct, 0.0, rel_tol=1e-9, abs_tol=1e-9)


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
