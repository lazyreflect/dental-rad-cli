"""Tests for `dental_rad_cli.pipeline.family_a`.

Pure-math unit tests of the apex-free mm CEJ→bone-crest head.
Validates orientation-agnostic distance, AAP tier mapping, the
sanity cap, and per-tooth site construction from synthetic band
masks.
"""

from __future__ import annotations

import numpy as np

from dental_rad_cli.pipeline.family_a import (
    band_centerline_y_at_x,
    calibrate_px_per_mm,
    per_tooth_family_a,
    per_tooth_landmarks_via_masks,
    severity_tier_mm,
    site_mm,
)


# ---------------------------------------------------------------------------
# site_mm — absolute distance + sanity cap
# ---------------------------------------------------------------------------


def test_site_mm_mandibular_orientation() -> None:
    # Mandibular PA: crown at top, bone at higher y than CEJ.
    # CEJ y=200, bone y=230, px_per_mm=10 → 3 mm.
    assert site_mm(200.0, 230.0, 10.0) == 3.0


def test_site_mm_maxillary_orientation() -> None:
    # Maxillary PA: crown at bottom, bone at LOWER y than CEJ.
    # CEJ y=300, bone y=270 → bone-y is negative diff. Absolute → 3 mm.
    assert site_mm(300.0, 270.0, 10.0) == 3.0


def test_site_mm_returns_zero_when_landmarks_coincide() -> None:
    # CEJ and bone at the same y — anatomically near-zero bone loss.
    assert site_mm(200.0, 200.0, 10.0) == 0.0


def test_site_mm_none_when_cej_missing() -> None:
    assert site_mm(None, 230.0, 10.0) is None


def test_site_mm_none_when_bone_missing() -> None:
    assert site_mm(200.0, None, 10.0) is None


def test_site_mm_none_when_calibration_invalid() -> None:
    assert site_mm(200.0, 230.0, 0.0) is None
    assert site_mm(200.0, 230.0, -1.0) is None


def test_site_mm_rejects_implausibly_large_distance() -> None:
    # 30 mm bone loss is past the sanity cap (25 mm).
    assert site_mm(100.0, 400.0, 10.0) is None  # 300 px / 10 = 30 mm


def test_site_mm_accepts_distance_at_cap() -> None:
    # Exactly at the 25 mm cap is accepted.
    assert site_mm(100.0, 350.0, 10.0) == 25.0


# ---------------------------------------------------------------------------
# severity_tier_mm — AAP staging thresholds
# ---------------------------------------------------------------------------


def test_severity_tier_below_mild_is_none() -> None:
    # < 2 mm = healthy, no tier assigned.
    assert severity_tier_mm(1.5) is None
    assert severity_tier_mm(0.0) is None


def test_severity_tier_at_mild_boundary_is_mild() -> None:
    # 2.0 mm is exactly the lower bound for mild.
    assert severity_tier_mm(2.0) == "mild"


def test_severity_tier_in_mild_range() -> None:
    assert severity_tier_mm(3.0) == "mild"
    assert severity_tier_mm(3.99) == "mild"


def test_severity_tier_at_moderate_boundary_is_moderate() -> None:
    assert severity_tier_mm(4.0) == "moderate"


def test_severity_tier_in_moderate_range() -> None:
    assert severity_tier_mm(5.0) == "moderate"
    assert severity_tier_mm(5.99) == "moderate"


def test_severity_tier_at_severe_boundary_is_severe() -> None:
    assert severity_tier_mm(6.0) == "severe"


def test_severity_tier_well_into_severe() -> None:
    assert severity_tier_mm(15.0) == "severe"


def test_severity_tier_none_in_none_out() -> None:
    assert severity_tier_mm(None) is None


# ---------------------------------------------------------------------------
# band_centerline_y_at_x — column median
# ---------------------------------------------------------------------------


def test_band_centerline_returns_none_for_empty_column() -> None:
    band = np.zeros((100, 100), dtype=bool)
    assert band_centerline_y_at_x(band, 50.0) is None


def test_band_centerline_returns_none_for_out_of_image_x() -> None:
    band = np.ones((100, 100), dtype=bool)
    assert band_centerline_y_at_x(band, -1.0) is None
    assert band_centerline_y_at_x(band, 100.0) is None


def test_band_centerline_single_pixel_column() -> None:
    band = np.zeros((100, 100), dtype=bool)
    band[42, 50] = True
    assert band_centerline_y_at_x(band, 50.0) == 42.0


def test_band_centerline_multi_pixel_column_is_median() -> None:
    # Thick band column with pixels at y=10..30. Median = 20.
    band = np.zeros((100, 100), dtype=bool)
    band[10:31, 50] = True  # 21 pixels: 10..30 inclusive.
    assert band_centerline_y_at_x(band, 50.0) == 20.0


def test_band_centerline_rounds_fractional_x() -> None:
    band = np.zeros((100, 100), dtype=bool)
    band[42, 50] = True
    # x=49.6 rounds to 50.
    assert band_centerline_y_at_x(band, 49.6) == 42.0
    # x=50.4 rounds to 50.
    assert band_centerline_y_at_x(band, 50.4) == 42.0


# ---------------------------------------------------------------------------
# per_tooth_family_a — end-to-end on synthetic bands
# ---------------------------------------------------------------------------


def test_per_tooth_family_a_healthy_tooth_mandibular() -> None:
    """Mandibular tooth: bbox at top, CEJ near crown, bone just below."""
    cej_band = np.zeros((400, 400), dtype=bool)
    bone_band = np.zeros((400, 400), dtype=bool)
    # CEJ at y=100, bone at y=110 → 10 px = 1 mm at px_per_mm=10.
    cej_band[95:106, :] = True
    bone_band[105:116, :] = True
    bbox = (50.0, 50.0, 150.0, 300.0)
    mesial, distal = per_tooth_family_a(cej_band, bone_band, bbox, 10.0)
    # Both sites should produce ~1 mm, tier=None (sub-mild).
    assert mesial.mm_estimate is not None
    assert 0.5 < mesial.mm_estimate < 1.5
    assert mesial.tier is None  # healthy
    assert distal.mm_estimate is not None
    assert 0.5 < distal.mm_estimate < 1.5


def test_per_tooth_family_a_severe_loss_maxillary() -> None:
    """Maxillary tooth: CEJ at y=300, bone-crest at y=240 (60 px = 6 mm)."""
    cej_band = np.zeros((400, 400), dtype=bool)
    bone_band = np.zeros((400, 400), dtype=bool)
    cej_band[295:306, :] = True  # CEJ centerline at y=300
    bone_band[235:246, :] = True  # bone centerline at y=240 (above CEJ)
    bbox = (50.0, 50.0, 150.0, 350.0)
    mesial, distal = per_tooth_family_a(cej_band, bone_band, bbox, 10.0)
    # Both sites should produce ~6 mm, tier=severe.
    assert mesial.mm_estimate is not None
    assert 5.5 < mesial.mm_estimate < 6.5
    assert mesial.tier == "severe"
    assert distal.tier == "severe"


def test_per_tooth_family_a_no_cej_at_site() -> None:
    cej_band = np.zeros((400, 400), dtype=bool)
    bone_band = np.zeros((400, 400), dtype=bool)
    # CEJ exists only at x=100..150; mesial=50 won't find it.
    cej_band[95:106, 100:151] = True
    bone_band[125:136, :] = True
    bbox = (50.0, 50.0, 150.0, 300.0)
    mesial, distal = per_tooth_family_a(cej_band, bone_band, bbox, 10.0)
    # Mesial site (x=50) — no CEJ pixel → reason = "no_cej_at_site".
    assert mesial.mm_estimate is None
    assert mesial.reason == "no_cej_at_site"
    # Distal site (x=150) — both bands present → mm computed.
    assert distal.mm_estimate is not None


# ---------------------------------------------------------------------------
# calibrate_px_per_mm
# ---------------------------------------------------------------------------


def test_calibrate_px_per_mm_returns_none_for_empty_input() -> None:
    assert calibrate_px_per_mm([]) is None


def test_calibrate_px_per_mm_returns_none_for_degenerate_bboxes() -> None:
    # Zero-height bboxes — skip.
    assert calibrate_px_per_mm([(0.0, 100.0, 50.0, 100.0)]) is None


def test_calibrate_px_per_mm_uses_median() -> None:
    # Heights 100, 200, 300 — median 200. 200 / 21 ≈ 9.52.
    bboxes = [
        (0.0, 0.0, 50.0, 100.0),
        (0.0, 0.0, 50.0, 200.0),
        (0.0, 0.0, 50.0, 300.0),
    ]
    result = calibrate_px_per_mm(bboxes)
    assert result is not None
    assert abs(result - (200.0 / 21.0)) < 1e-6


def test_calibrate_px_per_mm_respects_custom_anchor() -> None:
    # Custom mean tooth height = 10 mm; bbox height 100 → 10 px/mm.
    bboxes = [(0.0, 0.0, 50.0, 100.0)]
    assert calibrate_px_per_mm(bboxes, mean_tooth_height_mm=10.0) == 10.0


# ---------------------------------------------------------------------------
# per_tooth_landmarks_via_masks — Lee/Kabir-style anatomical landmarks
# ---------------------------------------------------------------------------


def _rect_mask(shape: tuple, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def test_landmarks_via_masks_mandibular_orientation() -> None:
    """Mandibular tooth: CEJ in upper part, bone below."""
    shape = (400, 400)
    tooth = _rect_mask(shape, 100, 50, 200, 300)
    cej_band = _rect_mask(shape, 0, 100, 400, 116)
    bone = _rect_mask(shape, 0, 140, 400, 400)
    mesial, distal, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
    )
    assert mesial.mm_estimate is not None
    # CEJ at band-center y=107.5 (median of band 100-115), bone at y=140
    # → 32.5 px / 10 = 3.25 mm
    assert 3.0 < mesial.mm_estimate < 3.5
    assert pos["cej_mesial"][0] == 100.0  # leftmost CEJ-on-tooth = mesial edge
    assert pos["cej_distal"][0] == 199.0
    # Bone landmarks are inside the tooth + apical to CEJ.
    assert pos["bone_mesial"][1] > pos["cej_mesial"][1]  # bone below CEJ


def test_landmarks_via_masks_maxillary_orientation() -> None:
    """Maxillary tooth: CEJ in lower part, bone above."""
    shape = (400, 400)
    tooth = _rect_mask(shape, 100, 100, 200, 350)
    cej_band = _rect_mask(shape, 0, 290, 400, 306)
    bone = _rect_mask(shape, 0, 0, 400, 240)
    mesial, distal, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
    )
    assert mesial.mm_estimate is not None
    # Inverted orientation handled: bone_y < cej_y, abs distance still positive.
    assert pos["bone_mesial"][1] < pos["cej_mesial"][1]  # bone above CEJ
    # CEJ at band-center y=297.5 (median of band 290-305), bone at y=239
    # → 58.5 px / 10 = 5.85 mm
    assert 5.5 < mesial.mm_estimate < 6.0


def test_landmarks_via_masks_no_cej_overlap_returns_none() -> None:
    shape = (400, 400)
    tooth = _rect_mask(shape, 100, 50, 200, 300)
    cej_band = _rect_mask(shape, 0, 350, 400, 360)  # CEJ band below tooth
    bone = _rect_mask(shape, 0, 200, 400, 400)
    mesial, distal, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
    )
    assert pos is None
    assert mesial.reason == "no_cej_at_site"


def test_landmarks_via_masks_no_bone_overlap_returns_reason() -> None:
    shape = (400, 400)
    tooth = _rect_mask(shape, 100, 50, 200, 300)
    cej_band = _rect_mask(shape, 0, 100, 400, 116)
    bone = _rect_mask(shape, 250, 0, 400, 400)  # bone away from tooth
    mesial, distal, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
    )
    assert pos is not None
    assert pos["cej_mesial"] is not None
    assert pos["bone_mesial"] is None
    assert mesial.reason == "no_bone_at_site"


def test_landmarks_via_masks_implausible_distance_rejected() -> None:
    """Cap rejects > 25 mm distance — for catastrophic model errors."""
    shape = (400, 400)
    tooth = _rect_mask(shape, 100, 50, 200, 350)
    cej_band = _rect_mask(shape, 0, 60, 400, 76)
    # Bone-on-tooth far below CEJ → > 25 mm at 5 px/mm.
    bone = _rect_mask(shape, 0, 250, 400, 400)
    mesial, distal, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=5.0,
    )
    assert mesial.mm_estimate is None
    assert mesial.reason == "implausible_mm"


# ---------------------------------------------------------------------------
# landmark_rule parameterization (BRneg-2 / 2026-05-12)
# ---------------------------------------------------------------------------


def _setup_bimodal_bone_case(shape=(400, 400)):
    """Construct a fixture where bone-on-tooth-ring has both a shallow
    cluster (just below CEJ) and a deep cluster (further apical).
    Simulates the 907-style wide bone-mask distribution where different
    rules give noticeably different bone-y landmarks."""
    tooth = _rect_mask(shape, 100, 50, 200, 350)
    cej_band = _rect_mask(shape, 0, 90, 400, 106)
    bone = np.zeros(shape, dtype=bool)
    # Shallow cluster (just below CEJ).
    bone[115:130, 0:400] = True
    # Deep cluster (much further apical).
    bone[260:280, 0:400] = True
    return tooth, cej_band, bone


def test_landmark_rule_min_y_half_picks_shallow_on_bimodal() -> None:
    tooth, cej_band, bone = _setup_bimodal_bone_case()
    mesial, _, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="min_y_half",
    )
    # min_y picks the SHALLOW cluster → small mm.
    assert mesial.mm_estimate is not None
    assert mesial.mm_estimate < 2.5  # ~1.5 mm range (15 px / 10)


def test_landmark_rule_max_y_half_picks_deep_on_bimodal() -> None:
    tooth, cej_band, bone = _setup_bimodal_bone_case()
    mesial, _, pos = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="max_y_half",
    )
    # max_y picks the DEEP cluster → large mm.
    assert mesial.mm_estimate is not None
    assert mesial.mm_estimate > 15.0  # ~17 mm or more


def test_landmark_rule_median_y_half_differs_from_min() -> None:
    tooth, cej_band, bone = _setup_bimodal_bone_case()
    mesial_min, _, _ = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="min_y_half",
    )
    mesial_median, _, _ = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="median_y_half",
    )
    # median follows the larger cluster; on bimodal distributions it
    # gives a meaningfully different y than min. Whether shallow or
    # deep depends on which cluster is bigger.
    assert mesial_min.mm_estimate is not None
    assert mesial_median.mm_estimate is not None
    assert abs(mesial_median.mm_estimate - mesial_min.mm_estimate) > 1.0


def test_landmark_rule_wide_aware_uses_median_when_spread_large() -> None:
    tooth, cej_band, bone = _setup_bimodal_bone_case()
    # Spread y-range ≈ 260 - 115 = 145 px >> 50 → wide → median.
    mesial_wide, _, _ = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="wide_aware",
    )
    mesial_median, _, _ = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="median_y_half",
    )
    # Wide-aware fell into the median branch → matches median rule.
    assert mesial_wide.mm_estimate is not None
    assert mesial_median.mm_estimate is not None
    assert abs(mesial_wide.mm_estimate - mesial_median.mm_estimate) < 0.5


def test_landmark_rule_wide_aware_uses_min_when_spread_narrow() -> None:
    # Thin bone band → narrow spread → wide_aware uses min_y_half.
    shape = (400, 400)
    tooth = _rect_mask(shape, 100, 50, 200, 350)
    cej_band = _rect_mask(shape, 0, 90, 400, 106)
    bone = _rect_mask(shape, 0, 115, 400, 125)  # thin band, ~10px spread
    mesial_wide, _, _ = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="wide_aware",
    )
    mesial_min, _, _ = per_tooth_landmarks_via_masks(
        tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        landmark_rule="min_y_half",
    )
    # Narrow spread → wide_aware falls into the min branch.
    assert mesial_wide.mm_estimate is not None
    assert mesial_min.mm_estimate is not None
    assert abs(mesial_wide.mm_estimate - mesial_min.mm_estimate) < 0.1


def test_landmark_rule_unknown_raises() -> None:
    tooth, cej_band, bone = _setup_bimodal_bone_case()
    try:
        per_tooth_landmarks_via_masks(
            tooth, cej_band, bone, px_per_mm=10.0,
            landmark_rule="nonsense_rule",
        )
    except ValueError as e:
        assert "unknown landmark_rule" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown landmark_rule")


def test_landmark_rule_env_var_fallback() -> None:
    """When landmark_rule=None, falls back to DENTAL_RAD_LANDMARK_RULE env var."""
    import os
    tooth, cej_band, bone = _setup_bimodal_bone_case()
    prev = os.environ.get("DENTAL_RAD_LANDMARK_RULE")
    try:
        os.environ["DENTAL_RAD_LANDMARK_RULE"] = "max_y_half"
        mesial, _, _ = per_tooth_landmarks_via_masks(
            tooth, cej_band, bone, px_per_mm=10.0, bone_erosion_px=0,
        )
        # Should pick deep cluster via max_y_half from env.
        assert mesial.mm_estimate is not None
        assert mesial.mm_estimate > 15.0
    finally:
        if prev is None:
            os.environ.pop("DENTAL_RAD_LANDMARK_RULE", None)
        else:
            os.environ["DENTAL_RAD_LANDMARK_RULE"] = prev
