"""Tests for the v2 Family A routing in `analyze._build_findings_from_stages`.

Validates the three operational states defined by the karpathy-findings
ship strategy:

1. Polyline model loaded + high confidence → Family A mm pathway runs;
   `BoneLossSite.mm_estimate` populated; tier derived from mm.
2. Polyline model loaded + low confidence → emit "low_model_confidence"
   findings; DO NOT fall back to the v1 apex pathway (apex predictions
   are unreliable on the same images the polyline rejected).
3. Polyline model NOT loaded → v1 keypoint+apex pathway runs as legacy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from dental_rad_cli.analyze import _build_findings_from_stages
from dental_rad_cli.schema import CariesFinding


def _make_band(
    shape: Tuple[int, int],
    y_center: int,
    half_width: int = 15,
) -> np.ndarray:
    """Build a horizontal band mask centered at y_center."""
    h, w = shape
    band = np.zeros((h, w), dtype=bool)
    y1 = max(0, y_center - half_width)
    y2 = min(h, y_center + half_width + 1)
    band[y1:y2, :] = True
    return band


def _make_detection(fdi: str, bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    return {
        "fdi": fdi,
        "bbox": bbox,
        "confidence": 0.9,
        "root_class": "single",
    }


# ---------------------------------------------------------------------------
# State 1: polyline loaded + high confidence → Family A
# ---------------------------------------------------------------------------


def test_family_a_routes_high_conf_polyline_to_mm_pathway() -> None:
    image_shape = (400, 400)
    # CEJ band centered at y=100, bone band centered at y=130 → ~30px = ~3mm
    # at the v0 calibration if bbox height = 21*10 = 210, px_per_mm = 10.
    cej_band = _make_band(image_shape, y_center=100)
    bone_polys = [[(0.0, 115.0), (400.0, 115.0), (400.0, 145.0), (0.0, 145.0)]]
    detections = [_make_detection("1", (100.0, 50.0, 200.0, 260.0))]

    teeth, _summary, low_conf = _build_findings_from_stages(
        detections=detections,
        keypoints=[],  # no keypoints — Family A doesn't need them
        tooth_polys=[],
        bone_polys=bone_polys,
        caries=[],
        image_shape=image_shape,
        cej_band=cej_band,
        cej_band_max_conf=0.95,  # above CEJ_POLYLINE_CONF_THRESHOLD
    )

    assert len(teeth) == 1
    site = teeth[0].bone_loss.mesial
    assert site is not None
    # mm_estimate should be populated (Family A produced a measurement).
    assert site.mm_estimate is not None
    # The legacy pct field stays None in Family A mode (apex-free).
    assert site.pct is None


# ---------------------------------------------------------------------------
# State 2: polyline loaded + low confidence → low_model_confidence
# ---------------------------------------------------------------------------


def test_polyline_below_threshold_blocks_apex_fallback() -> None:
    """When the polyline model is loaded but confidence is below the
    threshold, the apex-pathway must NOT run — its predictions are
    unreliable on the same image distribution. Emit
    low_model_confidence instead.
    """
    image_shape = (400, 400)
    cej_band = _make_band(image_shape, y_center=100)
    bone_polys = [[(0.0, 115.0), (400.0, 115.0), (400.0, 145.0), (0.0, 145.0)]]
    detections = [_make_detection("1", (100.0, 50.0, 200.0, 260.0))]
    # Apex pathway would otherwise produce a measurement — keypoints
    # are populated. But polyline is BELOW threshold, so we expect the
    # apex pathway to be skipped.
    keypoints = [{
        "fdi": "1",
        "cej": [(120.0, 100.0), (180.0, 100.0)],
        "bone_crest": [(120.0, 130.0), (180.0, 130.0)],
        "apex": (150.0, 250.0),
    }]

    teeth, _summary, low_conf = _build_findings_from_stages(
        detections=detections,
        keypoints=keypoints,
        tooth_polys=[],
        bone_polys=bone_polys,
        caries=[],
        image_shape=image_shape,
        cej_band=cej_band,
        cej_band_max_conf=0.3,  # below CEJ_POLYLINE_CONF_THRESHOLD (0.5)
    )

    assert len(teeth) == 1
    site = teeth[0].bone_loss.mesial
    # Apex pathway must NOT have populated this — pct should be None.
    assert site is None or site.pct is None
    # mm_estimate also None (Family A didn't run since below threshold).
    assert site is None or site.mm_estimate is None
    # Must have emitted a low_model_confidence finding for this tooth.
    low_model_findings = [
        f for f in low_conf if f.reason == "low_model_confidence"
    ]
    assert len(low_model_findings) >= 1
    assert low_model_findings[0].tooth == "1"


# ---------------------------------------------------------------------------
# State 3: polyline NOT loaded → v1 keypoint+apex pathway
# ---------------------------------------------------------------------------


def test_no_polyline_runs_legacy_keypoint_pathway() -> None:
    """When the polyline model is absent entirely (cej_band is None),
    the v1 keypoint+apex pathway runs as before.
    """
    image_shape = (400, 400)
    bone_polys = [[(0.0, 115.0), (400.0, 115.0), (400.0, 145.0), (0.0, 145.0)]]
    detections = [_make_detection("1", (100.0, 50.0, 200.0, 260.0))]
    keypoints = [{
        "fdi": "1",
        "cej": [(120.0, 100.0), (180.0, 100.0)],
        "bone_crest": [(120.0, 130.0), (180.0, 130.0)],
        "apex": (150.0, 250.0),
    }]

    teeth, _summary, low_conf = _build_findings_from_stages(
        detections=detections,
        keypoints=keypoints,
        tooth_polys=[],
        bone_polys=bone_polys,
        caries=[],
        image_shape=image_shape,
        cej_band=None,  # polyline model not loaded
        cej_band_max_conf=0.0,
    )

    assert len(teeth) == 1
    site = teeth[0].bone_loss.mesial
    assert site is not None
    # pct should be populated by the keypoint pathway.
    assert site.pct is not None
    # mm_estimate stays None (legacy mode doesn't compute mm).
    assert site.mm_estimate is None
    # No low_model_confidence findings — we used the legacy path.
    assert not any(
        f.reason == "low_model_confidence" for f in low_conf
    )
