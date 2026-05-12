"""Family A bone-loss math — apex-free mm CEJ→bone-crest distance.

Replaces the Wimalasiri-style percent-of-root-length formula in
`severity.py` with the mm-based formulation used by the FDA-cleared
commercial vendors (Overjet K210187, Adravision K232440, Pearl
K243230, AlGhaihab/Denti.AI 2025). Apex-free → works on bitewings AND
periapicals without requiring the root tip to be in frame.

Clinical rationale (Joseph 2026-05-12): "Not needing the apex will be
much more useful clinically. It gets missed sometimes with PA
radiographs. Having to retake x-rays exposes the patient to more
radiation. If we can accomplish what we need with just bitewings and
anterior PAs then even less radiation."

Inputs are predicted CEJ band + predicted bone-crest band as 2D
binary masks at image resolution, plus per-tooth bboxes. Per tooth,
the band centerline y is extracted at ``bbox.x1`` (mesial) and
``bbox.x2`` (distal); the vertical difference / px_per_mm is the
clinical mm number.

AAP staging thresholds (AAP/EFP 2017):

    < 2 mm   → healthy (no stage assigned)
    2-4 mm   → Stage I  (mild)
    4-6 mm   → Stage II (moderate)
    ≥ 6 mm   → Stage III (severe)

Px→mm calibration assumed to be a per-image scalar passed in by the
caller. v0 uses median tooth bbox height / 21 mm population anchor;
v0.5 will swap in per-tooth-class priors from the parallel
`dental-tooth-numbering` substrate work.

Pure functions. No file I/O, no model loading, no global state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dental_rad_cli.schema import BoneLossSite, SeverityTier

# AAP/EFP 2017 mm thresholds, mapping to the schema's 3-tier severity.
# Below `_MILD_MIN_MM` is "healthy" — tier = None.
_MILD_MIN_MM: float = 2.0
_MODERATE_MIN_MM: float = 4.0
_SEVERE_MIN_MM: float = 6.0

# Upper bound on a plausible CEJ→bone-crest mm distance. Beyond this,
# the measurement is almost certainly a model error (e.g., the band
# centerline lookup hit a neighboring tooth's annotation). Real
# clinical bone loss tops out around 15-20 mm before tooth loss;
# 25 mm is a safe ceiling that still catches catastrophic errors.
_MAX_PLAUSIBLE_MM: float = 25.0


def band_centerline_y_at_x(band: np.ndarray, x: float) -> Optional[float]:
    """Median y of band pixels at integer column ``round(x)``.

    Treats the band's centerline as the per-column median of its
    non-zero pixels. For a buffered polyline (30-px-wide strip), this
    is a stable centerline approximation that doesn't require
    skeletonization — works directly on the proto-mask output.

    Returns None if x is out of the image or the column has no band
    pixels (degenerate site).
    """
    h, w = band.shape
    xi = int(round(x))
    if xi < 0 or xi >= w:
        return None
    column = band[:, xi]
    ys = np.flatnonzero(column)
    if ys.size == 0:
        return None
    return float(np.median(ys))


def site_mm(
    cej_y: Optional[float],
    bone_y: Optional[float],
    px_per_mm: float,
) -> Optional[float]:
    """Compute mm CEJ→bone-crest at one site (mesial or distal).

    Uses absolute distance: ``|bone_y - cej_y| / px_per_mm``. The
    absolute is load-bearing — DenPAR mixes maxillary and mandibular
    periapicals. In mandibular PAs the crown is at the top of the
    image so bone is at higher y than CEJ (bone_y - cej_y > 0). In
    maxillary PAs the crown is at the bottom and the root points up,
    so anatomically-correct bone-crest sits at LOWER y than CEJ
    (bone_y - cej_y < 0). A signed check would incorrectly reject the
    maxillary half of the dataset (24% rejection rate observed on
    the GT-band smoke test before this fix).

    Jaw-aware signed measurement is a v0.5 concern: combine with FDI
    numbering from the parallel `dental-tooth-numbering` substrate to
    know maxillary vs mandibular per tooth, then validate the sign
    against expected anatomy.

    Returns None when either landmark is missing, calibration is
    invalid, or the absolute distance exceeds `_MAX_PLAUSIBLE_MM`
    (catastrophic model error — neighboring-tooth pollution etc.).
    """
    if cej_y is None or bone_y is None:
        return None
    if px_per_mm <= 0:
        return None
    mm = abs(bone_y - cej_y) / px_per_mm
    if mm > _MAX_PLAUSIBLE_MM:
        return None
    return float(mm)


def severity_tier_mm(mm: Optional[float]) -> Optional[SeverityTier]:
    """Map mm CEJ→bone-crest to AAP/EFP 2017 severity tier.

    None in → None out (no measurement). Sub-threshold mm → None
    (healthy — not a tier).
    """
    if mm is None:
        return None
    if mm < _MILD_MIN_MM:
        return None  # Healthy — not a staged tier.
    if mm < _MODERATE_MIN_MM:
        return "mild"
    if mm < _SEVERE_MIN_MM:
        return "moderate"
    return "severe"


def per_tooth_family_a(
    cej_band: np.ndarray,
    bone_band: np.ndarray,
    bbox: tuple[float, float, float, float],
    px_per_mm: float,
) -> tuple[BoneLossSite, BoneLossSite]:
    """Compute Family A bone-loss for one tooth at mesial + distal sites.

    Returns ``(mesial, distal)`` ``BoneLossSite`` instances. Each
    site's ``mm_estimate`` carries the mm CEJ→bone-crest distance;
    its ``tier`` is derived from mm via AAP thresholds. The ``pct``
    field is left as None (Family A doesn't compute percent — apex-
    free by design). The ``reason`` field carries a machine-readable
    code when a site can't be measured.
    """
    x1, _y1, x2, _y2 = bbox

    cej_m_y = band_centerline_y_at_x(cej_band, x1)
    cej_d_y = band_centerline_y_at_x(cej_band, x2)
    bone_m_y = band_centerline_y_at_x(bone_band, x1)
    bone_d_y = band_centerline_y_at_x(bone_band, x2)

    def _build_site(
        cej_y: Optional[float], bone_y: Optional[float]
    ) -> BoneLossSite:
        mm = site_mm(cej_y, bone_y, px_per_mm)
        if mm is None:
            # Figure out why for the reason code.
            if cej_y is None and bone_y is None:
                reason = "no_landmarks_at_site"
            elif cej_y is None:
                reason = "no_cej_at_site"
            elif bone_y is None:
                reason = "no_bone_at_site"
            elif px_per_mm <= 0:
                reason = "no_calibration"
            else:
                reason = "implausible_mm"
            return BoneLossSite(
                pct=None, tier=None, reason=reason, mm_estimate=None
            )
        return BoneLossSite(
            pct=None,
            tier=severity_tier_mm(mm),
            reason=None,
            mm_estimate=mm,
        )

    mesial = _build_site(cej_m_y, bone_m_y)
    distal = _build_site(cej_d_y, bone_d_y)
    return mesial, distal


def calibrate_px_per_mm(
    bboxes: list[tuple[float, float, float, float]],
    mean_tooth_height_mm: float = 21.0,
) -> Optional[float]:
    """V0 px→mm calibration via median bbox height / population mean.

    21 mm is a rough population anchor for periapical tooth height
    (full crown + root). v0.5 should swap in per-tooth-class priors
    from the dental-tooth-numbering substrate (e.g. max central
    incisor 22 mm, max first molar 20 mm crown+root).

    Returns None if no bboxes (no calibration possible).
    """
    heights = [
        (bbox[3] - bbox[1]) for bbox in bboxes if bbox[3] > bbox[1]
    ]
    if not heights:
        return None
    return float(np.median(heights)) / mean_tooth_height_mm
