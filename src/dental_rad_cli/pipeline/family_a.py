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


def band_centerline_y_at_x(
    band: np.ndarray,
    x: float,
    bbox_y_range: Optional[tuple[float, float]] = None,
) -> Optional[float]:
    """Median y of band pixels at integer column ``round(x)``.

    Treats the band's centerline as the per-column median of its
    non-zero pixels. For a buffered polyline (30-px-wide strip), this
    is a stable centerline approximation that doesn't require
    skeletonization — works directly on the proto-mask output.

    ``bbox_y_range`` (optional) restricts the y-axis scan to ``[y1, y2]``.
    Critical on bitewings: the same x column can carry CEJ band pixels
    in BOTH the upper AND lower arch frames; the unconstrained median
    picks an arbitrary cluster and produces cross-frame contamination
    (a ~16 mm "bone loss" reading on bw01 was diagnosed as exactly
    this — distance from upper-frame CEJ to lower-frame bone-crest).
    Restricting to the tooth's bbox y-range eliminates the failure
    mode by construction.

    Returns None if x is out of the image, the column has no band
    pixels in range, or (when bbox_y_range is given) no band pixels
    fall inside the bbox.
    """
    h, w = band.shape
    xi = int(round(x))
    if xi < 0 or xi >= w:
        return None
    column = band[:, xi]
    ys = np.flatnonzero(column)
    if ys.size == 0:
        return None
    if bbox_y_range is not None:
        y1, y2 = bbox_y_range
        ys = ys[(ys >= y1) & (ys <= y2)]
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
    x1, y1, x2, y2 = bbox
    bbox_y = (y1, y2)

    cej_m_y = band_centerline_y_at_x(cej_band, x1, bbox_y)
    cej_d_y = band_centerline_y_at_x(cej_band, x2, bbox_y)
    bone_m_y = band_centerline_y_at_x(bone_band, x1, bbox_y)
    bone_d_y = band_centerline_y_at_x(bone_band, x2, bbox_y)

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


def per_tooth_landmarks_via_masks(
    tooth_mask: np.ndarray,
    cej_band: np.ndarray,
    bone_mask: np.ndarray,
    px_per_mm: float,
) -> tuple[
    BoneLossSite,
    BoneLossSite,
    Optional[dict[str, Optional[tuple[float, float]]]],
]:
    """Anatomically-correct CEJ + bone-crest landmarks via mask intersection.

    Mirrors Lee/Kabir 2022's three-segmentation methodology: intersect
    the CEJ band with the tooth mask to get "CEJ ON this tooth"; same
    for bone. Then identify specific mesial/distal landmarks at the
    extremal points of these intersections. This guarantees landmarks
    sit ON the tooth surface — the same constraint a periodontal probe
    walks along during clinical measurement.

    The bbox-edge fallback in ``per_tooth_family_a`` placed landmarks
    at ``(bbox.x1, band_centerline_y)`` — which is approximately at
    the tooth's mesial edge but can be off by 5-30 px on noisy bboxes,
    and the band-centerline y at that x may sit in the interproximal
    bone septum rather than ON the tooth surface. This function fixes
    that by anchoring every landmark to a pixel that is BOTH inside
    the tooth mask AND inside the relevant band/mask.

    Landmark definitions (mirroring clinical anatomy):

    - **Mesial CEJ:** leftmost pixel of ``tooth_mask & cej_band``.
      The mesial-most point on the tooth where the CEJ band crosses
      the tooth's mesial surface.
    - **Distal CEJ:** rightmost pixel of the same intersection.
    - **Mesial bone-crest:** in the mesial half of the tooth, the
      most-CORONAL pixel of ``tooth_mask & bone_mask`` that is apical
      to the mesial CEJ. "Coronal" / "apical" direction is inferred
      from the CEJ centroid y vs the tooth-mask centroid y (mandibular:
      CEJ above center → apical = below; maxillary: CEJ below center
      → apical = above).
    - **Distal bone-crest:** same in the distal half.

    mm distance is computed as ``|bone_y - cej_y| / px_per_mm`` —
    vertical projection, approximating the tooth long axis. v0.6+
    will replace this with a PCA-derived true long-axis projection
    once we tune it against tilted-tooth cases.

    Returns ``(mesial_site, distal_site, positions_dict)``. positions_dict
    keys: ``cej_mesial``, ``cej_distal``, ``bone_mesial``, ``bone_distal``;
    each value is ``(x, y)`` or ``None``. Returns ``(None, None, None)``
    if no CEJ pixels overlap the tooth mask (no landmark possible).
    """
    cej_on_tooth = tooth_mask & cej_band
    bone_on_tooth = tooth_mask & bone_mask

    if not cej_on_tooth.any():
        return (
            BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_cej_at_site",
            ),
            BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_cej_at_site",
            ),
            None,
        )

    # CEJ landmark = leftmost/rightmost column of cej_on_tooth, at the
    # column's MEDIAN y. Picking the topmost pixel (np.argmin on the
    # flat (y, x) array) systematically biased the landmark to the
    # coronal edge of a 30-px-wide CEJ band → ~15 px (~0.9 mm) too high.
    # The clinically meaningful CEJ point is at the band's centerline.
    ys_cej, xs_cej = np.where(cej_on_tooth)
    min_x_cej = int(xs_cej.min())
    max_x_cej = int(xs_cej.max())
    cej_mesial = (
        float(min_x_cej),
        float(np.median(ys_cej[xs_cej == min_x_cej])),
    )
    cej_distal = (
        float(max_x_cej),
        float(np.median(ys_cej[xs_cej == max_x_cej])),
    )

    # Orientation: compare CEJ centroid y to tooth centroid y.
    # CEJ is typically near the cervical region — close to the crown.
    # If CEJ is in the upper half of the tooth mask → crown at top
    # (mandibular orientation in image space) → apical direction is +y.
    # Else maxillary → apical direction is -y.
    ys_tooth, xs_tooth = np.where(tooth_mask)
    if ys_tooth.size == 0:
        return (
            BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_tooth_mask",
            ),
            BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_tooth_mask",
            ),
            {
                "cej_mesial": cej_mesial,
                "cej_distal": cej_distal,
                "bone_mesial": None,
                "bone_distal": None,
            },
        )
    tooth_cy = float(ys_tooth.mean())
    cej_cy = (cej_mesial[1] + cej_distal[1]) / 2.0
    apical_sign = 1.0 if cej_cy < tooth_cy else -1.0

    # Find bone-crest landmark in each half of the tooth.
    positions: dict[str, Optional[tuple[float, float]]] = {
        "cej_mesial": cej_mesial,
        "cej_distal": cej_distal,
        "bone_mesial": None,
        "bone_distal": None,
    }

    if not bone_on_tooth.any():
        return (
            BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_bone_at_site",
            ),
            BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_bone_at_site",
            ),
            positions,
        )

    ys_bone, xs_bone = np.where(bone_on_tooth)
    tooth_mid_x = (float(xs_tooth.min()) + float(xs_tooth.max())) / 2.0

    def _bone_landmark(
        cej_pt: tuple[float, float], mesial_side: bool
    ) -> Optional[tuple[float, float]]:
        # Filter bone pixels to the relevant half of the tooth.
        side_mask = xs_bone <= tooth_mid_x if mesial_side else xs_bone > tooth_mid_x
        if not side_mask.any():
            return None
        ys_side = ys_bone[side_mask]
        xs_side = xs_bone[side_mask]
        # Filter to bone pixels apical to CEJ along the orientation axis.
        cej_y = cej_pt[1]
        if apical_sign > 0:  # apical = larger y
            apical_mask = ys_side > cej_y
        else:  # apical = smaller y
            apical_mask = ys_side < cej_y
        if not apical_mask.any():
            return None
        ys_apical = ys_side[apical_mask]
        xs_apical = xs_side[apical_mask]
        # Most-coronal pixel = closest to CEJ along the long axis.
        if apical_sign > 0:
            idx = int(np.argmin(ys_apical))  # smallest y = most coronal
        else:
            idx = int(np.argmax(ys_apical))  # largest y = most coronal
        return (float(xs_apical[idx]), float(ys_apical[idx]))

    bone_mesial = _bone_landmark(cej_mesial, mesial_side=True)
    bone_distal = _bone_landmark(cej_distal, mesial_side=False)
    positions["bone_mesial"] = bone_mesial
    positions["bone_distal"] = bone_distal

    def _site(
        cej_pt: tuple[float, float], bone_pt: Optional[tuple[float, float]]
    ) -> BoneLossSite:
        if bone_pt is None:
            return BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_bone_at_site",
            )
        if px_per_mm <= 0:
            return BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="no_calibration",
            )
        mm = abs(bone_pt[1] - cej_pt[1]) / px_per_mm
        if mm > _MAX_PLAUSIBLE_MM:
            return BoneLossSite(
                pct=None, tier=None, mm_estimate=None,
                reason="implausible_mm",
            )
        return BoneLossSite(
            pct=None,
            tier=severity_tier_mm(mm),
            mm_estimate=float(mm),
            reason=None,
        )

    return _site(cej_mesial, bone_mesial), _site(cej_distal, bone_distal), positions


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
