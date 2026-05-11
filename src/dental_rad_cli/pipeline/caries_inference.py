"""Caries inference helper — YOLOv8s 3-class → :class:`CariesFinding` list.

Runs the trained caries detector against a CLAHE-preprocessed image and
emits :class:`dental_rad_cli.schema.CariesFinding` instances.

Class-to-schema mapping (full rationale in ``docs/caries-class-mapping.md``)::

    model class 0 ("initial",  RA1+RA2+RA3) → schema depth "E1"
    model class 1 ("moderate", RB4+RC5)     → schema depth "D1"
    model class 2 ("deep",     RC6)         → schema depth "D3"

The intermediate ``E2`` and ``D2`` tiers of the schema's ``CariesDepth``
literal are reserved for a future 5-class model; v0 emits only ``E1``,
``D1``, and ``D3``.

Surface determination
---------------------

The caries lesion's surface (mesial / distal / occlusal / etc.) is
inferred from the geometric relationship between the caries bbox and
the parent tooth bbox:

- caries-bbox-center x < tooth-bbox-center x → ``"mesial"``
  (caries to the left of tooth center — patient-anatomy convention
  matches the right side of the radiograph for the patient's left;
  the rule layer downstream handles laterality if needed)
- caries-bbox-center x > tooth-bbox-center x → ``"distal"``

If ``tooth_bboxes`` is ``None`` or no parent tooth is found
(caries-bbox center not contained in any tooth bbox), surface
defaults to ``"unknown"`` so the orchestrator can flag the finding
for human review rather than guess.

The orchestrator (``analyze.py``) is expected to call this helper
**after** tooth detection runs, so the tooth bbox list is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from ..schema import CariesDepth, CariesFinding, ToothFinding
from ..training.preprocess import apply_clahe

# Model class id (3-class collapse) → schema CariesDepth literal.
# See docs/caries-class-mapping.md for the rationale.
_CLASS_TO_DEPTH: Final[dict[int, CariesDepth]] = {
    0: "E1",  # initial (enamel-confined)
    1: "D1",  # moderate (outer dentin)
    2: "D3",  # deep (inner dentin / pulp-near)
}

# Inference defaults — match the methodology cadence for caries:
# 0.5 confidence floor, standard NMS IoU 0.45.
_DEFAULT_CONF: Final[float] = 0.5
_DEFAULT_IOU: Final[float] = 0.45


def detect_caries(
    image: np.ndarray,
    model_weights: Path,
    tooth_bboxes: list[ToothFinding] | None = None,
    conf_threshold: float = _DEFAULT_CONF,
    iou_threshold: float = _DEFAULT_IOU,
) -> list[CariesFinding]:
    """Run YOLOv8s caries inference and return :class:`CariesFinding` rows.

    Args:
        image: RGB radiograph as a uint8 numpy array (H, W, 3). CLAHE
            preprocessing is applied inside this function — callers
            should pass the **raw** RGB image, not a pre-CLAHEd one.
        model_weights: Path to the trained ``.pt`` weights file.
        tooth_bboxes: Optional list of :class:`ToothFinding` from the
            upstream tooth-detection stage. Used to assign each caries
            lesion to a parent tooth and infer the mesial/distal
            surface. If ``None`` (no tooth detection run yet), every
            finding's ``surface`` is ``"unknown"``.
        conf_threshold: Confidence floor; predictions below this are
            dropped. Defaults to ``0.5``.
        iou_threshold: NMS IoU threshold. Defaults to ``0.45``.

    Returns:
        A list of :class:`CariesFinding`, one per surviving detection.
        Empty list if no caries detected above the threshold.
    """
    # Local heavy imports — same lazy pattern as analyze.py + training.
    from ultralytics import YOLO  # type: ignore

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"detect_caries expects (H, W, 3) RGB; got shape {image.shape}"
        )

    preprocessed = apply_clahe(image)

    model = YOLO(str(model_weights))
    results = model.predict(
        preprocessed,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    findings: list[CariesFinding] = []
    if not results:
        return findings

    # Ultralytics returns a list (one per input image). We passed one.
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return findings

    # Each box has .xyxy (tensor [N,4]), .conf (tensor [N]), .cls (tensor [N]).
    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls).astype(int)

    for i in range(len(xyxy)):
        cls_id = int(classes[i])
        if cls_id not in _CLASS_TO_DEPTH:
            # Unknown class id — skip rather than guess.
            continue
        conf = float(confs[i])
        if conf < conf_threshold:
            continue  # defensive — Ultralytics should already filter.

        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        surface = _surface_for_caries((x1, y1, x2, y2), tooth_bboxes)
        findings.append(
            CariesFinding(
                surface=surface,
                depth=_CLASS_TO_DEPTH[cls_id],
                bbox=(x1, y1, x2, y2),
                confidence=conf,
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(tensor_like) -> np.ndarray:
    """Convert a torch tensor (or already-numpy) to numpy without an import."""
    if hasattr(tensor_like, "cpu"):
        return tensor_like.cpu().numpy()
    if hasattr(tensor_like, "numpy"):
        return tensor_like.numpy()
    return np.asarray(tensor_like)


def _surface_for_caries(
    caries_bbox: tuple[float, float, float, float],
    tooth_bboxes: list[ToothFinding] | None,
) -> str:
    """Return ``"mesial"`` / ``"distal"`` / ``"unknown"`` for a caries lesion.

    Logic:
    1. If ``tooth_bboxes`` is None or empty → ``"unknown"``.
    2. Find the tooth whose bbox contains the caries-bbox center;
       if none contains it, pick the nearest (by bbox-center
       Euclidean distance).
    3. Compare caries-bbox-center x to parent-tooth-bbox-center x.
       Left → ``"mesial"``, right → ``"distal"`` (the rule layer
       handles patient-laterality conversion downstream).
    """
    if not tooth_bboxes:
        return "unknown"

    cx = 0.5 * (caries_bbox[0] + caries_bbox[2])
    cy = 0.5 * (caries_bbox[1] + caries_bbox[3])

    containing: list[ToothFinding] = []
    for t in tooth_bboxes:
        if t.bbox is None:
            continue
        tx1, ty1, tx2, ty2 = t.bbox
        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            containing.append(t)

    if containing:
        parent = containing[0]
    else:
        # Nearest by bbox-center distance among teeth with a bbox.
        candidates = [t for t in tooth_bboxes if t.bbox is not None]
        if not candidates:
            return "unknown"
        parent = min(
            candidates,
            key=lambda t: _bbox_center_distance((cx, cy), t.bbox),  # type: ignore[arg-type]
        )

    assert parent.bbox is not None  # narrowed above
    tcx = 0.5 * (parent.bbox[0] + parent.bbox[2])
    return "mesial" if cx < tcx else "distal"


def _bbox_center_distance(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> float:
    bcx = 0.5 * (bbox[0] + bbox[2])
    bcy = 0.5 * (bbox[1] + bbox[3])
    return float(np.hypot(point[0] - bcx, point[1] - bcy))
