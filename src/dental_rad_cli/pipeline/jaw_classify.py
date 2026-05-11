"""Maxillary vs mandibular IOPA classifier.

Pure function. No I/O.

Methodology brief §3.1: per-tooth, compare CEJ y-coordinate to apex
y-coordinate. In standard radiograph orientation, smaller y = higher in
the image plane.

- If CEJ y < apex y (CEJ above apex in image space, i.e. apex points
  down): the tooth's crown is up and the root points down — that's the
  geometry of a **maxillary** tooth (apex at the top of the screen, CEJ
  below — wait, see below).
- If CEJ y > apex y (CEJ below apex): mandibular.

The brief's text reads: "If CEJ is above (smaller y) the apex →
mandibular; else maxillary." This is consistent with conventional IOPA
orientation where mandibular teeth are imaged with the crown at the
**bottom** of the film (apex toward the top of the image → smaller y),
and maxillary teeth with the crown at the **top** (apex toward the
bottom → larger y). We follow the brief literally.

The image-level decision is a majority vote across teeth; with missing
or zero coordinates the brief's silent default is **mandibular** — we
preserve that default and document the fallback.
"""

from __future__ import annotations

from typing import List

from dental_rad_cli.schema import Jaw, ToothWithKeypoints


def _per_tooth_label(t: ToothWithKeypoints) -> Jaw:
    """Default: mandibular when inputs are missing (brief §3.1)."""
    if t.cej_y is None or t.apex_y is None:
        return "mandibular"
    # Brief: CEJ above (smaller y) apex → mandibular; else maxillary.
    if t.cej_y < t.apex_y:
        return "mandibular"
    return "maxillary"


def classify_jaw(teeth: List[ToothWithKeypoints]) -> Jaw:
    """Return ``"maxillary"`` or ``"mandibular"`` for the IOPA.

    Empty input → ``"mandibular"`` (silent fallback per brief §3.1 +
    reimplementation gotcha #12). Tied majority votes also fall back to
    ``"mandibular"`` for symmetry with the missing-data fallback.
    """
    if not teeth:
        return "mandibular"

    maxillary = 0
    mandibular = 0
    for t in teeth:
        label = _per_tooth_label(t)
        if label == "maxillary":
            maxillary += 1
        else:
            mandibular += 1

    if maxillary > mandibular:
        return "maxillary"
    return "mandibular"
