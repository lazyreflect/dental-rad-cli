"""Diagnose the 24% bone-above-cej rate from the Family A smoke test.

The smoke test on GT-derived bands (scripts/smoke_test_family_a.py)
showed 206/859 teeth (24%) flagged anatomically impossible: predicted
bone-crest centerline at the bbox edge sits coronal to predicted CEJ
centerline. Anatomically bone-crest is always apical to CEJ.

Three plausible causes to distinguish:

1. **Multi-band column ambiguity.** At x = bbox.x1, both arches'
   bands may be present (BW with upper + lower arches). Median y of
   non-zero pixels lands between the two bands or on the wrong one.

2. **Interproximal column outside tooth.** bbox.x1 may fall inside
   the interproximal contact rather than at the mesial CEJ position.
   Bone-crest dips coronal between teeth (interdental septum is
   apical to alveolar crest at mid-tooth). So the LOCAL bone-crest
   at x=bbox.x1 may actually be coronal to the current tooth's CEJ.

3. **Buffered polygon corner artifacts.** Sharp turns in the polyline
   produce polygon corners after buffering that extend in unexpected
   directions, leaving pixels at y values that don't represent the
   polyline's anatomical line.

This script renders per-tooth crops of flagged cases with both bands +
the bbox + the centerline pickup positions overlaid, so a human (or
the next session) can determine which cause dominates. Reports the
first 20 cases to output/diagnostics/bone-above-cej-{tooth_idx}.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dental_rad_cli.data.denpar_adapter import (  # noqa: E402
    _bone_polygons_for_image,
    _cej_polygons_for_image,
    _load_keypoint_json,
    _split_dir,
)
from dental_rad_cli.pipeline.family_a import (  # noqa: E402
    band_centerline_y_at_x,
    calibrate_px_per_mm,
)


def _polygons_to_mask(polygons: list, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def _render_diagnostic(
    img: np.ndarray,
    cej_band: np.ndarray,
    bone_band: np.ndarray,
    bbox: tuple[float, float, float, float],
    cej_m_y: float | None,
    cej_d_y: float | None,
    bone_m_y: float | None,
    bone_d_y: float | None,
    out_path: Path,
) -> None:
    """Render a per-tooth diagnostic image with bands + bbox + centerline picks."""
    overlay = img.copy()

    # Overlay CEJ band (green) + bone band (red) at semi-transparency.
    cej_layer = overlay.copy()
    cej_layer[cej_band] = (0, 255, 0)
    bone_layer = overlay.copy()
    bone_layer[bone_band] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.5, cej_layer, 0.5, 0)
    overlay = cv2.addWeighted(overlay, 0.7, bone_layer, 0.3, 0)

    # Draw bbox.
    x1, y1, x2, y2 = [int(round(c)) for c in bbox]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Mark centerline picks: CEJ as yellow circle, bone as cyan circle.
    for x, cej_y, bone_y, label in (
        (x1, cej_m_y, bone_m_y, "M"),
        (x2, cej_d_y, bone_d_y, "D"),
    ):
        if cej_y is not None:
            cv2.circle(overlay, (x, int(round(cej_y))), 8, (0, 255, 255), -1)
            cv2.circle(overlay, (x, int(round(cej_y))), 8, (0, 0, 0), 1)
        if bone_y is not None:
            cv2.circle(overlay, (x, int(round(bone_y))), 8, (255, 255, 0), -1)
            cv2.circle(overlay, (x, int(round(bone_y))), 8, (0, 0, 0), 1)
        cv2.putText(
            overlay, label, (x - 10, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    # Annotate with the offending diff numbers.
    annotations = []
    if cej_m_y is not None and bone_m_y is not None:
        annotations.append(f"M: cej_y={cej_m_y:.0f} bone_y={bone_m_y:.0f} diff={bone_m_y - cej_m_y:+.0f}px")
    if cej_d_y is not None and bone_d_y is not None:
        annotations.append(f"D: cej_y={cej_d_y:.0f} bone_y={bone_d_y:.0f} diff={bone_d_y - cej_d_y:+.0f}px")
    for i, line in enumerate(annotations):
        cv2.putText(
            overlay, line, (10, 30 + 25 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            overlay, line, (10, 30 + 25 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

    # Crop to bbox + 100 px padding for readability.
    pad = 100
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(overlay.shape[1], x2 + pad)
    cy2 = min(overlay.shape[0], y2 + pad)
    crop = overlay[cy1:cy2, cx1:cx2]

    cv2.imwrite(str(out_path), crop)


def main() -> int:
    denpar = Path("data/denpar")
    testing = _split_dir(denpar, "Testing")
    images_dir = testing / "Images"
    out_dir = Path("output/diagnostics/bone-above-cej")
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))

    n_flagged = 0
    n_rendered = 0
    max_renders = 20
    causes: dict[str, int] = {
        "centerline_outside_bbox_y_range": 0,
        "mesial_only": 0,
        "distal_only": 0,
        "both_sides": 0,
        "other": 0,
    }

    for stem in stems:
        if n_rendered >= max_renders:
            break
        img_path = images_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cej_polys = _cej_polygons_for_image(testing, stem)
        bone_polys = _bone_polygons_for_image(testing, stem)
        if not cej_polys or not bone_polys:
            continue
        cej_band = _polygons_to_mask(cej_polys, (h, w))
        bone_band = _polygons_to_mask(bone_polys, (h, w))

        kp = _load_keypoint_json(testing, stem)
        if kp is None:
            continue
        bboxes = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b in (kp.get("bboxes") or [])
        ]
        if not bboxes:
            continue

        for tooth_idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            cej_m_y = band_centerline_y_at_x(cej_band, x1)
            cej_d_y = band_centerline_y_at_x(cej_band, x2)
            bone_m_y = band_centerline_y_at_x(bone_band, x1)
            bone_d_y = band_centerline_y_at_x(bone_band, x2)

            # Detect a "bone above cej" case: diff < -3 px at either site.
            m_flag = (cej_m_y is not None and bone_m_y is not None
                      and bone_m_y - cej_m_y < -3)
            d_flag = (cej_d_y is not None and bone_d_y is not None
                      and bone_d_y - cej_d_y < -3)
            if not (m_flag or d_flag):
                continue
            n_flagged += 1

            # Classify cause.
            if m_flag and d_flag:
                causes["both_sides"] += 1
            elif m_flag:
                causes["mesial_only"] += 1
            elif d_flag:
                causes["distal_only"] += 1
            # Check if centerline pick is outside the bbox y range.
            for cej_y, bone_y in (
                (cej_m_y, bone_m_y), (cej_d_y, bone_d_y)
            ):
                if cej_y is None or bone_y is None:
                    continue
                if not (y1 <= cej_y <= y2) or not (y1 <= bone_y <= y2):
                    causes["centerline_outside_bbox_y_range"] += 1
                    break

            if n_rendered < max_renders:
                out_path = out_dir / f"{stem}-tooth{tooth_idx}.png"
                _render_diagnostic(
                    img, cej_band, bone_band, bbox,
                    cej_m_y, cej_d_y, bone_m_y, bone_d_y,
                    out_path,
                )
                n_rendered += 1

    print(f"Rendered {n_rendered} diagnostic crops to {out_dir}")
    print()
    print("Cause distribution (across first ~20 flagged cases):")
    for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"  {cause:38s}: {n}")
    print()
    print("Open the rendered PNGs to spot-check which cause dominates.")
    print("Key (per crop):")
    print("  green band  = CEJ supervision")
    print("  red band    = bone-crest supervision")
    print("  yellow bbox = tooth")
    print("  yellow dots = CEJ centerline pick at bbox.x1 / bbox.x2")
    print("  cyan dots   = bone centerline pick at bbox.x1 / bbox.x2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
