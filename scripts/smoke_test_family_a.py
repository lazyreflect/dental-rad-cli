"""Smoke test for the Family A math head against GT supervision.

Builds ground-truth CEJ and bone-crest bands from DenPAR Testing
points (same y-band-clustering supervision construction the polyline
model trains on) and runs Family A on those GT bands. If the math
produces sensible mm distributions per AAP cutoffs, the math head is
sound — and the only remaining question is whether the trained model
matches the GT bands.

This decouples math correctness from model accuracy. Designed to be
runnable BEFORE the polyline model finishes training.

Usage::

    python scripts/smoke_test_family_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dental_rad_cli.data.denpar_adapter import (  # noqa: E402
    _cej_polygons_for_image,
    _bone_polygons_for_image,
    _load_keypoint_json,
    _split_dir,
)
from dental_rad_cli.pipeline.family_a import (  # noqa: E402
    calibrate_px_per_mm,
    per_tooth_family_a,
)


def _polygons_to_mask(
    polygons: list, img_shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize a list of polygons to a binary mask at image resolution."""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 255)
    return mask.astype(bool)


def main() -> int:
    denpar = Path("data/denpar")
    testing = _split_dir(denpar, "Testing")
    images_dir = testing / "Images"

    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    print(f"Smoke-testing Family A on {len(stems)} Testing images.\n")

    all_mm: list[float] = []
    n_teeth_total = 0
    n_teeth_with_at_least_one_site = 0
    stage_counts = {"healthy": 0, "mild": 0, "moderate": 0, "severe": 0}
    n_no_landmarks = 0
    n_bone_above_cej = 0

    for stem in stems:
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
        px_per_mm = calibrate_px_per_mm(bboxes)
        if px_per_mm is None or px_per_mm <= 0:
            continue

        for bbox in bboxes:
            n_teeth_total += 1
            mesial, distal = per_tooth_family_a(
                cej_band, bone_band, bbox, px_per_mm
            )
            site_mms = [
                s.mm_estimate
                for s in (mesial, distal)
                if s.mm_estimate is not None
            ]
            if site_mms:
                n_teeth_with_at_least_one_site += 1
                worst = max(site_mms)
                all_mm.append(worst)
                if worst < 2.0:
                    stage_counts["healthy"] += 1
                elif worst < 4.0:
                    stage_counts["mild"] += 1
                elif worst < 6.0:
                    stage_counts["moderate"] += 1
                else:
                    stage_counts["severe"] += 1
            else:
                if (
                    mesial.reason == "no_landmarks_at_site"
                    and distal.reason == "no_landmarks_at_site"
                ):
                    n_no_landmarks += 1
                elif (
                    mesial.reason == "bone_coronal_to_cej"
                    or distal.reason == "bone_coronal_to_cej"
                ):
                    n_bone_above_cej += 1

    print("=" * 60)
    print("Family A on GT-derived bands — Testing split")
    print("=" * 60)
    print(f"Total teeth seen:               {n_teeth_total}")
    print(f"Teeth with ≥1 measurable site:  {n_teeth_with_at_least_one_site} "
          f"({100*n_teeth_with_at_least_one_site/max(n_teeth_total,1):.1f}%)")
    print(f"Teeth with no landmarks:        {n_no_landmarks}")
    print(f"Teeth flagged bone-above-cej:   {n_bone_above_cej}")
    print()

    if all_mm:
        a = np.array(all_mm)
        print("Worst-site mm distribution (per-tooth):")
        print(f"  median: {np.median(a):.2f} mm")
        print(f"  mean:   {a.mean():.2f} mm")
        print(f"  p10/25/75/90: "
              f"{np.percentile(a,10):.2f} / {np.percentile(a,25):.2f} / "
              f"{np.percentile(a,75):.2f} / {np.percentile(a,90):.2f}")
        print(f"  max:    {a.max():.2f} mm")
        print()
        print("AAP staging breakdown:")
        total = sum(stage_counts.values())
        for stage in ("healthy", "mild", "moderate", "severe"):
            n = stage_counts[stage]
            print(f"  {stage:9s}: {n:4d} ({100*n/max(total,1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
