"""Dissect the bone-mask pipeline for the worst-error dev case (stem 907).

Joseph (2026-05-12) clinically corrected: the middle tooth on 907 has
real severe periodontal bone loss; GT is correct; the model under-
predicts. The handoff's claim that erosion=5 fixed severe-perio under-
prediction is falsified by this image.

This script replicates the per-tooth pipeline for 907 step by step and
saves every intermediate mask as an overlay PNG. We want to see:
  - What the raw bone segmentation predicts on the middle tooth
  - Where the eroded mask edge stops
  - What `bone_on_tooth` (tooth-boundary-ring & eroded_bone) yields
  - Where the final landmark lands vs. the actual deep alveolar crest
  - Where GT places the bone-crest for comparison

Outputs go to output/diagnostics/907-dissection/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dental_rad_cli.analyze import _get_or_create_bundle  # noqa: E402
from dental_rad_cli.analyze import (  # noqa: E402
    _bbox_contains, _polygon_centroid, _rasterize_polygon,
)
from dental_rad_cli.data.denpar_adapter import _split_dir  # noqa: E402
from dental_rad_cli.pipeline.family_a import (  # noqa: E402
    _erode_mask, calibrate_px_per_mm,
)
from benchmark_eval import _bbox_iou, _derive_gt_mm  # noqa: E402


STEM = "907"
OUT = Path("output/diagnostics/907-dissection")
WEIGHTS = Path("weights")
DENPAR = Path("data/denpar")
BONE_EROSION_PX = 5
TOOTH_ERODE_FOR_RING_PX = 5


def _to_numpy(x):
    try:
        return x.cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _extract_polys(yolo_model, rgb, device: str = "cpu",
                   conf: float = 0.5):
    results = yolo_model.predict(rgb, conf=conf, verbose=False, device=device)
    if not results:
        return []
    res0 = results[0]
    masks = getattr(res0, "masks", None)
    if masks is None:
        return []
    xy = getattr(masks, "xy", None)
    if xy is None:
        return []
    out = []
    for poly in xy:
        poly_np = _to_numpy(poly)
        if poly_np.ndim != 2 or poly_np.shape[1] != 2 or len(poly_np) < 3:
            continue
        out.append([(float(p[0]), float(p[1])) for p in poly_np])
    return out


def _save(img: np.ndarray, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / name), img)
    print(f"  wrote {OUT / name}")


def _mask_overlay(base_bgr: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """Blend a binary mask onto a BGR base image."""
    out = base_bgr.copy()
    layer = np.zeros_like(out)
    layer[mask.astype(bool)] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0.0)


def main() -> int:
    testing = _split_dir(DENPAR, "Testing")
    img_path = testing / "Images" / f"{STEM}.jpg"
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"image not found at {img_path}")
        return 1
    H, W = img_bgr.shape[:2]
    print(f"image {STEM}: {W}x{H}")

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bundle = _get_or_create_bundle(WEIGHTS)

    # Run the three segmentation models.
    print("running segmentation_tooth ...")
    tooth_polys = _extract_polys(bundle.get_segmentation_tooth(), rgb,
                                 conf=0.5)
    print(f"  tooth polys: {len(tooth_polys)}")
    print("running segmentation_bone ...")
    bone_polys = _extract_polys(bundle.get_segmentation_bone(), rgb,
                                conf=0.5)
    print(f"  bone polys: {len(bone_polys)}")
    print("running segmentation_cej ...")
    cej_polys = _extract_polys(bundle.get_segmentation_cej(), rgb,
                               conf=0.25)
    print(f"  cej polys: {len(cej_polys)}")

    # Rasterize.
    bone_mask = np.zeros((H, W), dtype=np.uint8)
    for bp in bone_polys:
        bone_mask |= _rasterize_polygon(bp, (H, W))
    bone_mask_bool = bone_mask.astype(bool)
    bone_eroded = _erode_mask(bone_mask_bool, BONE_EROSION_PX)

    cej_band = np.zeros((H, W), dtype=np.uint8)
    for cp in cej_polys:
        cej_band |= _rasterize_polygon(cp, (H, W))
    cej_band_bool = cej_band.astype(bool)

    # GT for 907.
    gt_teeth = _derive_gt_mm(testing, STEM)
    print(f"GT teeth: {len(gt_teeth)}")

    # Run tooth detector for bboxes (the analyze pipeline uses the
    # *detector* output for bboxes, not seg polygons). For dissection
    # purposes we approximate per-tooth bbox from each tooth seg
    # polygon since the actual call needs detections from analyze().
    # Build per-poly bboxes:
    tooth_bboxes_from_seg = []
    for tp in tooth_polys:
        xs = [p[0] for p in tp]
        ys = [p[1] for p in tp]
        tooth_bboxes_from_seg.append(
            (min(xs), min(ys), max(xs), max(ys))
        )
    # Sort left-to-right by bbox x-center for "left / middle / right".
    order = sorted(range(len(tooth_polys)),
                   key=lambda i: (tooth_bboxes_from_seg[i][0]
                                  + tooth_bboxes_from_seg[i][2]) / 2)
    print(f"tooth seg polys sorted L→R by xcenter: {order}")

    # px_per_mm from the seg-derived bboxes.
    px_per_mm = calibrate_px_per_mm(tooth_bboxes_from_seg)
    print(f"px_per_mm: {px_per_mm:.3f}")

    # ===== Global overlays =====

    base = img_bgr.copy()

    # 1. raw bone mask in red, eroded bone mask in deeper red.
    over = _mask_overlay(base, bone_mask_bool, (50, 50, 200), 0.35)
    over = _mask_overlay(over, bone_eroded, (0, 0, 255), 0.5)
    _save(over, "01_bone_mask_raw_vs_eroded5.png")

    # 2. cej band in yellow.
    over = _mask_overlay(base, cej_band_bool, (0, 220, 220), 0.5)
    _save(over, "02_cej_band.png")

    # 3. all three seg masks together (blue=tooth, red=bone, yellow=cej).
    over = base.copy()
    tooth_union = np.zeros((H, W), dtype=bool)
    for tp in tooth_polys:
        tooth_union |= _rasterize_polygon(tp, (H, W)).astype(bool)
    over = _mask_overlay(over, tooth_union, (255, 100, 100), 0.3)
    over = _mask_overlay(over, bone_mask_bool, (0, 0, 200), 0.35)
    over = _mask_overlay(over, cej_band_bool, (0, 220, 220), 0.4)
    _save(over, "03_all_seg_masks.png")

    # ===== Per-tooth dissection =====

    for rank, idx in enumerate(order):
        tp = tooth_polys[idx]
        bx = tooth_bboxes_from_seg[idx]
        tag = ["left", "middle", "right"][rank] if rank < 3 else f"t{rank}"
        print(f"\n--- tooth {tag} (idx={idx}, bbox={tuple(round(c) for c in bx)}) ---")

        tooth_mask = _rasterize_polygon(tp, (H, W)).astype(bool)
        tooth_eroded = _erode_mask(tooth_mask, TOOTH_ERODE_FOR_RING_PX)
        boundary_ring = tooth_mask & ~tooth_eroded

        bone_on_tooth_raw = tooth_mask & bone_mask_bool
        bone_on_tooth_eroded = tooth_mask & bone_eroded
        bone_on_ring_eroded = boundary_ring & bone_eroded
        bone_on_ring_raw = boundary_ring & bone_mask_bool

        cej_on_tooth = tooth_mask & cej_band_bool

        # Stats for narrative.
        def _ys_range(mask):
            ys, _ = np.where(mask)
            if ys.size == 0:
                return None
            return int(ys.min()), int(ys.max()), int(np.median(ys))

        print(f"  tooth y-range: {_ys_range(tooth_mask)}")
        print(f"  cej_on_tooth y-range: {_ys_range(cej_on_tooth)}")
        print(f"  bone_on_tooth_raw y-range: {_ys_range(bone_on_tooth_raw)}")
        print(f"  bone_on_tooth_eroded y-range: "
              f"{_ys_range(bone_on_tooth_eroded)}")
        print(f"  bone_on_ring_eroded y-range (FINAL): "
              f"{_ys_range(bone_on_ring_eroded)}")
        print(f"  bone_on_ring_raw y-range: {_ys_range(bone_on_ring_raw)}")

        # Composite overlay: tooth mask faint, ring distinct, bone_eroded
        # red, bone_on_ring_eroded bright cyan (the final candidates).
        over = base.copy()
        over = _mask_overlay(over, tooth_mask, (200, 150, 150), 0.18)
        over = _mask_overlay(over, boundary_ring, (255, 200, 100), 0.45)
        over = _mask_overlay(over, bone_eroded, (0, 0, 220), 0.45)
        over = _mask_overlay(over, bone_on_ring_eroded, (255, 255, 0), 0.85)
        over = _mask_overlay(over, cej_on_tooth, (0, 220, 220), 0.5)
        # bbox.
        cv2.rectangle(over,
                      (int(bx[0]), int(bx[1])),
                      (int(bx[2]), int(bx[3])),
                      (255, 255, 255), 1)
        # Caption strip.
        h, w = over.shape[:2]
        cap = np.zeros((28, w, 3), dtype=np.uint8)
        cv2.putText(cap,
                    f"tooth={tag} idx={idx}  "
                    f"BLUE=tooth FAINT  ORANGE=ring  RED=bone_eroded "
                    f" YELLOW=bone_on_ring_eroded (FINAL)  CYAN=cej_on_tooth",
                    (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        out = np.vstack([cap, over])
        _save(out, f"04_tooth_{rank}_{tag}_idx{idx}_dissection.png")

    # ===== GT for context =====

    over = base.copy()
    for g in gt_teeth:
        bb = [int(round(c)) for c in g["bbox"]]
        cv2.rectangle(over, (bb[0], bb[1]), (bb[2], bb[3]),
                      (255, 255, 0), 2)
        for cej_pt, bone_y in (
            (g["mesial_cej"], g.get("mesial_bone_y")),
            (g["distal_cej"], g.get("distal_bone_y")),
        ):
            if cej_pt is None or bone_y is None:
                continue
            cv2.circle(over, (int(cej_pt[0]), int(cej_pt[1])), 7,
                       (0, 255, 255), -1)
            cv2.circle(over, (int(cej_pt[0]), int(bone_y)), 7,
                       (0, 165, 255), -1)
            cv2.line(over,
                     (int(cej_pt[0]), int(cej_pt[1])),
                     (int(cej_pt[0]), int(bone_y)),
                     (0, 165, 255), 2)
    _save(over, "05_gt_overlay.png")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
