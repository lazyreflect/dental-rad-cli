"""Dissect bone-mask extent at every severe-bucket dev site.

BR9: the 2026-05-12 evening "CEJ-x sampling" experiment (BRneg-1)
failed to fix severe-perio under-prediction. Conclusion: the 907-style
adjacent-tooth contamination isn't the representative failure mode.
Next hypothesis: the bone *segmentation mask itself* doesn't extend
down to the deep alveolar crest in severe-perio cases. If true, no
landmark-selection rule can recover; the fix is training-side
(labels/loss/data). If false (mask reaches deep but rule misses),
algorithm-side fix is still possible.

This script:
  1. Iterates dev split
  2. For each GT tooth with gt_mm >= 6 mm (severe + extreme buckets)
  3. Runs inference and inspects the bone-mask distribution AT the
     CEJ x-coordinate (mesial and distal)
  4. Compares bone-mask y-extent vs. GT bone-crest y
  5. Categorizes each site:
       - mask_reaches_deep   : bone-mask y-max within tolerance of GT bone-y
       - mask_short          : bone-mask y-max well coronal to GT bone-y
       - mask_bimodal        : bone-mask has both shallow + deep clusters
       - no_bone_at_cej_x    : no bone-mask pixels in the column
  6. Writes a markdown summary + per-site overlay PNGs

Output:
  output/diagnostics/severe-sites/SUMMARY.md
  output/diagnostics/severe-sites/<stem>__t<idx>.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dental_rad_cli.analyze import _get_or_create_bundle  # noqa: E402
from dental_rad_cli.analyze import _rasterize_polygon  # noqa: E402
from dental_rad_cli.data.denpar_adapter import _split_dir  # noqa: E402
from dental_rad_cli.pipeline.family_a import (  # noqa: E402
    _BONE_EROSION_PX_DEFAULT, _erode_mask,
)
from benchmark_eval import _derive_gt_mm, _load_split  # noqa: E402


GT_SEVERE_MIN_MM = 6.0  # severe + extreme
OUT = Path("output/diagnostics/severe-sites")
WEIGHTS = Path("weights")
DENPAR = Path("data/denpar")
SPLITS = Path("splits")
BONE_EROSION_PX = _BONE_EROSION_PX_DEFAULT  # 5

# Y tolerance for "mask reaches deep": within this many px of GT bone-y
# we call the mask deep-enough.
MASK_REACHES_TOL_PX = 30  # ~1 mm at typical px_per_mm

# X tolerance window for sampling bone mask at CEJ x.
CEJ_X_TOL_PX = 10


def _to_numpy(x):
    try:
        return x.cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def _extract_polys(yolo_model, rgb, conf: float = 0.5):
    results = yolo_model.predict(rgb, conf=conf, verbose=False, device="cpu")
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


def _bone_mask_y_at_x(bone_mask: np.ndarray, x_center: float,
                     tol_px: int) -> np.ndarray:
    """Return array of y-values where bone_mask is True within ±tol_px
    of column x_center."""
    H, W = bone_mask.shape
    xlo = max(0, int(round(x_center - tol_px)))
    xhi = min(W, int(round(x_center + tol_px + 1)))
    if xhi <= xlo:
        return np.array([], dtype=int)
    sub = bone_mask[:, xlo:xhi]
    ys, _ = np.where(sub)
    return ys


def _categorize(mask_ys: np.ndarray, gt_bone_y: float, cej_y: float,
                apical_sign: float) -> tuple[str, dict]:
    """Categorize the bone-mask y-distribution at CEJ x vs GT bone-y."""
    stats = {
        "n_pixels": int(mask_ys.size),
        "y_min": None, "y_max": None, "y_median": None,
        "gt_bone_y": float(gt_bone_y),
        "cej_y": float(cej_y),
        "apical_sign": float(apical_sign),
    }
    if mask_ys.size == 0:
        return "no_bone_at_cej_x", stats
    y_min = float(mask_ys.min())
    y_max = float(mask_ys.max())
    y_median = float(np.median(mask_ys))
    stats["y_min"] = y_min
    stats["y_max"] = y_max
    stats["y_median"] = y_median

    # "Reaches deep" = the mask extends in the apical direction far
    # enough that it gets within MASK_REACHES_TOL_PX of GT bone-y.
    if apical_sign > 0:
        # apical = larger y; mask reaches deep means y_max >= gt_bone_y - tol
        mask_apical_extent = y_max
        reaches = mask_apical_extent >= (gt_bone_y - MASK_REACHES_TOL_PX)
    else:
        # apical = smaller y; mask reaches deep means y_min <= gt_bone_y + tol
        mask_apical_extent = y_min
        reaches = mask_apical_extent <= (gt_bone_y + MASK_REACHES_TOL_PX)
    stats["mask_apical_extent"] = mask_apical_extent
    stats["reaches"] = bool(reaches)

    # Bimodality: are there both shallow and deep pixels relative to a
    # midpoint between CEJ and GT bone? Cheap proxy: split mask into
    # shallow-half and deep-half by midpoint y, check both are non-empty.
    mid_y = (cej_y + gt_bone_y) / 2.0
    if apical_sign > 0:
        shallow = mask_ys[mask_ys < mid_y]
        deep = mask_ys[mask_ys >= mid_y]
    else:
        shallow = mask_ys[mask_ys > mid_y]
        deep = mask_ys[mask_ys <= mid_y]
    stats["n_shallow"] = int(shallow.size)
    stats["n_deep"] = int(deep.size)

    if reaches and shallow.size > 0 and deep.size > 0:
        return "mask_bimodal", stats
    if reaches:
        return "mask_reaches_deep", stats
    return "mask_short", stats


def _render_overlay(img_bgr: np.ndarray, bone_mask: np.ndarray,
                    bone_eroded: np.ndarray,
                    bbox: tuple, cej_x: float, cej_y: float,
                    gt_bone_y: float,
                    cat: str) -> np.ndarray:
    """Render per-site overlay: bone mask + GT markers + CEJ x line."""
    over = img_bgr.copy()
    # Bone mask raw (light red) + eroded (deeper red).
    layer = np.zeros_like(over)
    layer[bone_mask] = (50, 50, 200)
    over = cv2.addWeighted(over, 1.0, layer, 0.30, 0.0)
    layer = np.zeros_like(over)
    layer[bone_eroded] = (0, 0, 220)
    over = cv2.addWeighted(over, 1.0, layer, 0.45, 0.0)
    # bbox.
    cv2.rectangle(over,
                  (int(bbox[0]), int(bbox[1])),
                  (int(bbox[2]), int(bbox[3])),
                  (255, 255, 0), 1)
    # CEJ x column highlighted (vertical thin line).
    H = over.shape[0]
    cv2.line(over,
             (int(cej_x), 0), (int(cej_x), H),
             (255, 255, 255), 1)
    # GT CEJ (yellow) + GT bone (orange) at CEJ x.
    cv2.circle(over, (int(cej_x), int(cej_y)), 7, (0, 255, 255), -1)
    cv2.circle(over, (int(cej_x), int(cej_y)), 7, (0, 0, 0), 1)
    cv2.circle(over, (int(cej_x), int(gt_bone_y)), 7, (0, 165, 255), -1)
    cv2.circle(over, (int(cej_x), int(gt_bone_y)), 7, (0, 0, 0), 1)
    cv2.line(over, (int(cej_x), int(cej_y)),
             (int(cej_x), int(gt_bone_y)), (0, 165, 255), 2)
    # Caption.
    h, w = over.shape[:2]
    cap = np.zeros((32, w, 3), dtype=np.uint8)
    cv2.putText(cap, f"category: {cat}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([cap, over])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = _get_or_create_bundle(WEIGHTS)

    testing = _split_dir(DENPAR, "Testing")
    images_dir = testing / "Images"
    stems = sorted(_load_split(SPLITS, "dev"))
    print(f"scanning dev split ({len(stems)} images) for severe sites...")

    findings: list[dict] = []
    bbox_iou_threshold = 0.3

    import time
    t0 = time.perf_counter()

    for idx, stem in enumerate(stems):
        if idx % 25 == 0:
            print(f"  {idx}/{len(stems)}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
        gt_teeth = _derive_gt_mm(testing, stem)
        if not gt_teeth:
            continue
        # Pre-filter: does ANY tooth have a severe site?
        has_severe = any(
            ((g.get("mesial_mm") or 0) >= GT_SEVERE_MIN_MM
             or (g.get("distal_mm") or 0) >= GT_SEVERE_MIN_MM)
            for g in gt_teeth
        )
        if not has_severe:
            continue

        img_path = images_dir / f"{stem}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run bone segmentation only (we don't need cej or tooth for this).
        bone_polys = _extract_polys(bundle.get_segmentation_bone(), rgb,
                                   conf=0.5)
        bone_mask = np.zeros((H, W), dtype=np.uint8)
        for bp in bone_polys:
            bone_mask |= _rasterize_polygon(bp, (H, W))
        bone_mask_bool = bone_mask.astype(bool)
        bone_eroded = _erode_mask(bone_mask_bool, BONE_EROSION_PX)

        for gt_idx, g in enumerate(gt_teeth):
            bb = g["bbox"]
            bbox_cy = 0.5 * (bb[1] + bb[3])
            cej_y_avg = 0.5 * (g["mesial_cej"][1] + g["distal_cej"][1])
            # mandibular (cej above bbox center → apical = +y)
            apical_sign = 1.0 if cej_y_avg < bbox_cy else -1.0

            for surface, cej_pt, gt_mm, gt_bone_y in (
                ("mesial", g["mesial_cej"],
                 g.get("mesial_mm"), g.get("mesial_bone_y")),
                ("distal", g["distal_cej"],
                 g.get("distal_mm"), g.get("distal_bone_y")),
            ):
                if gt_mm is None or gt_bone_y is None:
                    continue
                if gt_mm < GT_SEVERE_MIN_MM:
                    continue
                cej_x, cej_y = float(cej_pt[0]), float(cej_pt[1])
                mask_ys = _bone_mask_y_at_x(
                    bone_mask_bool, cej_x, CEJ_X_TOL_PX
                )
                eroded_ys = _bone_mask_y_at_x(
                    bone_eroded, cej_x, CEJ_X_TOL_PX
                )
                cat, stats = _categorize(eroded_ys, gt_bone_y, cej_y,
                                          apical_sign)
                findings.append({
                    "stem": stem,
                    "gt_idx": gt_idx,
                    "surface": surface,
                    "gt_mm": float(gt_mm),
                    "cej_x": cej_x, "cej_y": cej_y,
                    "gt_bone_y": float(gt_bone_y),
                    "bbox": [float(c) for c in bb],
                    "category": cat,
                    **stats,
                    "raw_mask_n_pixels_at_x": int(mask_ys.size),
                })

                overlay = _render_overlay(
                    img_bgr, bone_mask_bool, bone_eroded,
                    bb, cej_x, cej_y, gt_bone_y, cat,
                )
                fname = f"{stem}__t{gt_idx}_{surface}_{cat}.png"
                cv2.imwrite(str(OUT / fname), overlay)

    elapsed = time.perf_counter() - t0
    print(f"\nfound {len(findings)} severe sites in dev "
          f"({elapsed:.0f}s)")

    if not findings:
        print("no severe sites — nothing to summarize.")
        return 0

    # Summarize.
    from collections import Counter
    cats = Counter(f["category"] for f in findings)
    print("\ncategory counts:")
    for c, n in cats.most_common():
        print(f"  {c:<22} {n}")

    # Per-category MAE-relevant info: did the algorithm correctly capture
    # the deep bone level? We don't have pred_mm here (we'd need to run
    # full analyze), but the category tells us if the bone mask was even
    # *able* to support a correct prediction.

    # Markdown summary.
    lines = [
        "# Severe-bucket dev sites — bone-mask extent dissection (BR9)",
        "",
        f"Filtered: GT bone-loss >= {GT_SEVERE_MIN_MM} mm. "
        f"n = {len(findings)} sites across "
        f"{len({f['stem'] for f in findings})} stems.",
        "",
        f"Tolerance for 'reaches deep': ±{MASK_REACHES_TOL_PX} px of "
        "GT bone-y, sampled within "
        f"±{CEJ_X_TOL_PX} px of the CEJ x-column. Eroded mask "
        f"({BONE_EROSION_PX} px) used for category decision.",
        "",
        "## Category breakdown",
        "",
        "| category | n | meaning |",
        "|---|---|---|",
        f"| mask_reaches_deep   | {cats.get('mask_reaches_deep', 0)} | "
        "bone-mask extends within tol of GT bone-y → algorithm-side fix "
        "possible (rule mis-picks within an adequate mask) |",
        f"| mask_bimodal        | {cats.get('mask_bimodal', 0)} | "
        "mask reaches deep AND has coronal pixels → 907-style "
        "contamination; algorithm-side fix needed |",
        f"| mask_short          | {cats.get('mask_short', 0)} | "
        "bone-mask max-apical extent stays well coronal to GT bone-y → "
        "training-side issue (mask itself can't support correct landmark) |",
        f"| no_bone_at_cej_x    | {cats.get('no_bone_at_cej_x', 0)} | "
        "no bone-mask pixels within ±10 px of CEJ x → segmentation "
        "didn't fire near the CEJ column |",
        "",
        "## Headline interpretation",
        "",
    ]
    short = cats.get('mask_short', 0)
    deep = cats.get('mask_reaches_deep', 0)
    bimodal = cats.get('mask_bimodal', 0)
    none = cats.get('no_bone_at_cej_x', 0)
    n = len(findings)
    pct_short = 100.0 * short / n if n else 0
    pct_deep = 100.0 * (deep + bimodal) / n if n else 0
    if pct_short >= 60:
        verdict = (
            "**Bone segmentation IS the bottleneck.** Most severe sites "
            "have masks that don't reach the deep alveolar crest. "
            "Algorithm-side fixes are bounded; the path forward is "
            "training-data / loss / labels for the bone segmentation "
            "model — OR accept that severe-perio cases are at the "
            "radiographic-detection ceiling for the current training "
            "regime."
        )
    elif pct_deep >= 60:
        verdict = (
            "**Algorithm-side fix space exists.** Most severe sites "
            "have bone masks that *do* extend to the deep crest; the "
            "landmark-selection rule is choosing the wrong y. The "
            "BRneg-1 CEJ-x sampling attempt didn't deliver because "
            "(hypothesis) the apical-extent isn't necessarily at the "
            "CEJ x specifically — may need a different rule like "
            "'most apical bone-on-tooth pixel within X% of the CEJ x', "
            "or weighted median, or training a small head."
        )
    else:
        verdict = (
            "**Mixed.** Some severe sites have short masks (training-"
            "side), others have deep masks with wrong-pick (algorithm-"
            "side). Look at per-site details for next move."
        )
    lines.append(verdict)
    lines += [
        "",
        f"- mask_short {short}/{n} ({pct_short:.0f}%)",
        f"- mask_reaches_deep + mask_bimodal {deep+bimodal}/{n} "
        f"({pct_deep:.0f}%)",
        f"- no_bone_at_cej_x {none}/{n} ({100.0*none/n:.0f}%)" if n else "",
        "",
        "## Per-site detail",
        "",
        "| stem | surface | gt_mm | cej_y | gt_bone_y | mask y_min | "
        "y_median | y_max | apical_extent | category |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in sorted(findings, key=lambda r: (-r["gt_mm"], r["stem"])):
        lines.append(
            f"| {f['stem']} | {f['surface']} | {f['gt_mm']:.2f} | "
            f"{f['cej_y']:.0f} | {f['gt_bone_y']:.0f} | "
            f"{f.get('y_min') if f.get('y_min') is None else int(f['y_min'])} | "
            f"{f.get('y_median') if f.get('y_median') is None else int(f['y_median'])} | "
            f"{f.get('y_max') if f.get('y_max') is None else int(f['y_max'])} | "
            f"{f.get('mask_apical_extent') if f.get('mask_apical_extent') is None else int(f['mask_apical_extent'])} | "
            f"{f['category']} |"
        )

    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
