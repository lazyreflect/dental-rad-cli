"""Benchmark eval — measures mm-MAE on DenPAR test split for direct
comparison to published commercial + academic CEJ→bone-crest benchmarks.

Anchors:
  Overjet K210187          0.300 mm (pooled BW+PA, press citation)
  Adravision K232440       0.434 mm BW / 0.504 mm PA  (FDA filing)
  Pearl Second Opinion BLE 0.860 mm BW / 0.450 mm PA  (FDA filing)
  VideaHealth K223296      < 1.5 mm BW+PA            (FDA filing)
  AlGhaihab/Denti.AI 2025  0.499 mm BW               (peer-reviewed)

The metric: for each GT tooth on the held-out DenPAR Testing split
where the ground-truth labels provide both CEJ points (mesial +
distal) AND a bone polyline that covers those x positions, derive
GT mm CEJ→bone-crest at each interproximal site. Run the pipeline
on the same image, collect predicted mm at the matched tooth's
sites. MAE = mean(|gt_mm − pred_mm|) over all matched sites.

GT derivation
-------------

GT mesial CEJ = leftmost CEJ_Points entry assigned to the tooth bbox.
GT distal CEJ = rightmost CEJ_Points entry assigned to the tooth bbox.
GT mesial bone-crest = bone-polyline y interpolated at x = GT
    mesial CEJ x. Uses the most-coronal y when multiple polylines
    cover that x (matches `_bone_crest_for_bbox` in the adapter).
GT distal bone-crest = same at GT distal CEJ x.

GT mm = |bone_y − cej_y| / px_per_mm. Vertical projection (same
y-difference assumption as our pipeline) so the metric isolates
landmark accuracy, not long-axis-projection methodology.

px_per_mm is per-image: median GT bbox height / 21 mm anchor.
Identical conversion is used on both GT and predicted sides so the
calibration error cancels.

Predicted mm
------------

Runs the full pipeline via ``analyze()``. For each predicted tooth
matching a GT tooth (bbox IoU > 0.3), pulls
``BoneLossSite.mm_estimate`` at mesial and distal sites. Skips
sites where Family A emitted None (model said "I don't know") —
these are excluded from MAE but counted toward coverage.

Outputs
-------

  results/benchmark-eval-{timestamp}.json — full per-site records.
  Stdout headline:
    mm-MAE mean / median / p90
    coverage (sites with predicted measurement / sites with GT)
    CEJ-band IoU (predicted vs GT-derived band)
    comparison vs published benchmarks

Split discipline
----------------

As of 2026-05-12, DenPAR Testing is split into:
  - dev (150 images, default for --split):  iterative architecture work
  - held-out (50 images, --split=held-out): one-shot final eval only

The default ``--split=dev`` enforces honest measurement during
iteration. ``--split=held-out`` is the end-of-development surface;
running it requires ``--confirm-held-out-touch`` AND a logged entry in
``splits/HELD_OUT_TOUCHES.md``.

The legacy ``--split=all`` mode (full 200) prints a warning and is kept
only for reproducing pre-2026-05-12 numbers.

Usage::

    python scripts/benchmark_eval.py --weights weights/             # dev
    python scripts/benchmark_eval.py --split=all                    # legacy
    python scripts/benchmark_eval.py --split=held-out \\
        --confirm-held-out-touch                                    # final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dental_rad_cli.analyze import _get_or_create_bundle, analyze  # noqa: E402
from dental_rad_cli.data.denpar_adapter import (  # noqa: E402
    _assign_points_to_bboxes,
    _bone_crest_for_bbox,
    _load_bonelevel_json,
    _load_keypoint_json,
    _sort_pair_left_right,
    _split_dir,
)
from dental_rad_cli.pipeline.family_a import (  # noqa: E402
    _MAX_PLAUSIBLE_MM,
    calibrate_px_per_mm,
)


# Benchmark anchors for the headline comparison line.
_BENCHMARKS_MM_MAE = {
    "Overjet K210187 (pooled BW+PA)": 0.300,
    "Adravision K232440 (BW)": 0.434,
    "Adravision K232440 (PA)": 0.504,
    "Pearl K243230 (PA)": 0.450,
    "Pearl K243230 (BW)": 0.860,
    "AlGhaihab/Denti.AI 2025 (BW)": 0.499,
    "VideaHealth K223296 (BW+PA)": 1.500,
}


@dataclass
class SiteError:
    stem: str
    fdi_idx: int
    surface: str  # "mesial" | "distal"
    gt_mm: float
    pred_mm: Optional[float]
    abs_err_mm: Optional[float]
    bbox_iou: float
    bbox: Optional[tuple]  # GT tooth bbox (x1, y1, x2, y2) in image coords
    image_w: Optional[int]
    image_h: Optional[int]


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _derive_gt_mm(testing_root: Path, stem: str) -> list[dict]:
    """For each GT tooth with both CEJ pts + bone polyline coverage,
    derive GT mm at mesial and distal sites.

    Returns list of {bbox, mesial_mm, distal_mm, mesial_cej, distal_cej,
                     mesial_bone, distal_bone, px_per_mm}.
    Empty list if image lacks the labels.
    """
    kp = _load_keypoint_json(testing_root, stem)
    if kp is None:
        return []
    bboxes_raw = kp.get("bboxes") or []
    if not bboxes_raw:
        return []
    bboxes = [tuple(float(c) for c in b) for b in bboxes_raw]
    cej_pts = [
        (float(p[0]), float(p[1])) for p in (kp.get("CEJ_Points") or [])
    ]

    # Bone polylines (alveolar bone-crest tracing).
    bl = _load_bonelevel_json(testing_root, stem) or {}
    bone_lines_raw = bl.get("Bone_Lines") or []
    bone_lines: list[list[tuple[float, float]]] = [
        [(float(p[0]), float(p[1])) for p in line] for line in bone_lines_raw
    ]
    if not bone_lines:
        return []

    cej_by_tooth = _assign_points_to_bboxes(cej_pts, bboxes)

    # per-image px_per_mm using median bbox height / 21 mm anchor.
    px_per_mm = calibrate_px_per_mm(bboxes)
    if px_per_mm is None or px_per_mm <= 0:
        return []

    out: list[dict] = []
    for i, bb in enumerate(bboxes):
        mesial_cej, distal_cej = _sort_pair_left_right(cej_by_tooth[i])
        if mesial_cej is None or distal_cej is None:
            continue

        # GT bone-crest y at each CEJ x via existing bbox-anchored
        # polyline interpolator. _bone_crest_for_bbox returns
        # (mesial, distal) at bbox.x1 / bbox.x2 — close to but not
        # exactly the CEJ x. We need finer control: interpolate at the
        # actual CEJ point's x.
        def _bone_y_at_x(target_x: float) -> Optional[float]:
            candidates: list[float] = []
            for line in bone_lines:
                for (xa, ya), (xb, yb) in zip(line, line[1:]):
                    lo, hi = (xa, xb) if xa <= xb else (xb, xa)
                    if lo <= target_x <= hi:
                        if xa == xb:
                            candidates.append(float(min(ya, yb)))
                        else:
                            t = (target_x - xa) / (xb - xa)
                            candidates.append(float(ya + t * (yb - ya)))
                        break  # one per polyline
            if not candidates:
                return None
            return min(candidates)  # most coronal = smallest y

        m_bone_y = _bone_y_at_x(mesial_cej[0])
        d_bone_y = _bone_y_at_x(distal_cej[0])
        if m_bone_y is None and d_bone_y is None:
            continue

        # Determine orientation (mandibular vs maxillary) from CEJ
        # position vs bbox center; bone is APICAL, so should be on the
        # opposite side of CEJ from the crown.
        bbox_cy = 0.5 * (bb[1] + bb[3])
        cej_y_avg = 0.5 * (mesial_cej[1] + distal_cej[1])
        # If CEJ is above bbox center → mandibular (crown at top,
        # apical = +y, bone should have larger y than CEJ)
        # else maxillary (apical = -y, bone should have smaller y).

        def _mm(cej_pt, bone_y) -> Optional[float]:
            if cej_pt is None or bone_y is None:
                return None
            mm = abs(bone_y - cej_pt[1]) / px_per_mm
            if mm > _MAX_PLAUSIBLE_MM:
                return None
            return mm

        m_mm = _mm(mesial_cej, m_bone_y)
        d_mm = _mm(distal_cej, d_bone_y)
        if m_mm is None and d_mm is None:
            continue
        out.append({
            "bbox": bb,
            "fdi_idx": i,
            "mesial_cej": mesial_cej,
            "distal_cej": distal_cej,
            "mesial_bone_y": m_bone_y,
            "distal_bone_y": d_bone_y,
            "mesial_mm": m_mm,
            "distal_mm": d_mm,
            "px_per_mm": float(px_per_mm),
        })
    return out


def _percentile_stats(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0, "mean": None, "median": None,
            "p10": None, "p25": None, "p75": None, "p90": None, "max": None,
        }
    a = np.asarray(values)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def _load_split(splits_dir: Path, split: str) -> Optional[list[str]]:
    """Return ordered list of stems for ``split``, or None for the
    legacy 'all' mode (use full image dir glob)."""
    if split == "all":
        return None
    fname = {"dev": "denpar_dev.txt",
             "held-out": "denpar_held_out.txt"}[split]
    path = splits_dir / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file {path} not found. Run "
            "`scripts/lock_held_out_split.py` first."
        )
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights"))
    ap.add_argument("--denpar-root", type=Path, default=Path("data/denpar"))
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument(
        "--split", choices=["dev", "held-out", "all"], default="dev",
        help=("dev (default, 150 imgs) for iteration; held-out (50) for "
              "final eval ONLY — requires --confirm-held-out-touch; "
              "all (200, legacy) emits a warning."),
    )
    ap.add_argument(
        "--confirm-held-out-touch", action="store_true",
        help="Required when --split=held-out. Log entry in "
             "splits/HELD_OUT_TOUCHES.md must be written FIRST.",
    )
    ap.add_argument(
        "--out-json", type=Path, default=None,
        help="Output JSON path (defaults to "
             "output/training-evidence/benchmark-eval-<split>-<ts>.json).",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N images of the split (0 = all). Diagnostic.",
    )
    ap.add_argument(
        "--landmark-rule", default=None,
        help="Override the family_a bone-landmark rule. Sets "
             "DENTAL_RAD_LANDMARK_RULE env var. Valid rules: "
             "min_y_half (default) median_y_half max_y_half "
             "min_y_at_cej_x median_y_at_cej_x max_y_at_cej_x wide_aware",
    )
    args = ap.parse_args()

    if args.landmark_rule is not None:
        import os
        os.environ["DENTAL_RAD_LANDMARK_RULE"] = args.landmark_rule
        print(f"NOTE: landmark rule override = {args.landmark_rule}")

    if args.split == "held-out" and not args.confirm_held_out_touch:
        print(
            "ERROR: --split=held-out requires --confirm-held-out-touch.\n"
            "Held-out is one-shot. Before re-running with the flag, append\n"
            "an entry to splits/HELD_OUT_TOUCHES.md explaining what number\n"
            "is being read and why this is a justified end-of-development\n"
            "touch (not iterative tuning).",
            file=sys.stderr,
        )
        return 2

    if args.split == "all":
        print(
            "WARN: --split=all uses the full 200-image Testing set, which\n"
            "is the historical pre-lock surface. New decisions should use\n"
            "--split=dev (150) and final eval --split=held-out (50).\n",
            file=sys.stderr,
        )

    if args.out_json is None:
        args.out_json = (
            Path("output/training-evidence")
            / f"benchmark-eval-{args.split}-{time.strftime('%Y-%m-%dT%H%M%S')}.json"
        )

    testing = _split_dir(args.denpar_root, "Testing")
    images_dir = testing / "Images"

    split_stems = _load_split(args.splits_dir, args.split)
    if split_stems is None:
        stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    else:
        all_on_disk = {p.stem for p in images_dir.glob("*.jpg")}
        missing = [s for s in split_stems if s not in all_on_disk]
        if missing:
            print(
                f"ERROR: split lists {len(missing)} stems not present on "
                f"disk: {missing[:5]}...",
                file=sys.stderr,
            )
            return 1
        stems = sorted(split_stems)

    if args.limit > 0:
        stems = stems[: args.limit]
    print(f"Benchmark eval — split={args.split} ({len(stems)} images).\n")

    bundle = _get_or_create_bundle(args.weights)

    errors_mm: list[float] = []
    gt_mms: list[float] = []
    pred_mms_for_matched: list[Optional[float]] = []
    n_gt_sites = 0
    n_pred_sites_when_gt = 0
    n_low_model_conf_sites = 0
    per_site_records: list[SiteError] = []

    t0 = time.perf_counter()
    for idx, stem in enumerate(stems):
        if idx % 25 == 0:
            print(f"... {idx}/{len(stems)}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

        gt_teeth = _derive_gt_mm(testing, stem)
        if not gt_teeth:
            continue

        img_path = images_dir / f"{stem}.jpg"
        img_for_shape = cv2.imread(str(img_path))
        if img_for_shape is None:
            continue
        img_h, img_w = img_for_shape.shape[:2]
        try:
            result = analyze(img_path, weights_dir=args.weights,
                             bundle=bundle, render=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {stem}: analyze() failed: {type(e).__name__}: {e}",
                  flush=True)
            continue

        # Match each GT tooth to a predicted tooth by bbox IoU.
        for gt in gt_teeth:
            best_iou = 0.0
            best_pred = None
            for t in result.teeth:
                iou = _bbox_iou(gt["bbox"], t.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = t
            if best_iou < 0.3 or best_pred is None:
                # GT exists but no predicted tooth matched.
                if gt["mesial_mm"] is not None:
                    n_gt_sites += 1
                    gt_mms.append(gt["mesial_mm"])
                if gt["distal_mm"] is not None:
                    n_gt_sites += 1
                    gt_mms.append(gt["distal_mm"])
                continue

            # Pull predicted mm.
            m_pred = (
                best_pred.bone_loss.mesial.mm_estimate
                if best_pred.bone_loss.mesial else None
            )
            d_pred = (
                best_pred.bone_loss.distal.mm_estimate
                if best_pred.bone_loss.distal else None
            )
            m_reason = (
                best_pred.bone_loss.mesial.reason
                if best_pred.bone_loss.mesial else None
            )
            d_reason = (
                best_pred.bone_loss.distal.reason
                if best_pred.bone_loss.distal else None
            )

            for surface, gt_mm, pred_mm, reason in (
                ("mesial", gt["mesial_mm"], m_pred, m_reason),
                ("distal", gt["distal_mm"], d_pred, d_reason),
            ):
                if gt_mm is None:
                    continue
                n_gt_sites += 1
                gt_mms.append(gt_mm)
                if pred_mm is not None:
                    n_pred_sites_when_gt += 1
                    err = abs(gt_mm - pred_mm)
                    errors_mm.append(err)
                    per_site_records.append(SiteError(
                        stem=stem, fdi_idx=gt["fdi_idx"], surface=surface,
                        gt_mm=gt_mm, pred_mm=pred_mm, abs_err_mm=err,
                        bbox_iou=best_iou,
                        bbox=tuple(gt["bbox"]),
                        image_w=img_w, image_h=img_h,
                    ))
                elif reason == "low_model_confidence":
                    n_low_model_conf_sites += 1

    elapsed = time.perf_counter() - t0

    err_stats = _percentile_stats(errors_mm)
    gt_stats = _percentile_stats(gt_mms)

    coverage = (
        n_pred_sites_when_gt / n_gt_sites if n_gt_sites > 0 else 0.0
    )

    print(f"\n{'='*70}\nBENCHMARK EVAL — DenPAR Testing split={args.split} "
          f"({len(stems)} images)\n{'='*70}")
    print(f"elapsed:               {elapsed:.0f}s")
    print(f"GT sites:              {n_gt_sites}")
    print(f"predicted (matched):   {n_pred_sites_when_gt} ({100*coverage:.1f}%)")
    print(f"low_model_confidence:  {n_low_model_conf_sites}")
    print()
    print("GT mm distribution:")
    print(f"  median={gt_stats['median']:.3f}  mean={gt_stats['mean']:.3f}  "
          f"p90={gt_stats['p90']:.3f}  max={gt_stats['max']:.3f}")
    print()
    print(f"mm-MAE on predicted sites (n={err_stats['n']}):")
    if err_stats['n'] > 0:
        print(f"  median={err_stats['median']:.3f}  mean(MAE)={err_stats['mean']:.3f}  "
              f"p90={err_stats['p90']:.3f}  max={err_stats['max']:.3f}")
    print()
    print("Vs published benchmarks (mm-MAE):")
    if err_stats['n'] > 0:
        for name, target in _BENCHMARKS_MM_MAE.items():
            delta = err_stats['mean'] - target
            arrow = "✓ BEAT" if err_stats['mean'] < target else "✗ MISS"
            print(f"  {arrow}  our {err_stats['mean']:.3f} vs {name}: {target:.3f}  (Δ {delta:+.3f})")
    print()
    print(f"per_site_mm_mae_mean: {err_stats['mean']:.3f}")
    if err_stats['mean'] is not None and err_stats['mean'] < 0.5:
        print("HEADLINE: under-0.5mm-MAE achieved")
    elif err_stats['mean'] is not None and err_stats['mean'] < 0.434:
        print("HEADLINE: beat Adravision BW")
    if err_stats['mean'] is not None and err_stats['mean'] < 0.3:
        print("HEADLINE: BEAT OVERJET")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(stems),
        "n_gt_sites": n_gt_sites,
        "n_predicted_matched": n_pred_sites_when_gt,
        "n_low_model_conf": n_low_model_conf_sites,
        "coverage": coverage,
        "elapsed_seconds": elapsed,
        "mm_mae_stats": err_stats,
        "gt_mm_stats": gt_stats,
        "benchmarks_mm_mae": _BENCHMARKS_MM_MAE,
        "per_site_records": [r.__dict__ for r in per_site_records],
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
