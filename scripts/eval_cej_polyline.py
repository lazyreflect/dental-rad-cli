"""Polyline-comparable CEJ evaluation — held-out DenPAR Testing split.

The v1 eval (`scripts/eval_keypoint_cej.py`) reports `cej_collapse_rate` —
fraction of predictions where the mesial and distal points predicted by
the keypoint head land within 10 px of each other. That metric goes
trivially to ~0 after the polyline pivot because the polyline post-
process emits endpoints at `bbox.x1` and `bbox.x2`, which are typically
80-300 px apart on a tooth bbox. Comparing pre-pivot to post-pivot
using `cej_collapse_rate` would be uninterpretable.

This script computes three metrics that ARE comparable across the two
architectures, and a px→mm conversion that anchors the headline number
to the AAP staging cutoff scale.

Modes
-----

- ``--mode keypoint`` — load the existing Keypoint R-CNN at
  ``weights/keypoint_cej.pt`` and report the baseline.
- ``--mode polyline`` — load the polyline-segmentation model at
  ``weights/segmentation_cej.pt`` and report the post-pivot number.

Both modes produce the same per-site y-error metric. Polyline mode
also produces CEJ-band pixel IoU and a polyline-degenerate-rate sanity
sibling that keypoint mode cannot produce.

Per-site y-error
----------------

For each ground-truth tooth bbox that has BOTH a mesial and distal CEJ
point in DenPAR v3 (~42% of teeth), compute:

  y_err_mesial = |pred_y_at_mesial_site − gt_mesial_y|
  y_err_distal = |pred_y_at_distal_site − gt_distal_y|

In keypoint mode the predicted-y-at-site is the matched detection's
left/right keypoint y (after sorting by x). In polyline mode it's the
band centerline's y at the GT x position (interpolated). Both are
clinically meaningful: bone-loss math runs along the tooth long axis,
so CEJ-y accuracy dominates over CEJ-x accuracy.

px→mm conversion
----------------

DenPAR v3 ships no per-image pixel scale. v0 uses an image-level
estimate: per image, ``px_per_mm = median(bbox_height_px) / 21.0``,
where 21 mm is a rough population mean of PA tooth length. v0.5 will
swap this for per-tooth-class priors once the parallel
`dental-tooth-numbering` substrate ships FDI numbering.

The mm number anchors to AlGhaihab 2025's published MAE 0.499 mm on BW
and the AAP staging cutoff of 2 mm. v2 targets: median < 1.0 mm,
p90 < 3.0 mm.

Usage
-----

::

    python scripts/eval_cej_polyline.py --mode keypoint
    python scripts/eval_cej_polyline.py --mode polyline \\
        --weights weights/segmentation_cej.pt

Outputs a structured summary plus the headline line
``per_site_y_error_median_mm: N.NN`` for grep-ability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dental_rad_cli.data.denpar_adapter import (  # noqa: E402
    _assign_points_to_bboxes,
    _load_keypoint_json,
    _sort_pair_left_right,
    _split_dir,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLAHE_CLIP = 40.0
_CLAHE_TILE = (8, 8)

# Population mean for px→mm conversion. Rough PA tooth length anchor.
# v0.5 replaces with per-tooth-class priors via dental-tooth-numbering.
_MEAN_PA_TOOTH_HEIGHT_MM = 21.0

# Bbox-prediction matching threshold (IoU below this counts as unmatched).
_BBOX_IOU_MATCH_MIN = 0.30

# Polyline-degenerate tolerance: band must cover x within this many pixels
# of bbox.x1 and bbox.x2 to count as non-degenerate.
_POLYLINE_EDGE_TOLERANCE_PX = 10

# Sanity sibling collapse threshold (matches v1 eval default).
_COLLAPSE_THRESHOLD_PX = 10.0


# ---------------------------------------------------------------------------
# Per-image GT
# ---------------------------------------------------------------------------


@dataclass
class GtTooth:
    """One tooth's ground truth, derived from DenPAR v3 keypoint JSON."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    mesial_cej: tuple[float, float] | None  # (x, y) or None
    distal_cej: tuple[float, float] | None


def _load_gt_for_image(testing_split: Path, stem: str) -> list[GtTooth] | None:
    """Build per-tooth GT for one test image.

    Uses the adapter's `_assign_points_to_bboxes` + `_sort_pair_left_right`
    heuristic — acknowledging it's the same loose pairing that produced
    training-time noise. For eval it's consistent with how training labels
    were built, and we filter to only teeth where it produced 2 CEJ pts
    (~42% of teeth).
    """
    kp = _load_keypoint_json(testing_split, stem)
    if kp is None:
        return None
    bboxes_raw = kp.get("bboxes") or []
    if not bboxes_raw:
        return None
    bboxes = [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in bboxes_raw
    ]
    cej_pts = [
        (float(p[0]), float(p[1])) for p in (kp.get("CEJ_Points") or [])
    ]
    cej_by_tooth = _assign_points_to_bboxes(cej_pts, bboxes)
    teeth: list[GtTooth] = []
    for i, bb in enumerate(bboxes):
        mesial, distal = _sort_pair_left_right(cej_by_tooth[i])
        teeth.append(GtTooth(bbox=bb, mesial_cej=mesial, distal_cej=distal))
    return teeth


def _px_per_mm_for_image(teeth: list[GtTooth]) -> float | None:
    """Image-level px→mm using median bbox height / 21 mm.

    Returns None if no teeth (we can't calibrate). Callers should fall
    back to px-only metrics for that image.
    """
    heights = [(t.bbox[3] - t.bbox[1]) for t in teeth if t.bbox[3] > t.bbox[1]]
    if not heights:
        return None
    return float(np.median(heights)) / _MEAN_PA_TOOTH_HEIGHT_MM


# ---------------------------------------------------------------------------
# Bbox matching
# ---------------------------------------------------------------------------


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _match_prediction_to_gt(
    pred_bbox: tuple[float, float, float, float], teeth: list[GtTooth]
) -> int | None:
    """Return index of GT tooth with highest IoU, or None if best < threshold."""
    best_iou = 0.0
    best_idx = -1
    for i, t in enumerate(teeth):
        iou = _bbox_iou(pred_bbox, t.bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    if best_iou < _BBOX_IOU_MATCH_MIN:
        return None
    return best_idx


# ---------------------------------------------------------------------------
# Common preprocessing
# ---------------------------------------------------------------------------


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
    return cv2.cvtColor(cv2.merge([cl.apply(L), a, b]), cv2.COLOR_LAB2RGB)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Keypoint-mode prediction
# ---------------------------------------------------------------------------


def _load_keypoint_model(weights_path: Path, device: torch.device):
    """Load the Keypoint R-CNN once; reuse across images."""
    from dental_rad_cli.training.keypoints import _build_model

    payload = torch.load(str(weights_path), map_location="cpu")
    state = (
        payload["state_dict"]
        if isinstance(payload, dict) and "state_dict" in payload
        else payload
    )
    num_keypoints = (
        int(payload.get("num_keypoints", 2)) if isinstance(payload, dict) else 2
    )
    model = _build_model(num_keypoints=num_keypoints)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def _predict_keypoint(
    model,
    device: torch.device,
    rgb: np.ndarray,
    score_threshold: float,
) -> list[tuple[tuple[float, float, float, float], tuple[float, float], tuple[float, float]]]:
    """Run Keypoint R-CNN; return list of (bbox, left_kpt_xy, right_kpt_xy)."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
    with torch.no_grad():
        out = model([t])[0]
    bboxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    kps = out["keypoints"].cpu().numpy()

    results: list[
        tuple[tuple[float, float, float, float], tuple[float, float], tuple[float, float]]
    ] = []
    for k in range(len(scores)):
        if scores[k] < score_threshold:
            continue
        x1, y1, x2, y2 = bboxes[k]
        p0 = (float(kps[k, 0, 0]), float(kps[k, 0, 1]))
        p1 = (float(kps[k, 1, 0]), float(kps[k, 1, 1]))
        left, right = (p0, p1) if p0[0] <= p1[0] else (p1, p0)
        results.append(((float(x1), float(y1), float(x2), float(y2)), left, right))
    return results


# ---------------------------------------------------------------------------
# Polyline-mode prediction
# ---------------------------------------------------------------------------


def _load_polyline_model(weights_path: Path):
    """Load the YOLOv8x-seg CEJ model once; reuse across images."""
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _predict_polyline(
    model, rgb: np.ndarray, conf_threshold: float
) -> np.ndarray:
    """Run YOLOv8x-seg CEJ model; return union of all detection masks.

    Returns a binary mask at original image resolution. If no detections
    above threshold, returns an all-zero mask.
    """
    h, w = rgb.shape[:2]
    out = model.predict(
        source=rgb,
        conf=conf_threshold,
        verbose=False,
        save=False,
    )
    if not out:
        return np.zeros((h, w), dtype=bool)
    result = out[0]
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((h, w), dtype=bool)
    # masks.data shape: (n, mask_h, mask_w) — typically downsampled.
    masks = result.masks.data.cpu().numpy().astype(bool)
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        m_rs = cv2.resize(
            m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        union |= m_rs
    return union


def _band_centerline_y_at_x(band: np.ndarray, x: float) -> float | None:
    """Median y of band pixels at integer column round(x). None if empty."""
    h, w = band.shape
    xi = int(round(x))
    if xi < 0 or xi >= w:
        return None
    column = band[:, xi]
    ys = np.flatnonzero(column)
    if ys.size == 0:
        return None
    return float(np.median(ys))


# ---------------------------------------------------------------------------
# CEJ-band ground-truth mask (polyline mode IoU)
# ---------------------------------------------------------------------------


def _build_gt_cej_band(
    teeth: list[GtTooth], img_shape: tuple[int, int]
) -> np.ndarray:
    """Build the y-band-clustered GT CEJ band at image resolution.

    Mirrors the polyline adapter's supervision construction so the IoU
    metric compares apples-to-apples: cluster CEJ points by y-proximity,
    connect cluster members in x-order, dilate to a 30-px band. We
    intentionally do this from the (mesial, distal) pairs we extracted
    per tooth — not from the raw flat CEJ_Points list — so 0-CEJ-pt
    teeth contribute nothing and the band only spans where we have GT.
    """
    h, w = img_shape
    band = np.zeros((h, w), dtype=np.uint8)
    pts: list[tuple[float, float]] = []
    for t in teeth:
        if t.mesial_cej is not None:
            pts.append(t.mesial_cej)
        if t.distal_cej is not None:
            pts.append(t.distal_cej)
    if not pts:
        return band.astype(bool)
    # Y-band cluster: group points within 30 px y of nearest neighbor in cluster.
    pts_sorted = sorted(pts, key=lambda p: p[1])
    clusters: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    last_y: float | None = None
    for p in pts_sorted:
        if last_y is None or abs(p[1] - last_y) <= 30.0:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
        last_y = p[1]
    if current:
        clusters.append(current)
    # Draw each cluster's polyline (sorted by x), dilate to 30 px total.
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        line = sorted(cluster, key=lambda p: p[0])
        pts_np = np.array(
            [[int(round(x)), int(round(y))] for x, y in line], dtype=np.int32
        )
        cv2.polylines(
            band, [pts_np], isClosed=False, color=255, thickness=30
        )
    return band.astype(bool)


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------


@dataclass
class ImageMetrics:
    """Accumulator for one image's per-site results."""

    n_teeth_with_both_cej: int = 0
    y_errors_px: list[float] = field(default_factory=list)
    y_errors_mm: list[float] = field(default_factory=list)
    md_distances_px: list[float] = field(default_factory=list)
    n_collapsed: int = 0
    n_teeth_evaluated_polyline: int = 0
    n_polyline_degenerate: int = 0
    iou: float | None = None


def _evaluate_image_keypoint(
    teeth: list[GtTooth],
    predictions: list[
        tuple[tuple[float, float, float, float], tuple[float, float], tuple[float, float]]
    ],
    px_per_mm: float | None,
) -> ImageMetrics:
    m = ImageMetrics()
    for pred_bbox, pred_left, pred_right in predictions:
        # Sanity-sibling: collapse distance regardless of GT match.
        d = math.hypot(pred_left[0] - pred_right[0], pred_left[1] - pred_right[1])
        m.md_distances_px.append(d)
        if d < _COLLAPSE_THRESHOLD_PX:
            m.n_collapsed += 1

        gt_idx = _match_prediction_to_gt(pred_bbox, teeth)
        if gt_idx is None:
            continue
        gt = teeth[gt_idx]
        if gt.mesial_cej is None or gt.distal_cej is None:
            continue
        m.n_teeth_with_both_cej += 1
        # Compare predicted left/right CEJ y to GT mesial/distal y.
        y_err_mesial_px = abs(pred_left[1] - gt.mesial_cej[1])
        y_err_distal_px = abs(pred_right[1] - gt.distal_cej[1])
        m.y_errors_px.extend([y_err_mesial_px, y_err_distal_px])
        if px_per_mm is not None and px_per_mm > 0:
            m.y_errors_mm.extend(
                [y_err_mesial_px / px_per_mm, y_err_distal_px / px_per_mm]
            )
    return m


def _evaluate_image_polyline(
    teeth: list[GtTooth],
    band: np.ndarray,
    px_per_mm: float | None,
) -> ImageMetrics:
    m = ImageMetrics()
    gt_band = _build_gt_cej_band(teeth, band.shape)
    if gt_band.any() or band.any():
        inter = np.logical_and(band, gt_band).sum()
        union = np.logical_or(band, gt_band).sum()
        m.iou = float(inter / union) if union > 0 else 0.0
    for t in teeth:
        if t.mesial_cej is None or t.distal_cej is None:
            continue
        m.n_teeth_with_both_cej += 1
        m.n_teeth_evaluated_polyline += 1
        # Polyline-degenerate check: does the band have pixels within
        # tolerance of bbox.x1 and bbox.x2?
        x1, _, x2, _ = t.bbox
        left_cover = any(
            _band_centerline_y_at_x(band, x1 + dx) is not None
            for dx in range(-_POLYLINE_EDGE_TOLERANCE_PX, _POLYLINE_EDGE_TOLERANCE_PX + 1)
        )
        right_cover = any(
            _band_centerline_y_at_x(band, x2 + dx) is not None
            for dx in range(-_POLYLINE_EDGE_TOLERANCE_PX, _POLYLINE_EDGE_TOLERANCE_PX + 1)
        )
        if not (left_cover and right_cover):
            m.n_polyline_degenerate += 1
            continue  # Don't compute y-error on degenerate sites.
        # Y-error at GT CEJ x positions.
        pred_y_mesial = _band_centerline_y_at_x(band, t.mesial_cej[0])
        pred_y_distal = _band_centerline_y_at_x(band, t.distal_cej[0])
        if pred_y_mesial is None or pred_y_distal is None:
            m.n_polyline_degenerate += 1
            continue
        y_err_mesial_px = abs(pred_y_mesial - t.mesial_cej[1])
        y_err_distal_px = abs(pred_y_distal - t.distal_cej[1])
        m.y_errors_px.extend([y_err_mesial_px, y_err_distal_px])
        if px_per_mm is not None and px_per_mm > 0:
            m.y_errors_mm.extend(
                [y_err_mesial_px / px_per_mm, y_err_distal_px / px_per_mm]
            )
    return m


# ---------------------------------------------------------------------------
# Top-level eval loop
# ---------------------------------------------------------------------------


def evaluate(
    mode: Literal["keypoint", "polyline"],
    weights_path: Path,
    denpar_root: Path,
    score_threshold: float = 0.5,
) -> dict:
    device = _resolve_device()
    testing_split = _split_dir(denpar_root, "Testing")
    images_dir = testing_split / "Images"
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no .jpg images in {images_dir}")

    # Load model once, reuse across images.
    if mode == "keypoint":
        model = _load_keypoint_model(weights_path, device)
    else:
        model = _load_polyline_model(weights_path)

    # Aggregate state.
    all_y_err_px: list[float] = []
    all_y_err_mm: list[float] = []
    all_md_dist_px: list[float] = []
    n_collapsed_total = 0
    n_md_total = 0
    n_teeth_with_both_total = 0
    n_polyline_degen_total = 0
    n_polyline_eval_total = 0
    iou_per_image: list[float] = []
    n_images_processed = 0

    t0 = time.perf_counter()
    for p in images:
        stem = p.stem
        teeth = _load_gt_for_image(testing_split, stem)
        if teeth is None:
            continue
        n_images_processed += 1
        px_per_mm = _px_per_mm_for_image(teeth)

        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = _apply_clahe(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if mode == "keypoint":
            preds = _predict_keypoint(model, device, rgb, score_threshold)
            m = _evaluate_image_keypoint(teeth, preds, px_per_mm)
        else:
            band = _predict_polyline(model, rgb, score_threshold)
            m = _evaluate_image_polyline(teeth, band, px_per_mm)
            if m.iou is not None:
                iou_per_image.append(m.iou)

        all_y_err_px.extend(m.y_errors_px)
        all_y_err_mm.extend(m.y_errors_mm)
        all_md_dist_px.extend(m.md_distances_px)
        n_collapsed_total += m.n_collapsed
        n_md_total += len(m.md_distances_px)
        n_teeth_with_both_total += m.n_teeth_with_both_cej
        n_polyline_degen_total += m.n_polyline_degenerate
        n_polyline_eval_total += m.n_teeth_evaluated_polyline

    elapsed = time.perf_counter() - t0

    def _percentiles(arr: list[float]) -> dict:
        if not arr:
            return {"median": None, "mean": None, "p10": None, "p25": None, "p75": None, "p90": None}
        a = np.array(arr)
        return {
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }

    y_err_px_stats = _percentiles(all_y_err_px)
    y_err_mm_stats = _percentiles(all_y_err_mm)
    md_dist_stats = _percentiles(all_md_dist_px)

    return {
        "mode": mode,
        "device": str(device),
        "n_images": n_images_processed,
        "n_sites_evaluated": len(all_y_err_px),  # 2 per qualifying tooth
        "n_teeth_with_both_cej": n_teeth_with_both_total,
        "per_site_y_error_px": y_err_px_stats,
        "per_site_y_error_mm": y_err_mm_stats,
        "polyline_degenerate_rate": (
            n_polyline_degen_total / max(n_polyline_eval_total, 1)
            if mode == "polyline"
            else None
        ),
        "polyline_band_iou_mean": (
            float(np.mean(iou_per_image)) if iou_per_image else None
        ),
        "polyline_band_iou_median": (
            float(np.median(iou_per_image)) if iou_per_image else None
        ),
        # Sanity sibling — collapse rate on mesial-distal predicted distance.
        "cej_collapse_rate": (
            n_collapsed_total / max(n_md_total, 1) if mode == "keypoint" else None
        ),
        "md_distance_px": md_dist_stats,
        "score_threshold": score_threshold,
        "elapsed_seconds": elapsed,
        "px_per_mm_anchor_tooth_height_mm": _MEAN_PA_TOOTH_HEIGHT_MM,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_stats(label: str, stats: dict, unit: str) -> list[str]:
    if stats["median"] is None:
        return [f"  {label}: (no data)"]
    return [
        f"  {label}_median: {stats['median']:.2f} {unit}",
        f"  {label}_mean:   {stats['mean']:.2f} {unit}",
        f"  {label}_p10/25/75/90: "
        f"{stats['p10']:.1f} / {stats['p25']:.1f} / "
        f"{stats['p75']:.1f} / {stats['p90']:.1f} {unit}",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["keypoint", "polyline"],
        required=True,
        help="keypoint = baseline R-CNN; polyline = post-pivot YOLOv8x-seg",
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="default: weights/keypoint_cej.pt (kp) or weights/segmentation_cej.pt (poly)",
    )
    ap.add_argument(
        "--denpar-root", type=Path, default=Path("data/denpar")
    )
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="optional path to write the full metrics dict as JSON",
    )
    args = ap.parse_args()

    if args.weights is None:
        args.weights = (
            Path("weights/keypoint_cej.pt")
            if args.mode == "keypoint"
            else Path("weights/segmentation_cej.pt")
        )
    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.denpar_root.is_dir():
        print(f"ERROR: denpar root not found: {args.denpar_root}", file=sys.stderr)
        return 1

    result = evaluate(
        args.mode,
        args.weights,
        args.denpar_root,
        score_threshold=args.score_threshold,
    )

    print(f"mode:                       {result['mode']}")
    print(f"device:                     {result['device']}")
    print(f"images:                     {result['n_images']}")
    print(f"teeth_with_both_cej:        {result['n_teeth_with_both_cej']}")
    print(f"sites_evaluated:            {result['n_sites_evaluated']}")
    print(f"elapsed_seconds:            {result['elapsed_seconds']:.1f}")
    print()
    print("per-site y-error (px):")
    for line in _format_stats("y_err", result["per_site_y_error_px"], "px"):
        print(line)
    print()
    print("per-site y-error (mm):")
    for line in _format_stats("y_err", result["per_site_y_error_mm"], "mm"):
        print(line)
    print()
    if result["mode"] == "polyline":
        if result["polyline_band_iou_mean"] is not None:
            print(
                f"polyline_band_iou_mean:     {result['polyline_band_iou_mean']:.3f}"
            )
            print(
                f"polyline_band_iou_median:   {result['polyline_band_iou_median']:.3f}"
            )
        if result["polyline_degenerate_rate"] is not None:
            print(
                f"polyline_degenerate_rate:   {result['polyline_degenerate_rate']:.4f}"
            )
    if result["mode"] == "keypoint":
        print("sanity — mesial-distal distance (px):")
        for line in _format_stats("md_dist", result["md_distance_px"], "px"):
            print(line)
        print(
            f"cej_collapse_rate (sanity): {result['cej_collapse_rate']:.4f}"
        )
    print()
    # Headline line for grep-ability.
    median_mm = result["per_site_y_error_mm"]["median"]
    if median_mm is None:
        print("per_site_y_error_median_mm: nan")
    else:
        print(f"per_site_y_error_median_mm: {median_mm:.3f}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
