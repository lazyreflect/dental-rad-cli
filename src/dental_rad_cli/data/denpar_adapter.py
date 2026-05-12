"""DenPAR v3 → training-format adapter.

Converts the DenPAR v3 directory layout (Zenodo record 16645076) into
the formats consumed by the two training paths in this repo:

- **YOLO** (Ultralytics): images + per-image `.txt` labels and a
  `dataset.yaml`. One label file per image; one line per object.
  Detection rows: ``class cx cy w h`` (normalized).
  Segmentation rows: ``class x1 y1 x2 y2 ...`` (normalized polygon).
- **COCO-keypoints**: a single JSON per split, in standard COCO format
  with a per-tooth flat ``keypoints`` field of length 18 (6 keypoints
  * 3). The trainer in ``training/keypoints.py`` slices indices
  ``cej=0:2``, ``bone=2:4``, ``apex=4:5`` from this layout.

## DenPAR v3 schema (verified by direct inspection — 2026-05-11)

The methodology brief at /tmp/dental-rad-methodology-brief.md
describes what upstream's training code expects (DenPAR **v2** flat
layout + COCO-keypoints with paired-per-tooth keypoints). DenPAR
**v3** does NOT match that — verified by reading the actual JSON files
on disk. The truth on disk is:

### Directory layout
```
data/denpar/Dataset/
├── Characteristics of radiographs included.xlsx
├── Training/                            (650 images)
│   ├── Images/                          *.jpg (variable resolution, e.g. 885x1167)
│   ├── Masks (Tooth-wise)/<image_id>/   mask1.png, mask2.png, ...  binary L-mode, 0/255
│   ├── Masks (Radiograph-wise)/         <image_id>.png             binary single mask
│   ├── Bone Level Annotations/          <image_id>.json
│   └── Key Points Annotations/          <image_id>.json
├── Validation/  (150 images, same shape)
└── Testing/     (200 images, same shape)
```

### Key Points JSON shape
Real sample (`Training/Key Points Annotations/10.json`):
```json
{
  "Image_id": "10.jpg",
  "bboxes":   [[592.0, 271.0, 786.0, 970.0], ...],          // pascal_voc absolute [x1,y1,x2,y2], one per tooth
  "CEJ_Points":  [[278.063, 519.795], ...],                 // loose 2-D points, NOT 1:1 with bboxes
  "Apex_Points": [[284.73, 1005.128], ...]
}
```

### Critical deltas from v2 / upstream expectation
1. **Keypoints are NOT paired-per-tooth.** Both `CEJ_Points` and
   `Apex_Points` are loose lists whose count differs from `len(bboxes)`.
   Example: 10.json has 4 teeth (bboxes), 8 CEJ points (~2/tooth), 4
   apex points. 1002.json has 4 bboxes, 5 CEJ points, 3 apex points.
   The adapter must pair points to bboxes via a geometric heuristic
   (nearest-bbox containment + nearest-bbox distance fallback).
2. **No bone-crest (AEAC) keypoints.** v2 had CEJ + AEAC + APEX;
   v3 ships bone-crest information as polylines in
   `Bone Level Annotations/<image_id>.json`, NOT as discrete keypoints.
   The adapter derives bone-crest keypoints per-tooth by intersecting
   the nearest bone polyline endpoint with the tooth bbox.
3. **No single/double-rooted labels.** v2 had `labels: [1,2,1,...]`;
   v3 has bboxes only. The adapter infers root count from the number
   of apex points falling under each bbox (1 → single; ≥2 → double),
   and emits `category_id = 1` (single) or `2` (double) to match the
   trainer's `num_classes=3` (bg + single + double) expectation.
4. **Per-tooth masks are folder-of-instances.** Each tooth bbox `i`
   corresponds to `Masks (Tooth-wise)/<image_id>/mask{i+1}.png` (1-indexed).
   Mask count always equals bbox count (verified spot-check).
   FDI numbering is NOT encoded — masks are ordered by bbox order.

### Bone Level Annotations JSON shape
Real sample (`Training/Bone Level Annotations/10.json`):
```json
{
  "Image_id": "10.jpg",
  "Num_of_Bone_Lines": 3,
  "Bone_Lines": [
    [[307.13, 727.62], [337.39, 723.47], [367.85, 694.55]],
    [[448.05, 676.75], [475.02, 673.40]],
    [[628.31, 675.72], [608.93, 732.86], [585.13, 767.93], [573.85, 836.04]]
  ]
}
```
Each polyline is an open chain of [x,y] points along the alveolar
bone crest between teeth. Used by the rule-layer pattern classifier
(`pipeline/pattern.py`) and as the seed for per-tooth bone-crest
keypoints (this adapter).

### FDI numbering
DenPAR v3 does NOT provide FDI tooth numbers. Per-tooth indices are
positional (bbox-order = mask-order). Inference-time FDI assignment is
a separate problem owned by the rule layer (`pipeline/`), not this
adapter. Per-tooth artifacts here carry a 0-indexed `tooth_idx` only.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Final, Literal

import numpy as np
from PIL import Image

_LOG = logging.getLogger(__name__)

# DenPAR v3 ships these three splits with these exact folder names.
_V3_SPLITS: Final[dict[str, str]] = {
    "Training": "train",
    "Validation": "val",
    "Testing": "test",
}

# Subfolder names inside each split.
_SUB_IMAGES: Final[str] = "Images"
_SUB_KEYPOINTS: Final[str] = "Key Points Annotations"
_SUB_BONELEVEL: Final[str] = "Bone Level Annotations"
_SUB_MASKS_TOOTH: Final[str] = "Masks (Tooth-wise)"
_SUB_MASKS_RAD: Final[str] = "Masks (Radiograph-wise)"

# COCO category ids — matched to keypoint trainer's `num_classes=3`
# (background + single + double). Background is implicit; we emit 1/2.
_CAT_SINGLE: Final[int] = 1
_CAT_DOUBLE: Final[int] = 2

# Total keypoints per tooth in the emitted COCO file — 6 (cej_l, cej_r,
# bone_l, bone_r, apex_a, apex_b). Trainer slices this.
_NUM_KEYPOINTS_TOTAL: Final[int] = 6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_dir(denpar_root: Path, v3_split: str) -> Path:
    """Resolve the v3 split directory (Training / Validation / Testing)."""
    p = denpar_root / "Dataset" / v3_split
    if not p.is_dir():
        raise FileNotFoundError(
            f"DenPAR v3 split not found: {p} — expected unzipped "
            "Zenodo record 16645076 layout."
        )
    return p


def _image_ids_for_split(split_root: Path) -> list[str]:
    """Return image stems (no extension) for a split, sorted for determinism."""
    images_dir = split_root / _SUB_IMAGES
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing Images dir: {images_dir}")
    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    if not stems:
        raise RuntimeError(f"no .jpg images in {images_dir}")
    return stems


def _load_image_size(images_dir: Path, stem: str) -> tuple[int, int]:
    """Return (width, height) for the image with this stem."""
    with Image.open(images_dir / f"{stem}.jpg") as im:
        return im.size  # PIL returns (W, H)


def _load_keypoint_json(split_root: Path, stem: str) -> dict | None:
    """Load the keypoint annotation JSON; return None if missing."""
    p = split_root / _SUB_KEYPOINTS / f"{stem}.json"
    if not p.is_file():
        return None
    with p.open() as fh:
        return json.load(fh)


def _load_bonelevel_json(split_root: Path, stem: str) -> dict | None:
    """Load the bone-level annotation JSON; return None if missing."""
    p = split_root / _SUB_BONELEVEL / f"{stem}.json"
    if not p.is_file():
        return None
    with p.open() as fh:
        return json.load(fh)


def _point_in_bbox(pt: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """Inclusive containment check; bbox is (x1,y1,x2,y2)."""
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _dist_point_to_bbox_center(
    pt: tuple[float, float], bbox: tuple[float, float, float, float]
) -> float:
    """Euclidean distance from a point to the bbox center (for fallback)."""
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    return float(np.hypot(pt[0] - cx, pt[1] - cy))


def _assign_points_to_bboxes(
    points: list[tuple[float, float]],
    bboxes: list[tuple[float, float, float, float]],
) -> list[list[tuple[float, float]]]:
    """Group loose 2-D points by which tooth bbox they belong to.

    Strategy (load-bearing pairing heuristic — see module docstring):
    1. For each point, find all bboxes containing it. Assign to the
       single containing bbox if there's exactly one.
    2. If a point is contained in 0 bboxes OR in 2+ bboxes (rare —
       overlapping teeth), assign to the bbox whose center is nearest.

    Returns a list of lists: `out[i]` are the points associated with
    `bboxes[i]`, in source order (no sort).
    """
    out: list[list[tuple[float, float]]] = [[] for _ in bboxes]
    for pt in points:
        containing: list[int] = [
            i for i, bb in enumerate(bboxes) if _point_in_bbox(pt, bb)
        ]
        if len(containing) == 1:
            idx = containing[0]
        else:
            # 0 or 2+ — fallback to nearest-center.
            idx = min(
                range(len(bboxes)),
                key=lambda i: _dist_point_to_bbox_center(pt, bboxes[i]),
            )
        out[idx].append(pt)
    return out


def _sort_pair_left_right(pts: list[tuple[float, float]]) -> tuple[
    tuple[float, float] | None, tuple[float, float] | None
]:
    """Pick up to 2 points and return (left, right) by x-coordinate.

    If fewer than 2 points are present, the missing slot is None.
    If 3+ points are present, return the two with extremal x.
    """
    if not pts:
        return None, None
    if len(pts) == 1:
        return pts[0], None
    pts_sorted = sorted(pts, key=lambda p: p[0])
    return pts_sorted[0], pts_sorted[-1]


def _bone_crest_for_bbox(
    bbox: tuple[float, float, float, float],
    bone_lines: list[list[tuple[float, float]]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Derive (mesial, distal) bone-crest keypoints for one tooth.

    DenPAR v3 has NO discrete bone-crest keypoints — only polylines
    tracing the alveolar bone crest level across the whole radiograph.
    The clinically meaningful bone-crest landmark for a tooth is at
    the mesial and distal interproximal sites (i.e. at the tooth bbox's
    left and right edges), NOT at the polyline endpoints (the prior
    v3-era heuristic that produced misleading training labels).

    Strategy: linearly interpolate each polyline segment at
    ``x = bbox.x1`` (mesial) and ``x = bbox.x2`` (distal). If multiple
    polylines cover the same x (rare but possible when both jaws are
    visible), prefer the most coronal one — closer to the CEJ — by
    taking the smallest y (image origin is top-left).

    Returns ``(mesial_point, distal_point)`` with either / both as
    ``None`` if no polyline covers the corresponding bbox edge.
    """
    x1, _y1, x2, _y2 = bbox

    def interp_y_at_x(target_x: float) -> float | None:
        """Linearly interpolate any covering polyline at ``target_x``."""
        candidates: list[float] = []
        for line in bone_lines:
            for (xa, ya), (xb, yb) in zip(line, line[1:]):
                lo, hi = (xa, xb) if xa <= xb else (xb, xa)
                if lo <= target_x <= hi:
                    if xa == xb:
                        # Vertical segment — both vertices share x;
                        # take the more coronal (smaller y).
                        candidates.append(float(min(ya, yb)))
                    else:
                        t = (target_x - xa) / (xb - xa)
                        candidates.append(float(ya + t * (yb - ya)))
                    break  # only first matching segment per polyline
        if not candidates:
            return None
        # Most coronal point wins (smallest y in image coords).
        return min(candidates)

    mesial_y = interp_y_at_x(x1)
    distal_y = interp_y_at_x(x2)

    mesial = (x1, mesial_y) if mesial_y is not None else None
    distal = (x2, distal_y) if distal_y is not None else None
    return mesial, distal


def _pair_with_visibility(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> list[float]:
    """Encode a (left, right) point pair as a 6-element COCO keypoint slice.

    Each point is `[x, y, vis]` with `vis=2` (labeled+visible) when
    present, `[0, 0, 0]` when absent. Order is fixed: left first, then
    right.
    """
    l_xyv = [left[0], left[1], 2.0] if left is not None else [0.0, 0.0, 0.0]
    r_xyv = [right[0], right[1], 2.0] if right is not None else [0.0, 0.0, 0.0]
    return l_xyv + r_xyv


def _infer_root_label(apex_points_for_tooth: list[tuple[float, float]]) -> int:
    """Single (1) vs double (2) rooted, from apex-point count under the bbox.

    Heuristic: ``len(apex_points_for_tooth) >= 2`` ⇒ double-rooted.
    This matches the upstream `num_classes=3` (bg + single + double)
    while honoring that v3 has no explicit label field.
    """
    return _CAT_DOUBLE if len(apex_points_for_tooth) >= 2 else _CAT_SINGLE


# ---------------------------------------------------------------------------
# Per-image extraction (shared by YOLO + COCO paths)
# ---------------------------------------------------------------------------


def _extract_per_tooth(
    split_root: Path, stem: str
) -> tuple[list[tuple[float, float, float, float]], list[list[float]], list[int]] | None:
    """Build per-tooth (bbox, 18-element keypoints, label) lists for one image.

    Returns None if the keypoint JSON is missing or contains no bboxes.
    Otherwise returns ``(bboxes, keypoints_flat, labels)`` where each
    list has length ``N_teeth`` and ``keypoints_flat[i]`` is the
    18-element COCO flat list (cej_l, cej_r, bone_l, bone_r, apex_a,
    apex_b) for tooth ``i``.
    """
    kp = _load_keypoint_json(split_root, stem)
    if kp is None:
        return None

    bboxes_raw = kp.get("bboxes") or []
    if not bboxes_raw:
        return None
    bboxes: list[tuple[float, float, float, float]] = [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in bboxes_raw
    ]

    cej_pts: list[tuple[float, float]] = [
        (float(p[0]), float(p[1])) for p in (kp.get("CEJ_Points") or [])
    ]
    apex_pts: list[tuple[float, float]] = [
        (float(p[0]), float(p[1])) for p in (kp.get("Apex_Points") or [])
    ]

    # Assign loose CEJ / apex points to bboxes.
    cej_by_tooth = _assign_points_to_bboxes(cej_pts, bboxes)
    apex_by_tooth = _assign_points_to_bboxes(apex_pts, bboxes)

    # Bone-crest from bone-level polylines, per tooth.
    bl = _load_bonelevel_json(split_root, stem) or {}
    bone_lines_raw = bl.get("Bone_Lines") or []
    bone_lines: list[list[tuple[float, float]]] = [
        [(float(p[0]), float(p[1])) for p in line] for line in bone_lines_raw
    ]

    keypoints_flat: list[list[float]] = []
    labels: list[int] = []
    for i, bbox in enumerate(bboxes):
        cej_l, cej_r = _sort_pair_left_right(cej_by_tooth[i])
        bone_l, bone_r = _bone_crest_for_bbox(bbox, bone_lines)
        apex_l, apex_r = _sort_pair_left_right(apex_by_tooth[i])

        flat = (
            _pair_with_visibility(cej_l, cej_r)
            + _pair_with_visibility(bone_l, bone_r)
            + _pair_with_visibility(apex_l, apex_r)
        )
        assert len(flat) == _NUM_KEYPOINTS_TOTAL * 3, "keypoint slice malformed"
        keypoints_flat.append(flat)
        labels.append(_infer_root_label(apex_by_tooth[i]))

    return bboxes, keypoints_flat, labels


def _mask_polygons_for_image(
    split_root: Path, stem: str, n_teeth: int
) -> list[list[tuple[float, float]]] | None:
    """Extract one polygon per tooth from `Masks (Tooth-wise)/<stem>/maskN.png`.

    Uses OpenCV `findContours` on each binary mask, returns the largest
    contour as a list of (x, y) points. Returns None if the mask folder
    is missing. Polygon coordinates are absolute pixels.
    """
    import cv2

    mask_dir = split_root / _SUB_MASKS_TOOTH / stem
    if not mask_dir.is_dir():
        return None
    polygons: list[list[tuple[float, float]]] = []
    for i in range(1, n_teeth + 1):
        p = mask_dir / f"mask{i}.png"
        if not p.is_file():
            polygons.append([])
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            polygons.append([])
            continue
        _, binm = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        contours, _hier = cv2.findContours(
            binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            polygons.append([])
            continue
        # Largest contour by area
        largest = max(contours, key=cv2.contourArea)
        # Shape (K, 1, 2) -> list of (x, y)
        pts = [(float(p[0][0]), float(p[0][1])) for p in largest]
        polygons.append(pts)
    return polygons


def _bone_polygons_for_image(
    split_root: Path, stem: str
) -> list[list[tuple[float, float]]]:
    """Synthesize bone-region polygons from the DenPAR v3 polylines.

    DenPAR v3 has NO bone region masks anywhere. The
    ``Masks (Radiograph-wise)/`` folder (which the v2-era pipeline
    treated as the bone mask) actually contains all-teeth-as-one-mask
    in v3 — verified by visual inspection. This was Subagent F's
    silent miss at hour-0; the overnight ``segmentation_bone.pt`` is
    structurally a duplicate of ``segmentation_tooth.pt`` as a result.

    Correct source: ``Bone Level Annotations/<stem>.json`` containing
    one or more polylines along the alveolar bone crest. Many BWs
    have multiple polylines at disjoint x ranges (one per bone-crest
    segment where bone is annotatable). We synthesize one polygon per
    disjoint strip:

    1. Load the polylines.
    2. Buffer each by ±``_BONE_STRIP_HALF_WIDTH`` px.
    3. Union overlapping strips, keep disjoint strips as separate
       polygons.
    4. Return ALL resulting polygons (caller emits each as a YOLO-seg
       instance).

    Returns ``[]`` if no polylines are present or all buffers fail.
    """
    p = split_root / "Bone Level Annotations" / f"{stem}.json"
    if not p.is_file():
        return []
    import json
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return []
    bone_lines = data.get("Bone_Lines") or []
    return _bone_polygons_from_polylines(bone_lines)


# Backward-compatibility alias retained briefly for tests that may
# still import the old name. Deprecated — call _bone_polygons_for_image.
def _bone_polygon_for_image(
    split_root: Path, stem: str
) -> list[tuple[float, float]] | None:
    polys = _bone_polygons_for_image(split_root, stem)
    if not polys:
        return None
    # Return the largest (preserves old caller semantics).
    return max(polys, key=lambda p: len(p))


# Width of the bone-crest strip generated by polyline buffering.
# Tighter values give the model a more precise localization signal
# but risk training instability if the polyline geometry is jagged.
# 30 px (15 each side) chosen empirically against DenPAR v3 BW samples
# at ~1100 px tall — covers the trabecular bone region just below the
# crest line without bleeding into tooth roots.
_BONE_STRIP_HALF_WIDTH: Final[int] = 15


def _bone_polygons_from_polylines(
    bone_lines: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Buffer + union polylines into a list of bone-region polygons.

    Uses shapely for buffer + union math. Each polyline becomes a
    "capsule" strip (rectangle with rounded ends) of width
    ``2 * _BONE_STRIP_HALF_WIDTH``. Overlapping strips merge; disjoint
    strips remain as separate polygons in the output list.

    Returns ``[]`` if input is empty / degenerate / produces no valid
    geometry. Otherwise returns one or more polygons, each as a list
    of (x, y) exterior coordinates.
    """
    if not bone_lines:
        return []
    try:
        from shapely.geometry import LineString, MultiPolygon, Polygon
        from shapely.ops import unary_union
    except ImportError:
        return []

    strips = []
    for line in bone_lines:
        if len(line) < 2:
            continue
        try:
            ls = LineString([(float(x), float(y)) for x, y in line])
            if not ls.is_valid or ls.is_empty:
                continue
            strip = ls.buffer(_BONE_STRIP_HALF_WIDTH)
            if not strip.is_empty:
                strips.append(strip)
        except (TypeError, ValueError):
            continue

    if not strips:
        return []

    merged = unary_union(strips)

    # Collect every Polygon in the result (Polygon, MultiPolygon, or
    # GeometryCollection containing polygons).
    polygons: list[Polygon] = []
    if isinstance(merged, Polygon):
        if not merged.is_empty:
            polygons.append(merged)
    elif isinstance(merged, MultiPolygon):
        for geom in merged.geoms:
            if isinstance(geom, Polygon) and not geom.is_empty:
                polygons.append(geom)
    else:
        # GeometryCollection or other — extract any Polygons inside.
        for geom in getattr(merged, "geoms", []):
            if isinstance(geom, Polygon) and not geom.is_empty:
                polygons.append(geom)

    return [
        [(float(x), float(y)) for x, y in p.exterior.coords]
        for p in polygons
    ]


# Backward-compat alias.
def _bone_polygon_from_polylines(
    bone_lines: list[list[tuple[float, float]]],
) -> list[tuple[float, float]] | None:
    polys = _bone_polygons_from_polylines(bone_lines)
    if not polys:
        return None
    return max(polys, key=lambda p: len(p))


# ---------------------------------------------------------------------------
# YOLO output
# ---------------------------------------------------------------------------


YoloTarget = Literal["tooth_detect", "tooth_seg", "bone_seg"]


def _yolo_detection_row(
    label: int, bbox: tuple[float, float, float, float], img_w: int, img_h: int
) -> str:
    """One YOLO detection row: ``class cx cy w h`` normalized to [0,1]."""
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2) / img_w
    cy = 0.5 * (y1 + y2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # class id: tooth-detect uses 0=single, 1=double (match upstream's
    # 2-class detector config). Map our 1/2 -> 0/1.
    cls = label - 1
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _yolo_seg_row(
    cls: int, polygon: list[tuple[float, float]], img_w: int, img_h: int
) -> str | None:
    """One YOLO segmentation row: ``class x1 y1 x2 y2 ...`` normalized."""
    if len(polygon) < 3:
        return None
    coords: list[str] = [str(cls)]
    for x, y in polygon:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return " ".join(coords)


def _write_yolo_dataset_yaml(
    output_root: Path,
    target: YoloTarget,
) -> Path:
    """Emit Ultralytics dataset.yaml. Returns the yaml path."""
    yaml_path = output_root / "dataset.yaml"
    if target == "tooth_detect":
        names_line = "names:\n  0: single\n  1: double\n"
        nc = 2
    elif target == "tooth_seg":
        names_line = "names:\n  0: tooth\n"
        nc = 1
    else:  # bone_seg
        names_line = "names:\n  0: bone\n"
        nc = 1
    yaml_path.write_text(
        f"# Auto-generated by denpar_adapter.build_yolo_dataset (target={target}).\n"
        f"path: {output_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {nc}\n"
        f"{names_line}",
        encoding="utf-8",
    )
    return yaml_path


def build_yolo_dataset(
    denpar_root: Path,
    output_root: Path,
    target: YoloTarget,
) -> Path:
    """Materialize a YOLO-format dataset for one target.

    Args:
        denpar_root: Path containing ``Dataset/`` (unzipped Zenodo
            record 16645076).
        output_root: Where to write the YOLO dataset. Existing files
            are overwritten; idempotent re-runs are safe.
        target: One of ``"tooth_detect"`` (2-class detection),
            ``"tooth_seg"`` (1-class instance segmentation), or
            ``"bone_seg"`` (1-class bone segmentation).

    Returns:
        Path to the generated ``dataset.yaml``.
    """
    denpar_root = Path(denpar_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (output_root / sub / split).mkdir(parents=True, exist_ok=True)

    for v3_split, lc_split in _V3_SPLITS.items():
        split_root = _split_dir(denpar_root, v3_split)
        images_dir = split_root / _SUB_IMAGES
        out_images = output_root / "images" / lc_split
        out_labels = output_root / "labels" / lc_split

        stems = _image_ids_for_split(split_root)
        for stem in stems:
            src_img = images_dir / f"{stem}.jpg"
            dst_img = out_images / f"{stem}.jpg"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            img_w, img_h = _load_image_size(images_dir, stem)
            rows: list[str] = []

            if target in ("tooth_detect", "tooth_seg"):
                extracted = _extract_per_tooth(split_root, stem)
                if extracted is None:
                    # Write empty label file — YOLO accepts background-only.
                    (out_labels / f"{stem}.txt").write_text("")
                    continue
                bboxes, _kps, labels = extracted

                if target == "tooth_detect":
                    for bbox, label in zip(bboxes, labels):
                        rows.append(_yolo_detection_row(label, bbox, img_w, img_h))
                else:  # tooth_seg
                    polygons = _mask_polygons_for_image(split_root, stem, len(bboxes))
                    if polygons is None:
                        (out_labels / f"{stem}.txt").write_text("")
                        continue
                    for poly in polygons:
                        row = _yolo_seg_row(0, poly, img_w, img_h)
                        if row is not None:
                            rows.append(row)

            else:  # bone_seg
                # v3 polylines often have x-disjoint strips (one per
                # bone-crest segment where bone is annotatable). Emit
                # each strip as a separate YOLO-seg instance so the
                # model learns multi-strip bone topology.
                polys = _bone_polygons_for_image(split_root, stem)
                for poly in polys:
                    row = _yolo_seg_row(0, poly, img_w, img_h)
                    if row is not None:
                        rows.append(row)

            (out_labels / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))

        _LOG.info(
            "yolo:%s split=%s wrote %d images + labels", target, lc_split, len(stems)
        )

    return _write_yolo_dataset_yaml(output_root, target)


# ---------------------------------------------------------------------------
# COCO-keypoints output
# ---------------------------------------------------------------------------


CocoLandmark = Literal["cej", "bone", "apex"]


def _coco_categories() -> list[dict]:
    """COCO categories for the keypoint dataset.

    Two object categories (single + double rooted), matching upstream's
    ``num_classes=3`` (background is implicit in COCO). Each declares
    the full 6-keypoint schema; the trainer's `CocoKeypointSlice`
    selects indices per ``landmark``.
    """
    keypoints_schema = ["cej_l", "cej_r", "bone_l", "bone_r", "apex_a", "apex_b"]
    skeleton: list[list[int]] = []  # 1-indexed pairs; none defined.
    return [
        {
            "id": _CAT_SINGLE,
            "name": "tooth_single",
            "supercategory": "tooth",
            "keypoints": keypoints_schema,
            "skeleton": skeleton,
        },
        {
            "id": _CAT_DOUBLE,
            "name": "tooth_double",
            "supercategory": "tooth",
            "keypoints": keypoints_schema,
            "skeleton": skeleton,
        },
    ]


def _coco_split_payload(denpar_root: Path, v3_split: str) -> dict:
    """Build the COCO JSON dict for one split."""
    split_root = _split_dir(denpar_root, v3_split)
    images_dir = split_root / _SUB_IMAGES
    stems = _image_ids_for_split(split_root)

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for img_id_int, stem in enumerate(stems, start=1):
        img_w, img_h = _load_image_size(images_dir, stem)
        images.append(
            {
                "id": img_id_int,
                "file_name": f"{stem}.jpg",
                "width": img_w,
                "height": img_h,
            }
        )
        extracted = _extract_per_tooth(split_root, stem)
        if extracted is None:
            continue
        bboxes, kps_flat, labels = extracted
        for bbox, flat, label in zip(bboxes, kps_flat, labels):
            x1, y1, x2, y2 = bbox
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            num_kp_present = sum(1 for v in flat[2::3] if v > 0)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id_int,
                    "category_id": int(label),
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "keypoints": [float(v) for v in flat],
                    "num_keypoints": int(num_kp_present),
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": (
                "DenPAR v3 → COCO-keypoints adapter output. "
                "6 keypoints per tooth: cej_l, cej_r, bone_l, bone_r, "
                "apex_a, apex_b. Slice by landmark in the trainer."
            ),
            "version": "1.0",
            "source_split": v3_split,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": _coco_categories(),
    }


def build_coco_keypoints(
    denpar_root: Path,
    output_root: Path,
    landmark: CocoLandmark,
) -> Path:
    """Materialize a COCO-keypoint dataset for one landmark type.

    The emitted COCO file contains the **full** 6-keypoint schema per
    annotation. The trainer (``training/keypoints.py::CocoKeypointSlice``)
    slices to the active landmark at load time. We still take
    ``landmark`` as a parameter so the layout matches the trainer's
    expectation of split-rooted dataset directories (one per landmark
    when downstream tooling wants to keep them separate); the file
    contents are identical across landmarks for a given split. This
    is intentional — keeps the heuristic pairing logic in one place.

    Output layout (matches ``training/keypoints.py``):
        <output_root>/
          train/images/*.jpg
          train/annotations.json
          val/images/*.jpg
          val/annotations.json
          test/images/*.jpg
          test/annotations.json

    Args:
        denpar_root: DenPAR v3 root (containing ``Dataset/``).
        output_root: Target directory (e.g. ``data/denpar/prepared/keypoints``).
        landmark: One of ``"cej"``, ``"bone"``, ``"apex"``. Currently
            only affects naming / log messages; the JSON contains all
            three landmark groups.

    Returns:
        Path to the train-split annotations JSON (the canonical
        return).
    """
    denpar_root = Path(denpar_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_ann: Path | None = None
    for v3_split, lc_split in _V3_SPLITS.items():
        split_root = _split_dir(denpar_root, v3_split)
        images_src = split_root / _SUB_IMAGES
        out_split = output_root / lc_split
        out_images = out_split / "images"
        out_images.mkdir(parents=True, exist_ok=True)

        # Copy images.
        for src in sorted(images_src.glob("*.jpg")):
            dst = out_images / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

        payload = _coco_split_payload(denpar_root, v3_split)
        ann_path = out_split / "annotations.json"
        with ann_path.open("w") as fh:
            json.dump(payload, fh)
        _LOG.info(
            "coco-keypoints:%s split=%s wrote %d images / %d annotations",
            landmark,
            lc_split,
            len(payload["images"]),
            len(payload["annotations"]),
        )
        if lc_split == "train":
            train_ann = ann_path

    assert train_ann is not None, "Training split must produce annotations.json"
    return train_ann
