"""Inference orchestrator for ``dental-rad-cli``.

Single entrypoint: :func:`analyze`. Composes the trained-model stack
(tooth detect → 3 keypoint passes → tooth/bone segmentation) into one
``AnalysisResult``, optionally writing JSON, an annotated PNG, and a
note-draft text file as side effects.

CLAHE preprocessing
-------------------

The keypoint R-CNN models in this pipeline were trained with CLAHE as
the **only** image augmentation, applied at both train and val time
(see methodology brief §1.2). The constants are::

    clipLimit     = 40.0
    tileGridSize  = (8, 8)

These MUST be applied verbatim at inference. The YOLO models (tooth
detect + segmentation) were trained on raw RGB and do **not** want
CLAHE. The orchestrator therefore keeps two image surfaces in memory:
the raw RGB for YOLO, the CLAHE-enhanced RGB for keypoint R-CNN.

Module lifecycle
----------------

Heavy weights are loaded lazily through :class:`ModelBundle`. The
bundle caches loaded models so repeated ``analyze()`` calls (e.g. a CLI
invocation against a glob of images) pay the load cost once. The
bundle also encapsulates the "weights/ missing" failure mode so the
CLI can exit cleanly with a useful message.

The caries pathway is reserved for v0.5; this module exposes the
integration point as a ``_run_caries_detection`` stub but does not
invoke it in v0.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from dental_rad_cli.schema import (
    AnalysisResult,
    CariesFinding,
    ImageMeta,
    Metadata,
    SCHEMA_VERSION,
    Summary,
    ToothFinding,
    ToothKeypointsFull,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLAHE constants — MUST match training. Do not edit without retraining.
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT: float = 40.0
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)

# Weights filenames the bundle expects under ``weights/``.
WEIGHTS_FILES: Dict[str, str] = {
    "tooth_detect": "tooth_detect.pt",
    "keypoint_cej": "keypoint_cej.pt",
    "keypoint_bone": "keypoint_bone.pt",
    "keypoint_apex": "keypoint_apex.pt",
    "segmentation_tooth": "segmentation_tooth.pt",
    "segmentation_bone": "segmentation_bone.pt",
    "segmentation_cej": "segmentation_cej.pt",
    "caries": "caries.pt",
}

# v2 CEJ polyline pivot: when the trained polyline model has at least
# this many confident predictions, route CEJ → Family A mm math
# (apex-free). Below threshold, fall back to the v1 keypoint pathway
# OR emit "low_model_confidence" findings if both fail. See
# output/training-evidence/2026-05-12-karpathy-findings.md for the
# stratification that picked this threshold (131/200 images at 100%
# success on the kept set).
CEJ_POLYLINE_CONF_THRESHOLD: float = 0.5


class WeightsNotFoundError(FileNotFoundError):
    """Raised when ``weights/`` is missing or a specific weight is absent.

    The CLI catches this and prints a one-line install hint, then exits
    with code 2.
    """


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Lazy-loading container for the six trained models.

    Instances are intended to be reused across multiple ``analyze()``
    calls. Each ``get_*`` method loads the corresponding weight file on
    first call and caches the result.

    The class deliberately does NOT import torch / ultralytics at
    module top. Tests can construct a bundle and exercise its dry-run
    pathway without the ML stack installed.
    """

    weights_dir: Path

    # Cached loaded models. ``Any`` because torch/ultralytics types
    # vary by install and we keep this module import-light.
    _tooth_detect: Optional[Any] = None
    _keypoint_cej: Optional[Any] = None
    _keypoint_bone: Optional[Any] = None
    _keypoint_apex: Optional[Any] = None
    _segmentation_tooth: Optional[Any] = None
    _segmentation_bone: Optional[Any] = None
    _caries: Optional[Any] = None

    def __post_init__(self) -> None:
        self.weights_dir = Path(self.weights_dir)

    def _weight_path(self, key: str) -> Path:
        """Resolve and validate a weight file under ``weights_dir``."""
        if not self.weights_dir.exists():
            raise WeightsNotFoundError(str(self.weights_dir))
        filename = WEIGHTS_FILES[key]
        path = self.weights_dir / filename
        if not path.exists():
            raise WeightsNotFoundError(str(path))
        return path

    def model_versions(self) -> Dict[str, str]:
        """Best-effort version tags for each weight (filename stem)."""
        if not self.weights_dir.exists():
            return {}
        out: Dict[str, str] = {}
        for key, filename in WEIGHTS_FILES.items():
            p = self.weights_dir / filename
            if p.exists():
                out[key] = p.stem
        return out

    # --- YOLO models (raw RGB) -----------------------------------------

    def get_tooth_detect(self) -> Any:
        if self._tooth_detect is None:
            from ultralytics import YOLO  # local import — heavy dep
            self._tooth_detect = YOLO(str(self._weight_path("tooth_detect")))
        return self._tooth_detect

    def get_segmentation_tooth(self) -> Any:
        if self._segmentation_tooth is None:
            from ultralytics import YOLO
            self._segmentation_tooth = YOLO(str(self._weight_path("segmentation_tooth")))
        return self._segmentation_tooth

    def get_segmentation_bone(self) -> Any:
        if self._segmentation_bone is None:
            from ultralytics import YOLO
            self._segmentation_bone = YOLO(str(self._weight_path("segmentation_bone")))
        return self._segmentation_bone

    _segmentation_cej: Optional[Any] = None

    def get_segmentation_cej(self) -> Any:
        if self._segmentation_cej is None:
            from ultralytics import YOLO
            self._segmentation_cej = YOLO(str(self._weight_path("segmentation_cej")))
        return self._segmentation_cej

    def segmentation_cej_weights_path(self) -> Optional[Path]:
        """Return the CEJ-polyline weights path if present, else None.

        Same graceful pattern as caries — when the polyline model hasn't
        been trained yet (or has been intentionally omitted), the
        orchestrator falls back to the v1 keypoint pathway.
        """
        try:
            return self._weight_path("segmentation_cej")
        except WeightsNotFoundError:
            return None

    def get_caries(self) -> Any:
        if self._caries is None:
            from ultralytics import YOLO
            self._caries = YOLO(str(self._weight_path("caries")))
        return self._caries

    def caries_weights_path(self) -> Optional[Path]:
        """Return the caries weights path if present, else None.

        Caries detection is graceful: when weights are absent (e.g. an
        old training run that predates caries), the orchestrator skips
        the stage rather than raising. The other stages remain strict.
        """
        try:
            return self._weight_path("caries")
        except WeightsNotFoundError:
            return None

    # --- Keypoint R-CNN models (CLAHE-enhanced RGB) --------------------

    # Default num_keypoints per landmark — must match training-time slice
    # (see ``training.keypoints._LANDMARK_NUM_KEYPOINTS``).
    _LANDMARK_DEFAULT_KP: ClassVar[Dict[str, int]] = {
        "keypoint_cej": 2,
        "keypoint_bone": 2,
        "keypoint_apex": 1,
    }

    def _load_kprcnn(self, key: str) -> Any:
        import torch  # local import — heavy dep

        path = self._weight_path(key)
        # The trainer (``training.keypoints.train``) saves a wrapper dict
        # ``{state_dict, num_keypoints, num_classes, landmark, best_val_loss}``.
        # Older runs may have saved a bare ``state_dict``; both are handled.
        payload = torch.load(str(path), map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
            num_keypoints = int(
                payload.get("num_keypoints", self._LANDMARK_DEFAULT_KP[key])
            )
        else:
            state = payload
            num_keypoints = self._LANDMARK_DEFAULT_KP[key]

        # Reuse the training-side model builder so the architecture
        # matches the saved state_dict by construction.
        from dental_rad_cli.training.keypoints import (  # type: ignore
            _build_model as build_keypoint_rcnn,
        )
        model = build_keypoint_rcnn(num_keypoints=num_keypoints)
        model.load_state_dict(state)
        model.eval()
        return model

    def get_keypoint_cej(self) -> Any:
        if self._keypoint_cej is None:
            self._keypoint_cej = self._load_kprcnn("keypoint_cej")
        return self._keypoint_cej

    def get_keypoint_bone(self) -> Any:
        if self._keypoint_bone is None:
            self._keypoint_bone = self._load_kprcnn("keypoint_bone")
        return self._keypoint_bone

    def get_keypoint_apex(self) -> Any:
        if self._keypoint_apex is None:
            self._keypoint_apex = self._load_kprcnn("keypoint_apex")
        return self._keypoint_apex


# ---------------------------------------------------------------------------
# Image I/O + preprocessing
# ---------------------------------------------------------------------------

def _load_image_rgb(image_path: Path) -> Any:
    """Read an image from disk and return an RGB numpy array."""
    import cv2  # local import

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def apply_clahe(rgb_image: Any) -> Any:
    """Apply CLAHE preprocessing for the keypoint R-CNN pathway.

    Uses the training constants ``clipLimit=40.0`` / ``tileGridSize=(8,
    8)``. CLAHE is a per-channel luminance transform; we operate on the
    L channel of LAB to avoid color drift, then return RGB.
    """
    import cv2

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """Return ``"cuda"`` / ``"mps"`` / ``"cpu"`` per available accelerator.

    The keypoint R-CNN models honor this via ``.to(device)`` (Task 6);
    Ultralytics models receive it as a ``device=`` kwarg.
    """
    try:
        import torch  # local import — heavy dep
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # mps may exist as an attr but report unavailable on non-Apple-Silicon.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Stage wirings
# ---------------------------------------------------------------------------


def _bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Axis-aligned IoU between two ``(x1, y1, x2, y2)`` bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _to_numpy(tensor_like: Any) -> Any:
    """Convert a torch tensor (or already-numpy) to numpy without importing torch."""
    import numpy as np

    if hasattr(tensor_like, "detach"):
        return tensor_like.detach().cpu().numpy()
    if hasattr(tensor_like, "cpu"):
        return tensor_like.cpu().numpy()
    if hasattr(tensor_like, "numpy"):
        return tensor_like.numpy()
    return np.asarray(tensor_like)


def _run_tooth_detection(
    bundle: ModelBundle,
    rgb: Any,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run YOLO tooth detection; return a list of detection dicts.

    Each dict carries::

        {
            "bbox": (x1, y1, x2, y2),
            "confidence": float,
            "root_class": "single" | "double" | "unknown",
            "fdi": str,   # geometric index (left-to-right by bbox-center x)
        }

    FDI numbering note: we do NOT attempt true ISO-3950 numbering — that
    requires anatomy reasoning the model does not provide. The string
    indices are opaque IDs for the rule layer (verified: aggregate's
    quadrant logic gracefully drops non-permanent-FDI strings).
    """
    model = bundle.get_tooth_detect()
    results = model.predict(
        rgb, conf=0.5, iou=0.45, verbose=False, device=device,
    )
    if not results:
        return []
    res0 = results[0]
    boxes = getattr(res0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls).astype(int)

    raw: List[Dict[str, Any]] = []
    # Trained model class map (verified 2026-05-12 against weights/tooth_detect.pt):
    #   {0: 'single', 1: 'double'} — nc=2. The upstream paper described
    #   an nc=3 (single / double / background) variant; our trained
    #   weights do not include the background class. Unknown class IDs
    #   are skipped defensively in case a future re-train adds them.
    for i in range(len(xyxy)):
        cls_id = int(classes[i])
        if cls_id == 0:
            root_class: str = "single"
        elif cls_id == 1:
            root_class = "double"
        else:
            # Unknown class — skip rather than guess.
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        raw.append(
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confs[i]),
                "root_class": root_class,
            }
        )

    # Geometric FDI assignment: number left-to-right by bbox-center x.
    raw.sort(key=lambda d: 0.5 * (d["bbox"][0] + d["bbox"][2]))
    for i, det in enumerate(raw, start=1):
        det["fdi"] = str(i)
    return raw


def _run_keypoint_passes(
    bundle: ModelBundle,
    rgb_clahe: Any,
    detections: List[Dict[str, Any]],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run CEJ / bone / apex Keypoint-RCNN passes; pair to tooth detections.

    For each tooth in ``detections``, finds the highest-IoU predicted
    instance from each landmark model (IoU threshold 0.3) and extracts
    its keypoints.

    Returns one dict per tooth, in the same order as ``detections``::

        {
            "fdi": str,
            "cej": [(x,y), (x,y)] | None,        # 2 keypoints
            "bone_crest": [(x,y), (x,y)] | None, # 2 keypoints
            "apex": (x, y) | None,                # 1 keypoint
        }
    """
    if not detections:
        return []

    import torch  # local heavy import

    landmark_keys = {
        "cej": ("keypoint_cej", bundle.get_keypoint_cej),
        "bone": ("keypoint_bone", bundle.get_keypoint_bone),
        "apex": ("keypoint_apex", bundle.get_keypoint_apex),
    }

    # rgb_clahe is HWC uint8; convert to CHW float32 in [0, 1] (matches
    # training-time transform in ``training.keypoints.CocoKeypointSlice``).
    tensor = (
        torch.from_numpy(rgb_clahe).permute(2, 0, 1).float() / 255.0
    ).to(device)
    batch = [tensor]

    # For each landmark, run inference and capture (bbox, keypoints) per
    # predicted instance.
    per_landmark_preds: Dict[str, List[Dict[str, Any]]] = {}
    for landmark, (_wkey, getter) in landmark_keys.items():
        model = getter()
        model = model.to(device)
        with torch.no_grad():
            outs = model(batch)
        if not outs:
            per_landmark_preds[landmark] = []
            continue
        out0 = outs[0]
        boxes = _to_numpy(out0.get("boxes")) if "boxes" in out0 else _to_numpy(out0["boxes"])
        kps = _to_numpy(out0["keypoints"])
        scores = _to_numpy(out0["scores"]) if "scores" in out0 else None

        instances: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            bb = boxes[i]
            instances.append(
                {
                    "bbox": (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                    "keypoints": kps[i],  # shape (K, 3)
                    "score": float(scores[i]) if scores is not None else 1.0,
                }
            )
        per_landmark_preds[landmark] = instances

    iou_threshold = 0.3
    out_rows: List[Dict[str, Any]] = []
    for det in detections:
        det_bbox = det["bbox"]
        row: Dict[str, Any] = {
            "fdi": det.get("fdi", ""),
            "cej": None,
            "bone_crest": None,
            "apex": None,
        }
        for landmark in ("cej", "bone", "apex"):
            best_iou = iou_threshold
            best_inst: Optional[Dict[str, Any]] = None
            for inst in per_landmark_preds.get(landmark, []):
                iou = _bbox_iou(det_bbox, inst["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_inst = inst
            if best_inst is None:
                continue
            kp_arr = best_inst["keypoints"]  # (K, 3)
            if landmark == "cej":
                # 2 keypoints — mesial, distal (training-side slice order).
                if kp_arr.shape[0] >= 2:
                    row["cej"] = [
                        (float(kp_arr[0, 0]), float(kp_arr[0, 1])),
                        (float(kp_arr[1, 0]), float(kp_arr[1, 1])),
                    ]
            elif landmark == "bone":
                if kp_arr.shape[0] >= 2:
                    row["bone_crest"] = [
                        (float(kp_arr[0, 0]), float(kp_arr[0, 1])),
                        (float(kp_arr[1, 0]), float(kp_arr[1, 1])),
                    ]
            else:  # apex
                if kp_arr.shape[0] >= 1:
                    row["apex"] = (float(kp_arr[0, 0]), float(kp_arr[0, 1]))
        out_rows.append(row)
    return out_rows


def _run_cej_polyline(
    bundle: ModelBundle,
    rgb: Any,
    device: str = "cpu",
    conf: float = 0.25,
) -> Tuple[Optional[Any], float, int]:
    """Run the CEJ polyline-segmentation model; return union band + stats.

    Returns ``(band_mask, max_conf, n_masks)``:

    - ``band_mask``: bool ndarray of shape (H, W) — union of all
      detection masks resized to image resolution. ``None`` if the
      model is not available or produces no masks.
    - ``max_conf``: maximum prediction confidence across all detections.
      ``0.0`` if no detections.
    - ``n_masks``: number of mask instances predicted (before union).

    Per the karpathy stratification (n=200 DenPAR Testing), max_conf
    ≥ 0.5 is the threshold above which prediction-success is 100%.
    Callers check ``max_conf >= CEJ_POLYLINE_CONF_THRESHOLD`` before
    consuming the band for Family A math; below that, the v1
    keypoint pathway runs as a fallback (or both fail and the site
    gets a "low_model_confidence" entry).
    """
    if bundle.segmentation_cej_weights_path() is None:
        return None, 0.0, 0
    import cv2
    import numpy as np

    model = bundle.get_segmentation_cej()
    results = model.predict(rgb, conf=conf, verbose=False, device=device)
    if not results:
        return None, 0.0, 0
    res0 = results[0]
    masks_obj = getattr(res0, "masks", None)
    if masks_obj is None or len(masks_obj) == 0:
        return None, 0.0, 0
    h, w = rgb.shape[:2]
    masks = masks_obj.data.cpu().numpy().astype(bool)
    band = np.zeros((h, w), dtype=bool)
    for m in masks:
        m_rs = cv2.resize(
            m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        band |= m_rs
    boxes = getattr(res0, "boxes", None)
    max_conf = 0.0
    if boxes is not None and boxes.conf is not None:
        confs = boxes.conf.cpu().numpy()
        if confs.size > 0:
            max_conf = float(confs.max())
    return band, max_conf, int(len(masks))


def _run_segmentation(
    bundle: ModelBundle,
    rgb: Any,
    device: str = "cpu",
) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """Run tooth + bone segmentation; return polygons per instance.

    Each polygon is a ``list[(x, y)]`` of float pixel coords. Ultralytics
    returns ``results[0].masks.xy`` as a list of ``np.ndarray`` of shape
    ``(K, 2)`` per instance; we coerce to native tuples for downstream
    rasterization with ``cv2.fillPoly``.
    """

    def _extract(yolo_model: Any) -> List[List[Tuple[float, float]]]:
        results = yolo_model.predict(rgb, conf=0.5, verbose=False, device=device)
        if not results:
            return []
        res0 = results[0]
        masks = getattr(res0, "masks", None)
        if masks is None:
            return []
        xy = getattr(masks, "xy", None)
        if xy is None:
            return []
        out: List[List[Tuple[float, float]]] = []
        for poly in xy:
            # poly may be np.ndarray (N, 2) or already a list of pairs.
            poly_np = _to_numpy(poly)
            if poly_np.ndim != 2 or poly_np.shape[1] != 2 or len(poly_np) < 3:
                continue
            out.append([(float(p[0]), float(p[1])) for p in poly_np])
        return out

    tooth_polys = _extract(bundle.get_segmentation_tooth())
    bone_polys = _extract(bundle.get_segmentation_bone())
    return tooth_polys, bone_polys


def _run_caries_detection(
    bundle: ModelBundle,
    rgb: Any,
    detections: List[Dict[str, Any]],
) -> List[CariesFinding]:
    """Run caries inference; map results to schema CariesFinding rows.

    Behaviour:
    - If ``weights/caries.pt`` is missing → returns ``[]`` (graceful
      skip; the rest of the pipeline still runs).
    - Otherwise calls ``pipeline.caries_inference.detect_caries`` with
      lightweight ToothFinding stubs constructed from raw tooth
      detections. Only the bbox is needed for surface assignment;
      keypoints / FDI / pattern fields are placeholder.
    - If ``detections`` is empty, caries is still run on the image; all
      findings come back with ``surface="unknown"`` for the rule layer
      to route through ``low_confidence_findings``.
    """
    weights_path = bundle.caries_weights_path()
    if weights_path is None:
        return []

    from dental_rad_cli.pipeline.caries_inference import detect_caries

    tooth_stubs: List[ToothFinding] = []
    for i, det in enumerate(detections):
        bbox = det.get("bbox") if isinstance(det, dict) else None
        if bbox is None:
            continue
        tooth_stubs.append(
            ToothFinding(
                fdi=str(det.get("fdi", i)),
                universal=str(det.get("universal", i)),
                bbox=tuple(float(v) for v in bbox),
                confidence=float(det.get("confidence", 0.0)),
                keypoints=ToothKeypointsFull(),
            )
        )

    return detect_caries(rgb, weights_path, tooth_bboxes=tooth_stubs or None)


def _polygon_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Cheap arithmetic-mean centroid (good enough for "is it inside the
    tooth bbox" routing; not the area-weighted centroid).
    """
    if not poly:
        return (0.0, 0.0)
    sx = sum(p[0] for p in poly)
    sy = sum(p[1] for p in poly)
    n = float(len(poly))
    return (sx / n, sy / n)


def _bbox_contains(
    bbox: Tuple[float, float, float, float],
    point: Tuple[float, float],
) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def _rasterize_polygon(
    poly: List[Tuple[float, float]],
    shape: Tuple[int, int],
) -> Any:
    """Rasterize a polygon to a uint8 binary mask of given (H, W)."""
    import cv2
    import numpy as np

    mask = np.zeros(shape, dtype=np.uint8)
    if not poly:
        return mask
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _build_findings_from_stages(
    detections: List[Dict[str, Any]],
    keypoints: List[Dict[str, Any]],
    tooth_polys: List[List[Tuple[float, float]]],
    bone_polys: List[List[Tuple[float, float]]],
    caries: List[CariesFinding],
    image_shape: Optional[Tuple[int, int]] = None,
    cej_band: Optional[Any] = None,
    cej_band_max_conf: float = 0.0,
) -> Tuple[List[ToothFinding], Summary, List[Any]]:
    """Compose rule-layer outputs into ToothFindings + Summary.

    ``image_shape`` is ``(H, W)`` used to rasterize polygons for the
    pattern classifier. If ``None``, the pattern stage is skipped and
    each tooth's pattern stays ``"unknown"``.

    ``cej_band`` is the optional CEJ polyline-segmentation band mask
    (np.ndarray bool, shape (H, W)). When provided and
    ``cej_band_max_conf >= CEJ_POLYLINE_CONF_THRESHOLD``, the
    Family A apex-free mm pathway runs in place of the v1 keypoint +
    apex % math. When confidence is insufficient OR Family A can't
    produce a measurement (band doesn't cover bbox edges), the
    pipeline falls back to the v1 keypoint pathway.

    Returns ``(teeth, summary, low_confidence_findings)``.
    """
    from dental_rad_cli.pipeline.severity import (
        compute_bone_loss_pct, severity_tier,
    )
    from dental_rad_cli.pipeline.family_a import (
        calibrate_px_per_mm, per_tooth_family_a,
    )
    from dental_rad_cli.pipeline.pattern import classify_pattern
    from dental_rad_cli.pipeline.aggregate import (
        aap_stage, quadrant_summary,
    )
    from dental_rad_cli.pipeline.jaw_classify import classify_jaw
    from dental_rad_cli.schema import (
        BoneLossPerSite,
        BoneLossSite,
        CariesSummaryEntry,
        LowConfidenceFinding,
        ToothKeypointsFull,
        ToothWithKeypoints,
        VerticalDefect,
    )

    # v2 Family A routing decision.
    #
    # When the polyline model is available (cej_band is not None even
    # at low confidence — it was loaded and run), the polyline becomes
    # the authoritative gate: above threshold → Family A; below
    # threshold → mark as low_model_confidence and DO NOT fall back to
    # the v1 keypoint+apex pathway. The reasoning: if the CEJ polyline
    # model isn't confident on this image, the apex predictions
    # (trained on the same DenPAR distribution) are also unreliable;
    # falling back produces a wrong measurement (e.g., apex predictions
    # hug bbox-top → overestimated pct → spurious "severe" tier).
    #
    # When the polyline model is NOT loaded (legacy install / weights
    # missing), the v1 keypoint+apex pathway runs as before.
    polyline_loaded = cej_band is not None
    use_family_a = (
        polyline_loaded and cej_band_max_conf >= CEJ_POLYLINE_CONF_THRESHOLD
    )
    polyline_below_threshold = (
        polyline_loaded and cej_band_max_conf < CEJ_POLYLINE_CONF_THRESHOLD
    )
    px_per_mm: Optional[float] = None
    if use_family_a and detections:
        det_bboxes = [
            tuple(float(c) for c in d["bbox"]) for d in detections
        ]
        px_per_mm = calibrate_px_per_mm(det_bboxes)

    kp_by_fdi: Dict[str, Dict[str, Any]] = {
        kp.get("fdi", ""): kp for kp in keypoints if kp.get("fdi") is not None
    }

    # Group caries findings by their parent tooth — caries_inference
    # already assigns a surface; we re-route by matching surface against
    # the geometric "closest tooth" heuristic the inference helper used.
    # The simplest correct mapping is: for each caries, find the tooth
    # whose bbox contains the caries-bbox center; the tooth's FDI string
    # becomes the routing key.
    caries_by_fdi: Dict[str, List[CariesFinding]] = {}
    unrouted_caries: List[CariesFinding] = []
    for c in caries:
        if c.bbox is None:
            unrouted_caries.append(c)
            continue
        ccx = 0.5 * (c.bbox[0] + c.bbox[2])
        ccy = 0.5 * (c.bbox[1] + c.bbox[3])
        matched_fdi: Optional[str] = None
        for det in detections:
            if _bbox_contains(det["bbox"], (ccx, ccy)):
                matched_fdi = det["fdi"]
                break
        if matched_fdi is None:
            unrouted_caries.append(c)
        else:
            caries_by_fdi.setdefault(matched_fdi, []).append(c)

    low_confidence: List[Any] = []
    teeth: List[ToothFinding] = []
    all_sites: List[BoneLossSite] = []
    tooth_for_jaw: List[ToothWithKeypoints] = []
    vertical_defects: List[VerticalDefect] = []
    summary_caries: List[CariesSummaryEntry] = []

    # Pre-rasterize the bone mask once — pattern uses it per tooth.
    bone_mask = None
    if image_shape is not None and bone_polys:
        import numpy as np

        bone_mask = np.zeros(image_shape, dtype=np.uint8)
        for bp in bone_polys:
            bone_mask |= _rasterize_polygon(bp, image_shape)

    for det in detections:
        fdi = det["fdi"]
        kp_row = kp_by_fdi.get(fdi, {})

        cej_pair = kp_row.get("cej")        # list[(x,y), (x,y)] | None
        bone_pair = kp_row.get("bone_crest")
        apex_pt = kp_row.get("apex")        # (x, y) | None

        # Per-site bone-loss math.
        mesial_site: Optional[BoneLossSite] = None
        distal_site: Optional[BoneLossSite] = None
        family_a_emitted = False

        # v2 Family A pathway: apex-free mm CEJ→bone-crest. Runs first
        # when the polyline model has confidence and we have a bone
        # mask to query.
        family_a_positions: Optional[Dict[str, Tuple[float, float]]] = None
        if (
            use_family_a
            and bone_mask is not None
            and px_per_mm is not None
            and px_per_mm > 0
        ):
            from dental_rad_cli.pipeline.family_a import band_centerline_y_at_x

            mesial_site, distal_site = per_tooth_family_a(
                cej_band, bone_mask, det["bbox"], px_per_mm,
            )
            family_a_emitted = (
                mesial_site.mm_estimate is not None
                or distal_site.mm_estimate is not None
            )
            # Synthesize landmark positions from the band centerlines so
            # the render layer (which draws CEJ→bone-crest segments
            # between two points) can show mm sites without keypoint
            # predictions. None where the band didn't cross that bbox
            # edge — render layer falls back gracefully.
            bx1, _, bx2, _ = det["bbox"]
            cej_m_y = band_centerline_y_at_x(cej_band, bx1)
            cej_d_y = band_centerline_y_at_x(cej_band, bx2)
            bone_m_y = band_centerline_y_at_x(bone_mask, bx1)
            bone_d_y = band_centerline_y_at_x(bone_mask, bx2)
            family_a_positions = {
                "cej_mesial": (bx1, cej_m_y) if cej_m_y is not None else None,
                "cej_distal": (bx2, cej_d_y) if cej_d_y is not None else None,
                "bone_mesial": (bx1, bone_m_y) if bone_m_y is not None else None,
                "bone_distal": (bx2, bone_d_y) if bone_d_y is not None else None,
            }
            if family_a_emitted:
                if mesial_site.tier is not None:
                    all_sites.append(mesial_site)
                if distal_site.tier is not None:
                    all_sites.append(distal_site)
                for site, surface in (
                    (mesial_site, "mesial"), (distal_site, "distal")
                ):
                    if site.mm_estimate is None:
                        low_confidence.append(
                            LowConfidenceFinding(
                                type="bone_loss",
                                tooth=fdi,
                                surface=surface,
                                reason=site.reason or "incomputable",
                            )
                        )

        # v1 keypoint pathway: legacy fallback. Only runs when the
        # polyline model is NOT loaded at all (legacy install). When
        # polyline IS loaded but below confidence threshold, prefer
        # "I don't know" over a wrong apex-pathway measurement.
        if not family_a_emitted and not polyline_below_threshold:
            if (
                cej_pair is not None
                and bone_pair is not None
                and apex_pt is not None
            ):
                cej_m, cej_d = cej_pair[0], cej_pair[1]
                bone_m, bone_d = bone_pair[0], bone_pair[1]
                pct_m = compute_bone_loss_pct(cej_m, bone_m, apex_pt)
                pct_d = compute_bone_loss_pct(cej_d, bone_d, apex_pt)
                mesial_site = BoneLossSite(
                    pct=pct_m,
                    tier=severity_tier(pct_m),
                    reason=None if pct_m is not None else "incomputable",
                )
                distal_site = BoneLossSite(
                    pct=pct_d,
                    tier=severity_tier(pct_d),
                    reason=None if pct_d is not None else "incomputable",
                )
                all_sites.append(mesial_site)
                all_sites.append(distal_site)
                if pct_m is None:
                    low_confidence.append(
                        LowConfidenceFinding(
                            type="bone_loss",
                            tooth=fdi,
                            surface="mesial",
                            reason="incomputable",
                        )
                    )
                if pct_d is None:
                    low_confidence.append(
                        LowConfidenceFinding(
                            type="bone_loss",
                            tooth=fdi,
                            surface="distal",
                            reason="incomputable",
                        )
                    )
            else:
                # Neither Family A nor v1 keypoints produced a usable
                # measurement. Legacy keypoint pathway preserves the
                # "keypoint / missing_landmarks" code so existing
                # downstream consumers + tests continue to work.
                # When polyline gating is the cause, callers handle
                # that case via the `elif polyline_below_threshold`
                # branch below.
                low_confidence.append(
                    LowConfidenceFinding(
                        type="keypoint",
                        tooth=fdi,
                        reason="missing_landmarks",
                    )
                )
        elif polyline_below_threshold and not family_a_emitted:
            # Polyline model is loaded but didn't have enough confidence
            # on this image. Skip the apex pathway entirely (it would
            # produce an unreliable measurement) and emit "manual
            # review recommended" per the karpathy ship strategy.
            low_confidence.append(
                LowConfidenceFinding(
                    type="bone_loss",
                    tooth=fdi,
                    reason="low_model_confidence",
                )
            )

        # Build ToothKeypointsFull from the matched per-tooth landmarks.
        # In Family A mode we synthesize landmark positions from the
        # band centerlines so the render layer can draw CEJ→bone-crest
        # segments without keypoint predictions. Otherwise fall back to
        # the v1 keypoint pairs. Per-keypoint confidence is 1.0 when
        # present; absence is None.
        if family_a_positions is not None:
            kp_full = ToothKeypointsFull(
                cej_mesial=(*family_a_positions["cej_mesial"], cej_band_max_conf)
                    if family_a_positions["cej_mesial"] is not None else None,
                cej_distal=(*family_a_positions["cej_distal"], cej_band_max_conf)
                    if family_a_positions["cej_distal"] is not None else None,
                bone_crest_mesial=(*family_a_positions["bone_mesial"], 1.0)
                    if family_a_positions["bone_mesial"] is not None else None,
                bone_crest_distal=(*family_a_positions["bone_distal"], 1.0)
                    if family_a_positions["bone_distal"] is not None else None,
                apex=None,  # apex-free in Family A mode
            )
        else:
            kp_full = ToothKeypointsFull(
                cej_mesial=(cej_pair[0][0], cej_pair[0][1], 1.0) if cej_pair else None,
                cej_distal=(cej_pair[1][0], cej_pair[1][1], 1.0) if cej_pair else None,
                bone_crest_mesial=(bone_pair[0][0], bone_pair[0][1], 1.0) if bone_pair else None,
                bone_crest_distal=(bone_pair[1][0], bone_pair[1][1], 1.0) if bone_pair else None,
                apex=(apex_pt[0], apex_pt[1], 1.0) if apex_pt else None,
            )

        # Pattern classification — requires tooth-mask + bone-mask + at
        # least one CEJ + bone-crest landmark.
        pattern_label = "unknown"
        if (
            image_shape is not None
            and bone_mask is not None
            and cej_pair is not None
            and bone_pair is not None
            and tooth_polys
        ):
            # Find this tooth's polygon: bbox-center containment first;
            # nearest-by-centroid fallback dropped (would risk pairing
            # the wrong tooth — better to emit "unknown").
            cx = 0.5 * (det["bbox"][0] + det["bbox"][2])
            cy = 0.5 * (det["bbox"][1] + det["bbox"][3])
            matched_poly: Optional[List[Tuple[float, float]]] = None
            for tp in tooth_polys:
                cen = _polygon_centroid(tp)
                if _bbox_contains(det["bbox"], cen):
                    matched_poly = tp
                    break
            if matched_poly is not None:
                tooth_mask = _rasterize_polygon(matched_poly, image_shape)
                pattern_label = classify_pattern(
                    tooth_mask=tooth_mask,
                    bone_mask=bone_mask,
                    cej_landmarks=list(cej_pair) if cej_pair else [],
                    bone_crest_landmarks=list(bone_pair) if bone_pair else [],
                )

        # Caries attached to this tooth.
        tooth_caries = caries_by_fdi.get(fdi, [])
        for c in tooth_caries:
            summary_caries.append(
                CariesSummaryEntry(
                    tooth=fdi,
                    surface=c.surface,
                    depth=c.depth,
                    confidence=c.confidence,
                )
            )
            if c.confidence < 0.75:
                low_confidence.append(
                    LowConfidenceFinding(
                        type="caries",
                        tooth=fdi,
                        surface=c.surface,
                        confidence=c.confidence,
                        reason="below_0.75_threshold",
                    )
                )

        tooth_finding = ToothFinding(
            fdi=fdi,
            universal=fdi,
            bbox=det["bbox"],
            confidence=det.get("confidence", 0.0),
            root_class=det.get("root_class", "unknown"),
            keypoints=kp_full,
            bone_loss=BoneLossPerSite(mesial=mesial_site, distal=distal_site),
            pattern=pattern_label,  # type: ignore[arg-type]
            caries=tooth_caries,
        )
        teeth.append(tooth_finding)

        if pattern_label == "angular_vertical":
            # Use the worst-pct site for the defect entry.
            worst_pct: Optional[float] = None
            for s in (mesial_site, distal_site):
                if s is not None and s.pct is not None:
                    if worst_pct is None or s.pct > worst_pct:
                        worst_pct = s.pct
            if worst_pct is not None:
                vertical_defects.append(
                    VerticalDefect(
                        site=f"tooth_{fdi}",
                        pct=worst_pct,
                        confidence=det.get("confidence", 0.0),
                    )
                )

        # Jaw classification input — CEJ-midpoint y and apex y.
        cej_y: Optional[float] = None
        if cej_pair is not None:
            cej_y = 0.5 * (cej_pair[0][1] + cej_pair[1][1])
        apex_y: Optional[float] = apex_pt[1] if apex_pt is not None else None
        tooth_for_jaw.append(ToothWithKeypoints(cej_y=cej_y, apex_y=apex_y))

    # Surface-level unrouted caries.
    for c in unrouted_caries:
        summary_caries.append(
            CariesSummaryEntry(
                tooth="unknown",
                surface=c.surface,
                depth=c.depth,
                confidence=c.confidence,
            )
        )
        if c.confidence < 0.75:
            low_confidence.append(
                LowConfidenceFinding(
                    type="caries",
                    surface=c.surface,
                    confidence=c.confidence,
                    reason="below_0.75_threshold",
                )
            )

    jaw = classify_jaw(tooth_for_jaw)
    stage = aap_stage(all_sites) if all_sites else "I"
    quadrants_map = quadrant_summary(teeth)
    quadrants = list(quadrants_map.values())

    # Top-level pattern string: "generalized_<dominant>" if all-same;
    # else mixed. We keep this descriptive — the rule layer downstream
    # treats this as a free-form summary token.
    if not teeth:
        bone_loss_pattern = "unknown"
    else:
        patterns = {t.pattern for t in teeth}
        if patterns == {"horizontal"}:
            bone_loss_pattern = "generalized_horizontal"
        elif "angular_vertical" in patterns and "horizontal" in patterns:
            bone_loss_pattern = "mixed"
        elif patterns == {"angular_vertical"}:
            bone_loss_pattern = "generalized_angular_vertical"
        else:
            bone_loss_pattern = "unknown"

    summary = Summary(
        bone_loss_pattern=bone_loss_pattern,
        aap_stage_estimate=stage,
        jaw_classification=jaw,
        vertical_defects=vertical_defects,
        caries_findings=summary_caries,
        quadrants=quadrants,
    )

    return teeth, summary, low_confidence


# ---------------------------------------------------------------------------
# Dry-run synthetic result builder
# ---------------------------------------------------------------------------

def _build_dry_run_result(image_path: Path) -> AnalysisResult:
    """Return a realistic AnalysisResult without invoking any model.

    Used by the CLI ``--dry-run`` flag and by the e2e wiring tests.
    Values are plausible but synthetic — never derived from any real
    image.
    """
    from dental_rad_cli.schema import (
        BoneLossPerSite,
        BoneLossSite,
        CariesSummaryEntry,
        LowConfidenceFinding,
        ToothKeypointsFull,
        VerticalDefect,
    )

    # Best-effort image dimensions — fall back to a canonical bitewing
    # size if the file is unreadable (the dry-run path must never crash
    # on a real-but-corrupt file).
    width, height = 1280, 960
    try:
        rgb = _load_image_rgb(image_path)
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
    except Exception:  # noqa: BLE001 — dry-run must never raise
        logger.debug("dry-run: could not read %s, using default dims", image_path)

    image = ImageMeta(
        path=image_path.name,
        width=width,
        height=height,
        type="bitewing",
    )

    teeth: List[ToothFinding] = [
        ToothFinding(
            fdi="30",
            universal="30",
            bbox=(640.0, 320.0, 800.0, 640.0),
            confidence=0.94,
            root_class="double",
            keypoints=ToothKeypointsFull(
                cej_mesial=(650.0, 410.0, 0.92),
                cej_distal=(790.0, 410.0, 0.91),
                bone_crest_mesial=(655.0, 470.0, 0.88),
                bone_crest_distal=(785.0, 480.0, 0.87),
                apex=(720.0, 620.0, 0.90),
            ),
            bone_loss=BoneLossPerSite(
                mesial=BoneLossSite(pct=18.0, tier="moderate"),
                distal=BoneLossSite(pct=22.0, tier="moderate"),
            ),
            pattern="horizontal",
        ),
        ToothFinding(
            fdi="19",
            universal="19",
            bbox=(420.0, 320.0, 580.0, 640.0),
            confidence=0.91,
            root_class="double",
            keypoints=ToothKeypointsFull(
                cej_mesial=(430.0, 410.0, 0.90),
                cej_distal=(570.0, 410.0, 0.89),
                bone_crest_mesial=(440.0, 510.0, 0.86),
                bone_crest_distal=(565.0, 460.0, 0.85),
                apex=(500.0, 620.0, 0.88),
            ),
            bone_loss=BoneLossPerSite(
                mesial=BoneLossSite(pct=32.0, tier="moderate"),
                distal=BoneLossSite(pct=14.0, tier="mild"),
            ),
            pattern="angular_vertical",
        ),
    ]

    summary = Summary(
        bone_loss_pattern="generalized_horizontal",
        aap_stage_estimate="II",
        jaw_classification="mandibular",
        vertical_defects=[
            VerticalDefect(site="mesial_19", pct=32.0, confidence=0.81),
        ],
        caries_findings=[
            CariesSummaryEntry(tooth="30", surface="occlusal", depth="D1", confidence=0.79),
        ],
    )

    low_confidence = [
        LowConfidenceFinding(
            type="caries",
            tooth="30",
            surface="occlusal",
            confidence=0.62,
            reason="below_0.75_threshold",
        ),
    ]

    metadata = Metadata(
        models={
            "tooth_detect": "dry-run-stub",
            "keypoint_cej": "dry-run-stub",
            "keypoint_bone": "dry-run-stub",
            "keypoint_apex": "dry-run-stub",
            "segmentation_tooth": "dry-run-stub",
            "segmentation_bone": "dry-run-stub",
        },
        runtime_seconds=0.0,
        device="cpu",
        schema_version=SCHEMA_VERSION,
        dry_run=True,
    )

    return AnalysisResult(
        image=image,
        teeth=teeth,
        summary=summary,
        low_confidence_findings=low_confidence,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Side-effect writers
# ---------------------------------------------------------------------------

def _write_json(result: AnalysisResult, out_path: Path) -> Path:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return out_path


def _write_annotated_png(
    result: AnalysisResult,
    image_path: Path,
    out_path: Path,
) -> Optional[Path]:
    """Render an annotated PNG via the rendering layer.

    Returns the output path on success, or ``None`` if rendering was
    skipped because the image could not be read (e.g. dry-run against a
    non-existent path).
    """
    try:
        from dental_rad_cli.render.annotate import render_annotated

        rgb = _load_image_rgb(image_path)
    except FileNotFoundError:
        logger.warning("annotated PNG skipped — image not readable: %s", image_path)
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return render_annotated(rgb, result, out_path)


def _write_note_draft(result: AnalysisResult, out_path: Path) -> Path:
    from dental_rad_cli.note_draft import render_note

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_note(result), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

# Module-level bundle cache for the CLI happy path. ``analyze()`` always
# accepts an explicit ``bundle`` for callers that want isolation.
_CACHED_BUNDLES: Dict[Path, ModelBundle] = {}


def _get_or_create_bundle(weights_dir: Path) -> ModelBundle:
    key = weights_dir.resolve() if weights_dir.exists() else weights_dir
    if key not in _CACHED_BUNDLES:
        _CACHED_BUNDLES[key] = ModelBundle(weights_dir=weights_dir)
    return _CACHED_BUNDLES[key]


def analyze(
    image_path: Path,
    weights_dir: Path = Path("weights/"),
    out_dir: Optional[Path] = None,
    emit_note_draft: bool = False,
    render: bool = True,
    dry_run: bool = False,
    bundle: Optional[ModelBundle] = None,
) -> AnalysisResult:
    """Run the full inference pipeline on one image.

    Parameters
    ----------
    image_path
        Path to a JPEG/PNG/TIFF radiograph.
    weights_dir
        Directory containing the six trained model weight files. Lazy
        loaded — only the models that fire are read from disk.
    out_dir
        If provided, writes ``{stem}.json`` (always), ``{stem}.annotated.png``
        (when ``render`` is true), and ``{stem}.note.txt`` (when
        ``emit_note_draft`` is true). If ``None``, no files are written
        and the return value is the only output.
    emit_note_draft
        Whether to also produce the template-rendered clinical note.
    render
        Whether to also produce the side-by-side annotated PNG.
    dry_run
        Skip all model invocation and return a synthetic ``AnalysisResult``
        with realistic dummy values. Useful for testing wiring without a
        GPU or trained weights present.
    bundle
        Optional pre-constructed :class:`ModelBundle`. Pass this to
        share loaded weights across multiple ``analyze()`` calls without
        going through the module-level cache.

    Returns
    -------
    AnalysisResult
        The structured findings. File writes (JSON / PNG / note) are
        side effects driven by the flags above.

    Raises
    ------
    WeightsNotFoundError
        If ``weights_dir`` does not exist or a required weight file is
        missing. Caught by the CLI to print a friendly install hint.
    FileNotFoundError
        If ``image_path`` is unreadable in non-dry-run mode.
    """
    image_path = Path(image_path)
    weights_dir = Path(weights_dir)
    started = time.perf_counter()

    if dry_run:
        result = _build_dry_run_result(image_path)
        if emit_note_draft:
            from dental_rad_cli.note_draft import render_note

            # Note draft is composed from the result; we rebuild with
            # the note text inlined for JSON consumers.
            result = _attach_note(result, render_note(result))
    else:
        if bundle is None:
            bundle = _get_or_create_bundle(weights_dir)

        # Preflight: weights existence check raises early with a
        # consistent error class that the CLI translates to exit code 2.
        if not weights_dir.exists():
            raise WeightsNotFoundError(str(weights_dir))

        rgb = _load_image_rgb(image_path)
        rgb_clahe = apply_clahe(rgb)

        device = _detect_device()
        height, width = int(rgb.shape[0]), int(rgb.shape[1])

        detections = _run_tooth_detection(bundle, rgb, device=device)
        keypoints = _run_keypoint_passes(bundle, rgb_clahe, detections, device=device)
        tooth_polys, bone_polys = _run_segmentation(bundle, rgb, device=device)
        cej_band, cej_band_max_conf, _n_cej_masks = _run_cej_polyline(
            bundle, rgb, device=device,
        )
        caries = _run_caries_detection(bundle, rgb, detections)

        teeth, summary, low_confidence = _build_findings_from_stages(
            detections, keypoints, tooth_polys, bone_polys, caries,
            image_shape=(height, width),
            cej_band=cej_band,
            cej_band_max_conf=cej_band_max_conf,
        )

        image_meta = ImageMeta(
            path=image_path.name,
            width=width,
            height=height,
            type="unknown",
        )
        metadata = Metadata(
            models=bundle.model_versions(),
            runtime_seconds=time.perf_counter() - started,
            device=device,
            schema_version=SCHEMA_VERSION,
            dry_run=False,
        )
        result = AnalysisResult(
            image=image_meta,
            teeth=teeth,
            summary=summary,
            low_confidence_findings=low_confidence,
            metadata=metadata,
        )
        if emit_note_draft:
            from dental_rad_cli.note_draft import render_note

            result = _attach_note(result, render_note(result))

    if out_dir is not None:
        out_dir = Path(out_dir)
        stem = image_path.stem
        _write_json(result, out_dir / f"{stem}.json")
        if render:
            _write_annotated_png(result, image_path, out_dir / f"{stem}.annotated.png")
        if emit_note_draft:
            _write_note_draft(result, out_dir / f"{stem}.note.txt")

    return result


def _attach_note(result: AnalysisResult, note: str) -> AnalysisResult:
    """Return a copy of ``result`` with ``note_draft`` set.

    Frozen dataclasses → we build a new instance rather than mutate.
    """
    return AnalysisResult(
        image=result.image,
        teeth=result.teeth,
        summary=result.summary,
        low_confidence_findings=result.low_confidence_findings,
        note_draft=note,
        metadata=result.metadata,
    )
