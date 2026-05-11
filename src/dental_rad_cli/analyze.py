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
from typing import Any, Dict, List, Optional, Tuple

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
    "caries": "caries.pt",
}


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

    def _load_kprcnn(self, key: str) -> Any:
        import torch  # local import — heavy dep

        path = self._weight_path(key)
        # State-dict-only load is the v0 convention (methodology brief
        # gotcha #14). Subagent B writes the matching save shape.
        state = torch.load(str(path), map_location="cpu")
        from dental_rad_cli.training.preprocess import (  # type: ignore
            build_keypoint_rcnn,
        )
        model = build_keypoint_rcnn(num_keypoints=2)
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
# Stub inference stages (v0 scaffolding)
# ---------------------------------------------------------------------------
#
# v0 ships the orchestration shape. Each stage below has a signature
# that future model integration (Subagent B's training output + a
# follow-up wiring task) will populate. For the v0 dry-run pathway and
# the unit tests, we provide a synthetic-results builder.

def _run_tooth_detection(bundle: ModelBundle, rgb: Any) -> List[Dict[str, Any]]:
    """Return list of tooth detections. v0: not wired."""
    # Placeholder until weights ship + integration task wires the model.
    # Implementation note: call bundle.get_tooth_detect()(rgb) and parse
    # the Ultralytics result object.
    return []


def _run_keypoint_passes(
    bundle: ModelBundle,
    rgb_clahe: Any,
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run the three keypoint passes (CEJ / bone-crest / apex)."""
    return []


def _run_segmentation(
    bundle: ModelBundle,
    rgb: Any,
) -> Tuple[List[Any], List[Any]]:
    """Run tooth + bone segmentation. Returns (tooth_polys, bone_polys)."""
    return [], []


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


def _build_findings_from_stages(
    detections: List[Dict[str, Any]],
    keypoints: List[Dict[str, Any]],
    tooth_polys: List[Any],
    bone_polys: List[Any],
    caries: List[CariesFinding],
) -> Tuple[List[ToothFinding], Summary]:
    """Compose rule-layer outputs into ToothFindings + Summary.

    Imports the rule-layer modules lazily so a missing rule module
    (e.g. while Subagent C is still landing files) does not break
    dry-run mode.
    """
    # Lazy imports — these modules may not exist yet at the time the
    # dry-run path or unit tests run.
    try:
        from dental_rad_cli.pipeline import severity  # noqa: F401
    except ImportError:
        pass
    # Future: invoke severity / pattern / aggregate / jaw_classify
    # against the stage outputs. v0 returns an empty tree.
    return [], Summary()


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
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
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
    out_path.write_text(render_note(result))
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

        detections = _run_tooth_detection(bundle, rgb)
        keypoints = _run_keypoint_passes(bundle, rgb_clahe, detections)
        tooth_polys, bone_polys = _run_segmentation(bundle, rgb)
        caries = _run_caries_detection(bundle, rgb, detections)

        teeth, summary = _build_findings_from_stages(
            detections, keypoints, tooth_polys, bone_polys, caries,
        )

        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        image_meta = ImageMeta(
            path=image_path.name,
            width=width,
            height=height,
            type="unknown",
        )
        metadata = Metadata(
            models=bundle.model_versions(),
            runtime_seconds=time.perf_counter() - started,
            device="cpu",
            schema_version=SCHEMA_VERSION,
            dry_run=False,
        )
        result = AnalysisResult(
            image=image_meta,
            teeth=teeth,
            summary=summary,
            low_confidence_findings=[],
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
