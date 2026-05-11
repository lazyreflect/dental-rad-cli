"""Canonical dataclass schema for `dental-rad-cli` analysis output.

This module is the **complete** schema for the JSON artifact emitted by
``analyze.analyze()``. A minimal subset previously lived here for the
rule-layer modules in ``pipeline/``; this version supersedes it and
remains backward-compatible with those imports.

Design notes
------------

- All dataclasses are ``frozen=True`` (immutable) and use ``slots=True``
  to make them cheap and to keep rule-layer functions pure. Frozen
  matters: the orchestrator builds the tree once at the end of inference
  and downstream consumers (rendering, note draft, JSON write) must not
  mutate it.
- ``to_dict()`` on every dataclass returns a plain ``dict`` shape that
  ``json.dumps`` accepts. We do NOT use ``dataclasses.asdict`` because
  it cannot represent ``None``-vs-missing-key cleanly and forces every
  optional field into the output.
- The ``low_confidence_findings`` field is a top-level *parallel*
  surface (not nested into individual teeth) on purpose: the
  doctor-facing artifact treats it as a single "things to double-check"
  list, and downstream renderers iterate it once to draw dashed
  outlines wherever the underlying finding lives.

Schema version: see ``SCHEMA_VERSION``. Bump when a non-additive change
ships (renamed field, semantic change in an existing field).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases (re-exported for the rule-layer modules)
# ---------------------------------------------------------------------------

# 2-D pixel coordinate; ``None`` signals "keypoint absent / not visible".
Point = Tuple[float, float]

# Confidence-tagged keypoint: ``(x, y, confidence)``. The model emits a
# per-detection score and we surface it verbatim so the renderer can
# draw low-confidence keypoints with a dashed outline.
ConfPoint = Tuple[float, float, float]

SeverityTier = Literal["mild", "moderate", "severe"]
BoneLossPattern = Literal["horizontal", "angular_vertical", "unknown"]
Jaw = Literal["maxillary", "mandibular"]
Quadrant = Literal["UR", "UL", "LL", "LR"]
AAPStage = Literal["I", "II", "III", "IV"]
ImageType = Literal["bitewing", "periapical", "panoramic", "unknown"]
RootClass = Literal["single", "double", "unknown"]
CariesDepth = Literal["E1", "E2", "D1", "D2", "D3"]

SCHEMA_VERSION: str = "0.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conf_point_to_list(p: Optional[ConfPoint]) -> Optional[List[float]]:
    """Serialize a ``(x, y, conf)`` triple to a JSON list (or ``None``)."""
    if p is None:
        return None
    return [float(p[0]), float(p[1]), float(p[2])]


def _bbox_to_list(b: Optional[Tuple[float, float, float, float]]) -> Optional[List[float]]:
    """Serialize an ``(x1, y1, x2, y2)`` tuple to a JSON list (or ``None``)."""
    if b is None:
        return None
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys whose value is ``None`` to keep JSON output compact.

    We keep keys whose value is an *empty* list/dict — those are
    intentional signals ("no caries found", "no defects") and removing
    them would make consumers unable to distinguish "ran and found
    none" from "did not run".
    """
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Image metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ImageMeta:
    """Metadata for the source image consumed by the pipeline."""

    path: str
    width: int
    height: int
    type: ImageType = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "width": int(self.width),
            "height": int(self.height),
            "type": self.type,
        }


# ---------------------------------------------------------------------------
# Rule-layer minimal surfaces (kept for backward compatibility with
# the existing severity/pattern/aggregate/jaw_classify modules).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToothKeypoints:
    """Reduced (midpoint) keypoints consumed by the rule layer.

    The full ``ToothFinding.keypoints`` map carries left/right pairs from
    the model. This struct holds the midpoints — the rule layer only
    needs one point per landmark.
    """

    cej: Optional[Point] = None
    bone_crest: Optional[Point] = None
    apex: Optional[Point] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "cej": list(self.cej) if self.cej else None,
            "bone_crest": list(self.bone_crest) if self.bone_crest else None,
            "apex": list(self.apex) if self.apex else None,
        })


@dataclass(frozen=True, slots=True)
class BoneLossSite:
    """Per-site (mesial/distal) bone-loss measurement.

    ``pct`` is the bone-loss percentage; ``tier`` is the AAP severity
    tier derived from ``pct``. Both may be None when keypoints are
    insufficient. ``reason`` carries a machine-readable rejection code.
    ``mm_estimate`` is a derived linear estimate (pixel→mm conversion
    requires a calibration step that v0 does not perform — left None
    until v0.5).
    """

    pct: Optional[float]
    tier: Optional[SeverityTier]
    reason: Optional[str] = None
    mm_estimate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "pct": None if self.pct is None else round(float(self.pct), 1),
            "tier": self.tier,
            "reason": self.reason,
            "mm_estimate": None if self.mm_estimate is None else round(float(self.mm_estimate), 2),
        })


@dataclass(frozen=True, slots=True)
class ToothWithKeypoints:
    """Input to ``jaw_classify.classify_jaw``.

    Holds the per-tooth CEJ-midpoint and apex-midpoint y-coordinates.
    """

    cej_y: Optional[float] = None
    apex_y: Optional[float] = None


@dataclass(frozen=True, slots=True)
class QuadrantSummary:
    """Roll-up of per-tooth findings within a quadrant."""

    quadrant: Quadrant
    n_teeth: int
    n_mild: int
    n_moderate: int
    n_severe: int
    worst_tier: Optional[SeverityTier]
    worst_pct: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "quadrant": self.quadrant,
            "n_teeth": self.n_teeth,
            "n_mild": self.n_mild,
            "n_moderate": self.n_moderate,
            "n_severe": self.n_severe,
            "worst_tier": self.worst_tier,
            "worst_pct": None if self.worst_pct is None else round(float(self.worst_pct), 1),
        })


# ---------------------------------------------------------------------------
# Per-tooth full finding (model-emitted left/right pairs preserved)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToothKeypointsFull:
    """All five keypoints per tooth, each with confidence.

    Left/right pairs for CEJ and bone-crest; a single apex point
    (single-rooted) or an additional ``apex_buccal`` for multi-rooted
    teeth (kept optional so single-rooted molars don't carry a dead
    field). All entries are ``(x, y, confidence)`` triples or ``None``
    when the model didn't emit a confident landmark.
    """

    cej_mesial: Optional[ConfPoint] = None
    cej_distal: Optional[ConfPoint] = None
    bone_crest_mesial: Optional[ConfPoint] = None
    bone_crest_distal: Optional[ConfPoint] = None
    apex: Optional[ConfPoint] = None
    apex_buccal: Optional[ConfPoint] = None  # multi-rooted only

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "cej_mesial": _conf_point_to_list(self.cej_mesial),
            "cej_distal": _conf_point_to_list(self.cej_distal),
            "bone_crest_mesial": _conf_point_to_list(self.bone_crest_mesial),
            "bone_crest_distal": _conf_point_to_list(self.bone_crest_distal),
            "apex": _conf_point_to_list(self.apex),
            "apex_buccal": _conf_point_to_list(self.apex_buccal),
        })


@dataclass(frozen=True, slots=True)
class BoneLossPerSite:
    """Mesial + distal bone-loss measurements for a single tooth."""

    mesial: Optional[BoneLossSite] = None
    distal: Optional[BoneLossSite] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "mesial": self.mesial.to_dict() if self.mesial else None,
            "distal": self.distal.to_dict() if self.distal else None,
        })


@dataclass(frozen=True, slots=True)
class CariesFinding:
    """One caries lesion on a tooth surface.

    Placeholder for v0.5 — the orchestrator does not invoke a caries
    model in v0, but the schema reserves the shape so downstream
    consumers can stabilize against it.
    """

    surface: str  # "mesial" / "distal" / "occlusal" / "MO" / etc.
    depth: CariesDepth
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "surface": self.surface,
            "depth": self.depth,
            "bbox": _bbox_to_list(self.bbox),
            "confidence": round(float(self.confidence), 3),
        })


@dataclass(frozen=True, slots=True)
class ToothFinding:
    """Full per-tooth finding: detection + keypoints + bone-loss + caries."""

    fdi: str
    universal: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    root_class: RootClass = "unknown"
    keypoints: ToothKeypointsFull = field(default_factory=ToothKeypointsFull)
    bone_loss: BoneLossPerSite = field(default_factory=BoneLossPerSite)
    pattern: BoneLossPattern = "unknown"
    caries: List[CariesFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "fdi": self.fdi,
            "universal": self.universal,
            "bbox": _bbox_to_list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "root_class": self.root_class,
            "keypoints": self.keypoints.to_dict(),
            "bone_loss": self.bone_loss.to_dict(),
            "pattern": self.pattern,
            "caries": [c.to_dict() for c in self.caries],
        })


# ---------------------------------------------------------------------------
# Summary + low-confidence surfaces
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VerticalDefect:
    """A vertical (angular) bone-loss defect surface on the summary."""

    site: str  # e.g. "mesial_19"
    pct: float
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site": self.site,
            "pct": round(float(self.pct), 1),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass(frozen=True, slots=True)
class CariesSummaryEntry:
    """A caries lesion as it appears on the summary surface."""

    tooth: str
    surface: str
    depth: CariesDepth
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tooth": self.tooth,
            "surface": self.surface,
            "depth": self.depth,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass(frozen=True, slots=True)
class Summary:
    """Top-level findings summary that the note-draft consumes."""

    bone_loss_pattern: str = "unknown"  # e.g. "generalized_horizontal"
    aap_stage_estimate: Optional[AAPStage] = None
    jaw_classification: Optional[Jaw] = None
    vertical_defects: List[VerticalDefect] = field(default_factory=list)
    caries_findings: List[CariesSummaryEntry] = field(default_factory=list)
    quadrants: List[QuadrantSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "bone_loss_pattern": self.bone_loss_pattern,
            "aap_stage_estimate": self.aap_stage_estimate,
            "jaw_classification": self.jaw_classification,
            "vertical_defects": [v.to_dict() for v in self.vertical_defects],
            "caries_findings": [c.to_dict() for c in self.caries_findings],
            "quadrants": [q.to_dict() for q in self.quadrants],
        })


@dataclass(frozen=True, slots=True)
class LowConfidenceFinding:
    """A finding flagged for human verification.

    ``type`` is a short tag ("caries", "keypoint", "bone_loss", etc.).
    The renderer uses dashed outlines for any finding whose ``tooth`` +
    ``surface`` matches a row here.
    """

    type: str
    tooth: Optional[str] = None
    surface: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "type": self.type,
            "tooth": self.tooth,
            "surface": self.surface,
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
            "reason": self.reason,
        })


# ---------------------------------------------------------------------------
# Runtime metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Metadata:
    """Provenance + runtime telemetry for the analysis result."""

    models: Dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    device: str = "cpu"
    schema_version: str = SCHEMA_VERSION
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": dict(self.models),
            "runtime_seconds": round(float(self.runtime_seconds), 3),
            "device": self.device,
            "schema_version": self.schema_version,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The full output of one ``analyze()`` invocation.

    Serialize via ``to_dict()`` for JSON, or pass directly to the
    rendering / note-draft layers.
    """

    image: ImageMeta
    teeth: List[ToothFinding] = field(default_factory=list)
    summary: Summary = field(default_factory=Summary)
    low_confidence_findings: List[LowConfidenceFinding] = field(default_factory=list)
    note_draft: Optional[str] = None
    metadata: Metadata = field(default_factory=Metadata)

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none({
            "schema_version": SCHEMA_VERSION,
            "image": self.image.to_dict(),
            "teeth": [t.to_dict() for t in self.teeth],
            "summary": self.summary.to_dict(),
            "low_confidence_findings": [f.to_dict() for f in self.low_confidence_findings],
            "note_draft": self.note_draft,
            "metadata": self.metadata.to_dict(),
        })
