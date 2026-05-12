"""Tests for the wired inference orchestrator (Tasks 1-6).

All model invocations are mocked — these tests verify the
detection / keypoint / segmentation / build-findings paths thread
correctly into an ``AnalysisResult`` without loading any real weights.

Mocking strategy:
- ``ModelBundle.get_*`` methods are patched to return fakes whose
  ``.predict()`` (YOLO) or ``__call__`` (KP-RCNN) emits schema-shaped
  outputs.
- The stage-stub functions are invoked directly with the fake bundle
  + a small synthetic RGB image so the math stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from dental_rad_cli.analyze import (
    ModelBundle,
    _bbox_iou,
    _build_findings_from_stages,
    _detect_device,
    _run_keypoint_passes,
    _run_segmentation,
    _run_tooth_detection,
)
from dental_rad_cli.schema import (
    AnalysisResult,
    BoneLossPerSite,
    CariesFinding,
    ToothFinding,
)


# ---------------------------------------------------------------------------
# Ultralytics + KP-RCNN fakes
# ---------------------------------------------------------------------------


class _FakeBoxes:
    """Mimics ``results[0].boxes`` from Ultralytics."""

    def __init__(
        self,
        xyxy: Iterable[Iterable[float]],
        conf: Iterable[float],
        cls: Iterable[int],
    ) -> None:
        self.xyxy = np.asarray(list(xyxy), dtype=np.float32)
        self.conf = np.asarray(list(conf), dtype=np.float32)
        self.cls = np.asarray(list(cls), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)


class _FakeMasks:
    """Mimics ``results[0].masks`` from Ultralytics segmentation."""

    def __init__(self, polygons: List[np.ndarray]) -> None:
        self.xy = polygons


class _FakeResult:
    def __init__(
        self,
        boxes: _FakeBoxes | None = None,
        masks: _FakeMasks | None = None,
    ) -> None:
        self.boxes = boxes
        self.masks = masks


def _fake_yolo(boxes: _FakeBoxes | None = None, masks: _FakeMasks | None = None) -> MagicMock:
    """Return a YOLO-instance-like MagicMock whose .predict() yields one result."""
    inst = MagicMock()
    inst.predict.return_value = [_FakeResult(boxes=boxes, masks=masks)]
    return inst


class _FakeKPModel:
    """Mimics torchvision KP-RCNN: callable, .to(device), returns one dict."""

    def __init__(self, instances: List[dict]) -> None:
        # instances: list of {"bbox": (x1,y1,x2,y2), "keypoints": np.ndarray (K, 3)}
        self._instances = instances

    def to(self, _device: str) -> "_FakeKPModel":
        return self

    def __call__(self, _batch: Any) -> List[dict]:
        if not self._instances:
            return [{"boxes": np.empty((0, 4)), "keypoints": np.empty((0, 0, 3)), "scores": np.empty((0,))}]
        boxes = np.stack([np.asarray(i["bbox"], dtype=np.float32) for i in self._instances])
        kps = np.stack([np.asarray(i["keypoints"], dtype=np.float32) for i in self._instances])
        scores = np.asarray([i.get("score", 1.0) for i in self._instances], dtype=np.float32)
        return [{"boxes": boxes, "keypoints": kps, "scores": scores}]


def _make_bundle(tmp_path: Path) -> ModelBundle:
    """Return a ModelBundle pointed at an empty dir — getters get patched."""
    return ModelBundle(weights_dir=tmp_path)


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_tooth_detection_parses_yolo_results_and_assigns_geometric_fdi(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)
    # Three detections, intentionally out of x-order. Class ids 0/1
    # (single/double); a third predicted as bg(2) which should be dropped.
    boxes = _FakeBoxes(
        xyxy=[
            [200.0, 50.0, 240.0, 100.0],  # x-center = 220
            [10.0, 50.0, 50.0, 100.0],    # x-center = 30   → FDI "1"
            [100.0, 50.0, 140.0, 100.0],  # x-center = 120  → FDI "2"
            [500.0, 50.0, 540.0, 100.0],  # bg — dropped
        ],
        conf=[0.9, 0.95, 0.85, 0.7],
        cls=[1, 0, 1, 2],
    )
    bundle.get_tooth_detect = lambda: _fake_yolo(boxes=boxes)  # type: ignore[method-assign]
    dets = _run_tooth_detection(bundle, _rgb())

    # bg dropped → 3 detections, ordered by x-center.
    assert [d["fdi"] for d in dets] == ["1", "2", "3"]
    assert [d["root_class"] for d in dets] == ["single", "double", "double"]
    # First detection (originally idx 1) was class 0 → "single", x-center 30.
    assert dets[0]["bbox"] == (10.0, 50.0, 50.0, 100.0)
    assert dets[0]["confidence"] == pytest.approx(0.95)


def test_run_keypoint_passes_pairs_by_iou_and_propagates_none_for_misses(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)

    # One tooth detection.
    detections = [
        {
            "fdi": "1",
            "bbox": (100.0, 100.0, 200.0, 300.0),
            "confidence": 0.9,
            "root_class": "double",
        }
    ]

    # CEJ model: one instance with bbox matching the tooth (high IoU).
    cej_inst = [
        {
            "bbox": (105.0, 105.0, 195.0, 295.0),
            "keypoints": np.array(
                [[120.0, 150.0, 1.0], [180.0, 150.0, 1.0]], dtype=np.float32
            ),
        }
    ]
    # Bone model: one instance, also high IoU.
    bone_inst = [
        {
            "bbox": (105.0, 105.0, 195.0, 295.0),
            "keypoints": np.array(
                [[125.0, 170.0, 1.0], [175.0, 170.0, 1.0]], dtype=np.float32
            ),
        }
    ]
    # Apex model: predicted way off to the side — IoU below 0.3 threshold → None.
    apex_inst = [
        {
            "bbox": (500.0, 500.0, 600.0, 700.0),
            "keypoints": np.array([[550.0, 600.0, 1.0]], dtype=np.float32),
        }
    ]

    bundle.get_keypoint_cej = lambda: _FakeKPModel(cej_inst)  # type: ignore[method-assign]
    bundle.get_keypoint_bone = lambda: _FakeKPModel(bone_inst)  # type: ignore[method-assign]
    bundle.get_keypoint_apex = lambda: _FakeKPModel(apex_inst)  # type: ignore[method-assign]

    kp_rows = _run_keypoint_passes(bundle, _rgb(h=400, w=400), detections, device="cpu")

    assert len(kp_rows) == 1
    row = kp_rows[0]
    assert row["fdi"] == "1"
    assert row["cej"] == [(120.0, 150.0), (180.0, 150.0)]
    assert row["bone_crest"] == [(125.0, 170.0), (175.0, 170.0)]
    assert row["apex"] is None  # IoU below threshold


def test_run_segmentation_extracts_polygons_for_tooth_and_bone(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle(tmp_path)
    tooth_poly = np.array(
        [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]], dtype=np.float32
    )
    bone_poly_a = np.array(
        [[0.0, 60.0], [100.0, 60.0], [100.0, 80.0], [0.0, 80.0]], dtype=np.float32
    )
    bone_poly_b = np.array(
        [[120.0, 60.0], [200.0, 60.0], [200.0, 80.0], [120.0, 80.0]], dtype=np.float32
    )

    bundle.get_segmentation_tooth = lambda: _fake_yolo(  # type: ignore[method-assign]
        masks=_FakeMasks([tooth_poly])
    )
    bundle.get_segmentation_bone = lambda: _fake_yolo(  # type: ignore[method-assign]
        masks=_FakeMasks([bone_poly_a, bone_poly_b])
    )

    tooth_polys, bone_polys = _run_segmentation(bundle, _rgb(h=200, w=300))
    assert len(tooth_polys) == 1
    assert len(bone_polys) == 2
    # Each polygon coerced to list[tuple[float,float]].
    assert tooth_polys[0][0] == (10.0, 10.0)
    assert bone_polys[1][2] == (200.0, 80.0)


def test_build_findings_threads_into_valid_analysis_result() -> None:
    """End-to-end through ``_build_findings_from_stages`` with one tooth."""
    detections = [
        {
            "fdi": "1",
            "bbox": (100.0, 50.0, 200.0, 350.0),
            "confidence": 0.92,
            "root_class": "single",
        }
    ]
    keypoints = [
        {
            "fdi": "1",
            "cej": [(110.0, 100.0), (190.0, 100.0)],
            "bone_crest": [(115.0, 130.0), (185.0, 130.0)],
            "apex": (150.0, 300.0),
        }
    ]
    # No tooth/bone polygons → pattern stays "unknown" (the rasterize
    # path is exercised separately).
    teeth, summary, low_conf = _build_findings_from_stages(
        detections=detections,
        keypoints=keypoints,
        tooth_polys=[],
        bone_polys=[],
        caries=[],
        image_shape=(400, 400),
    )

    assert len(teeth) == 1
    t = teeth[0]
    assert t.fdi == "1"
    assert t.root_class == "single"
    assert isinstance(t.bone_loss, BoneLossPerSite)
    # CEJ-bone-apex are colinear-ish here; pct should be a small positive
    # number well below the severe threshold.
    assert t.bone_loss.mesial is not None
    assert t.bone_loss.mesial.pct is not None
    assert 0.0 < t.bone_loss.mesial.pct < 50.0
    # CEJ y (100) < apex y (300) → mandibular per jaw rule (brief §3.1).
    assert summary.jaw_classification == "mandibular"
    # No severe sites here → Stage I or II.
    assert summary.aap_stage_estimate in ("I", "II")
    # Pattern unknown without polygons.
    assert t.pattern == "unknown"
    # No caries → empty.
    assert summary.caries_findings == []
    # Build full AnalysisResult — shape must be JSON-serializable.
    from dental_rad_cli.schema import ImageMeta, Metadata
    result = AnalysisResult(
        image=ImageMeta(path="t.jpg", width=400, height=400),
        teeth=teeth,
        summary=summary,
        low_confidence_findings=low_conf,
        metadata=Metadata(),
    )
    d = result.to_dict()
    assert d["teeth"][0]["fdi"] == "1"
    assert d["summary"]["jaw_classification"] == "mandibular"


def test_build_findings_routes_caries_to_parent_tooth_and_flags_low_conf() -> None:
    detections = [
        {
            "fdi": "1",
            "bbox": (0.0, 0.0, 100.0, 200.0),
            "confidence": 0.9,
            "root_class": "double",
        },
        {
            "fdi": "2",
            "bbox": (200.0, 0.0, 300.0, 200.0),
            "confidence": 0.9,
            "root_class": "double",
        },
    ]
    caries = [
        # Centered at (50, 50) — inside tooth #1.
        CariesFinding(surface="mesial", depth="D1", bbox=(30.0, 30.0, 70.0, 70.0), confidence=0.85),
        # Centered at (250, 50) — inside tooth #2; below 0.75 → low_conf.
        CariesFinding(surface="distal", depth="E1", bbox=(230.0, 30.0, 270.0, 70.0), confidence=0.60),
        # Centered far away — unrouted.
        CariesFinding(surface="occlusal", depth="D3", bbox=(900.0, 900.0, 950.0, 950.0), confidence=0.9),
    ]
    teeth, summary, low_conf = _build_findings_from_stages(
        detections=detections,
        keypoints=[],  # no kp → bone-loss skipped; we focus on caries routing
        tooth_polys=[],
        bone_polys=[],
        caries=caries,
        image_shape=(400, 400),
    )

    # Tooth #1 carries the first lesion.
    by_fdi = {t.fdi: t for t in teeth}
    assert len(by_fdi["1"].caries) == 1
    assert by_fdi["1"].caries[0].depth == "D1"
    # Tooth #2 carries the low-confidence lesion.
    assert len(by_fdi["2"].caries) == 1
    assert by_fdi["2"].caries[0].depth == "E1"
    # Summary lists all three findings; unrouted goes under "unknown".
    tooth_keys = {entry.tooth for entry in summary.caries_findings}
    assert {"1", "2", "unknown"}.issubset(tooth_keys)
    # Low-confidence rows: the 0.60 caries triggers; missing-landmark
    # rows are emitted per tooth because keypoints=[] → no cej/bone/apex.
    types = {lc.type for lc in low_conf}
    assert "caries" in types
    assert "keypoint" in types


def test_bbox_iou_basic_cases() -> None:
    # Identical → 1.0.
    assert _bbox_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(1.0)
    # Disjoint → 0.0.
    assert _bbox_iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 20.0, 20.0)) == 0.0
    # Half overlap.
    iou = _bbox_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0))
    # inter = 5*10=50, union = 100+100-50=150 → 1/3.
    assert iou == pytest.approx(50.0 / 150.0)


def test_detect_device_returns_known_string() -> None:
    """Smoke-test: returns one of the three known device strings."""
    dev = _detect_device()
    assert dev in ("cuda", "mps", "cpu")
