"""Tests for :func:`dental_rad_cli.pipeline.caries_inference.detect_caries`.

All YOLO model invocations are mocked — these tests verify the
class-id → schema-depth mapping, the confidence threshold, the
surface inference vs. parent tooth bboxes, and the empty / multi
detection cases. No real weights are loaded.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dental_rad_cli.pipeline.caries_inference import (
    _CLASS_TO_DEPTH,
    _surface_for_caries,
    detect_caries,
)
from dental_rad_cli.schema import CariesFinding, ToothFinding


# ---------------------------------------------------------------------------
# Helpers — build a fake Ultralytics result the helper can consume.
# ---------------------------------------------------------------------------


class _FakeBoxes:
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


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes | None) -> None:
        self.boxes = boxes


def _patched_yolo(boxes: _FakeBoxes | None):
    """Return a YOLO-class stub whose .predict() returns one fake result."""

    fake_instance = MagicMock()
    fake_instance.predict.return_value = [_FakeResult(boxes)]

    fake_yolo_class = MagicMock(return_value=fake_instance)
    return fake_yolo_class


def _ensure_ultralytics_stub() -> None:
    """Insert a minimal ``ultralytics`` module stub if the real one is absent.

    The inference helper does a local ``from ultralytics import YOLO``;
    if the package isn't installed in the test environment the import
    itself fails before the patch fires. This installs a stub that the
    test patches override.
    """
    if "ultralytics" in sys.modules:
        return
    mod = types.ModuleType("ultralytics")
    mod.YOLO = MagicMock()  # type: ignore[attr-defined]
    sys.modules["ultralytics"] = mod


def _make_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# detect_caries
# ---------------------------------------------------------------------------


def test_detect_caries_happy_path_three_classes(tmp_path: Path) -> None:
    _ensure_ultralytics_stub()
    boxes = _FakeBoxes(
        xyxy=[
            [10.0, 10.0, 30.0, 30.0],
            [40.0, 40.0, 60.0, 60.0],
            [5.0, 5.0, 15.0, 15.0],
        ],
        conf=[0.92, 0.81, 0.77],
        cls=[0, 1, 2],
    )
    with patch(
        "dental_rad_cli.pipeline.caries_inference.YOLO",
        _patched_yolo(boxes),
        create=True,
    ):
        # Also patch the lazy `from ultralytics import YOLO` site.
        with patch.dict(sys.modules, {"ultralytics": types.ModuleType("ultralytics")}):
            sys.modules["ultralytics"].YOLO = _patched_yolo(boxes)  # type: ignore[attr-defined]
            findings = detect_caries(
                image=_make_rgb(),
                model_weights=tmp_path / "fake.pt",
            )

    assert len(findings) == 3
    depths = [f.depth for f in findings]
    assert depths == ["E1", "D1", "D3"]
    # Bboxes round-trip.
    assert findings[0].bbox == (10.0, 10.0, 30.0, 30.0)
    # Surface defaults to 'unknown' without tooth_bboxes.
    assert all(f.surface == "unknown" for f in findings)
    # Confidence preserved.
    assert findings[0].confidence == pytest.approx(0.92)
    # Each result is a real CariesFinding.
    assert all(isinstance(f, CariesFinding) for f in findings)


def test_detect_caries_empty_result(tmp_path: Path) -> None:
    _ensure_ultralytics_stub()
    boxes = _FakeBoxes(xyxy=np.empty((0, 4)), conf=[], cls=[])
    fake_yolo = _patched_yolo(boxes)
    with patch.dict(sys.modules, {"ultralytics": types.ModuleType("ultralytics")}):
        sys.modules["ultralytics"].YOLO = fake_yolo  # type: ignore[attr-defined]
        findings = detect_caries(
            image=_make_rgb(),
            model_weights=tmp_path / "fake.pt",
        )
    assert findings == []


def test_detect_caries_filters_low_confidence(tmp_path: Path) -> None:
    """A row with confidence below the threshold is dropped.

    Ultralytics' predict() normally filters by ``conf=`` itself but we
    keep a defensive secondary filter; verify it fires when the fake
    box list ignores the threshold.
    """
    _ensure_ultralytics_stub()
    boxes = _FakeBoxes(
        xyxy=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        conf=[0.95, 0.30],  # second below default 0.5 threshold
        cls=[0, 1],
    )
    fake_yolo = _patched_yolo(boxes)
    with patch.dict(sys.modules, {"ultralytics": types.ModuleType("ultralytics")}):
        sys.modules["ultralytics"].YOLO = fake_yolo  # type: ignore[attr-defined]
        findings = detect_caries(
            image=_make_rgb(),
            model_weights=tmp_path / "fake.pt",
            conf_threshold=0.5,
        )
    assert len(findings) == 1
    assert findings[0].depth == "E1"


def test_detect_caries_surface_inference_via_parent_tooth(tmp_path: Path) -> None:
    _ensure_ultralytics_stub()
    # Two caries lesions on one tooth — one to the left of center,
    # one to the right.
    boxes = _FakeBoxes(
        xyxy=[
            [50.0, 50.0, 90.0, 100.0],  # center x = 70  (left of tooth-center 150)
            [200.0, 50.0, 240.0, 100.0],  # center x = 220 (right of tooth-center)
        ],
        conf=[0.9, 0.9],
        cls=[1, 1],
    )
    tooth = ToothFinding(
        fdi="30",
        bbox=(0.0, 0.0, 300.0, 200.0),  # center x = 150
        confidence=0.95,
    )
    fake_yolo = _patched_yolo(boxes)
    with patch.dict(sys.modules, {"ultralytics": types.ModuleType("ultralytics")}):
        sys.modules["ultralytics"].YOLO = fake_yolo  # type: ignore[attr-defined]
        findings = detect_caries(
            image=_make_rgb(h=200, w=300),
            model_weights=tmp_path / "fake.pt",
            tooth_bboxes=[tooth],
        )
    assert [f.surface for f in findings] == ["mesial", "distal"]


def test_detect_caries_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="expects.*RGB"):
        detect_caries(
            image=np.zeros((10, 10), dtype=np.uint8),  # 2-D grayscale
            model_weights=Path("/nonexistent.pt"),
        )


# ---------------------------------------------------------------------------
# Surface helper unit tests
# ---------------------------------------------------------------------------


def test_surface_for_caries_unknown_without_tooth_bboxes() -> None:
    assert _surface_for_caries((10.0, 10.0, 20.0, 20.0), None) == "unknown"
    assert _surface_for_caries((10.0, 10.0, 20.0, 20.0), []) == "unknown"


def test_surface_for_caries_nearest_tooth_fallback() -> None:
    # Caries lies outside both tooth bboxes — nearest by center distance wins.
    near = ToothFinding(fdi="30", bbox=(0.0, 0.0, 100.0, 100.0))
    far = ToothFinding(fdi="14", bbox=(500.0, 500.0, 600.0, 600.0))
    # Caries at (110-150, ...). Closer to `near` (center 50, 50) than `far` (550, 550).
    surface = _surface_for_caries((110.0, 50.0, 150.0, 90.0), [near, far])
    # Caries center (130, 70) is to the right of near's center (50, 50).
    assert surface == "distal"


def test_class_to_depth_is_exhaustive_and_in_range() -> None:
    """Guard against schema drift: every model class maps to a valid CariesDepth."""
    valid = {"E1", "E2", "D1", "D2", "D3"}
    assert set(_CLASS_TO_DEPTH) == {0, 1, 2}
    for depth in _CLASS_TO_DEPTH.values():
        assert depth in valid
