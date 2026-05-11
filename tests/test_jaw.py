"""Tests for `dental_rad_cli.pipeline.jaw_classify`.

The brief's rule: per-tooth, CEJ y < apex y → mandibular; otherwise
maxillary. Image-level decision is majority vote with silent fallback
to mandibular.
"""

from __future__ import annotations

from dental_rad_cli.pipeline.jaw_classify import classify_jaw
from dental_rad_cli.schema import ToothWithKeypoints


def _t(cej_y: float | None, apex_y: float | None) -> ToothWithKeypoints:
    return ToothWithKeypoints(cej_y=cej_y, apex_y=apex_y)


def test_classify_jaw_clear_mandibular() -> None:
    # All teeth: CEJ above apex (smaller y) → mandibular.
    teeth = [_t(100, 400), _t(110, 410), _t(105, 395)]
    assert classify_jaw(teeth) == "mandibular"


def test_classify_jaw_clear_maxillary() -> None:
    # All teeth: CEJ below apex (larger y) → maxillary.
    teeth = [_t(400, 100), _t(410, 110), _t(395, 105)]
    assert classify_jaw(teeth) == "maxillary"


def test_classify_jaw_majority_maxillary() -> None:
    teeth = [_t(400, 100), _t(400, 110), _t(110, 400)]
    assert classify_jaw(teeth) == "maxillary"


def test_classify_jaw_majority_mandibular() -> None:
    teeth = [_t(100, 400), _t(110, 410), _t(400, 110)]
    assert classify_jaw(teeth) == "mandibular"


def test_classify_jaw_tie_defaults_to_mandibular() -> None:
    # 1 vs 1 → tie → mandibular per the silent-fallback rule.
    teeth = [_t(100, 400), _t(400, 100)]
    assert classify_jaw(teeth) == "mandibular"


def test_classify_jaw_empty_defaults_to_mandibular() -> None:
    assert classify_jaw([]) == "mandibular"


def test_classify_jaw_missing_coords_defaults_per_tooth_to_mandibular() -> None:
    # Two teeth with missing coords + one clearly maxillary → 2 mand vs
    # 1 max → mandibular wins.
    teeth = [_t(None, None), _t(None, 50), _t(400, 100)]
    assert classify_jaw(teeth) == "mandibular"


def test_classify_jaw_equal_y_is_maxillary() -> None:
    # CEJ y == apex y is NOT "CEJ above" → tooth is maxillary by rule.
    teeth = [_t(200, 200)]
    assert classify_jaw(teeth) == "maxillary"
