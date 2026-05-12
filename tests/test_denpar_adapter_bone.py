"""Regression tests for the DenPAR v3 bone derivation logic.

Both ``_bone_crest_for_bbox`` and ``_bone_polygons_from_polylines``
were rewritten after a hour-0 bug: the original adapter (Subagent F)
read ``Masks (Radiograph-wise)/`` as the bone-segmentation source,
but in DenPAR v3 that folder contains all-teeth-as-one-binary-mask,
not bone. v3 has no bone region masks at all — bone is stored as
polylines in ``Bone Level Annotations/``.

These tests pin the new derivation behavior so the regression doesn't
return on a future refactor.
"""

from __future__ import annotations

import pytest

from dental_rad_cli.data.denpar_adapter import (
    _bone_crest_for_bbox,
    _bone_polygons_from_polylines,
)


# ----- _bone_crest_for_bbox -----------------------------------------------


def test_bone_crest_interpolates_at_bbox_edges():
    """Mesial + distal points land at exactly ``bbox.x1`` and ``bbox.x2``."""
    # Polyline crosses x = 30 (mesial) and x = 100 (distal) with linear slope.
    line = [(0.0, 200.0), (50.0, 220.0), (150.0, 250.0)]
    bbox = (30.0, 100.0, 100.0, 400.0)  # tooth at x=30-100

    mesial, distal = _bone_crest_for_bbox(bbox, [line])

    assert mesial is not None
    assert distal is not None
    # mesial.x == bbox.x1, distal.x == bbox.x2
    assert mesial[0] == pytest.approx(30.0)
    assert distal[0] == pytest.approx(100.0)
    # Interpolated y: between (0,200) and (50,220), x=30 → y = 212
    assert mesial[1] == pytest.approx(212.0)
    # Interpolated y: between (50,220) and (150,250), x=100 → y = 235
    assert distal[1] == pytest.approx(235.0)


def test_bone_crest_returns_none_when_polyline_does_not_cover_edge():
    """Distal returns None when polyline x-range ends before bbox.x2."""
    line = [(10.0, 200.0), (50.0, 220.0)]  # x ∈ [10, 50]
    bbox = (10.0, 100.0, 100.0, 400.0)  # bbox extends to x=100

    mesial, distal = _bone_crest_for_bbox(bbox, [line])

    assert mesial is not None
    assert distal is None  # x=100 is outside the polyline range


def test_bone_crest_picks_most_coronal_when_multiple_polylines():
    """When 2 polylines cover the same x, return the smaller-y (more coronal) one."""
    line_upper = [(0.0, 100.0), (200.0, 100.0)]  # y=100 (more coronal, smaller y)
    line_lower = [(0.0, 500.0), (200.0, 500.0)]  # y=500 (less coronal)
    bbox = (50.0, 50.0, 150.0, 600.0)

    mesial, distal = _bone_crest_for_bbox(bbox, [line_upper, line_lower])

    assert mesial[1] == pytest.approx(100.0)
    assert distal[1] == pytest.approx(100.0)


def test_bone_crest_returns_none_pair_for_empty_input():
    """No polylines → (None, None)."""
    bbox = (0.0, 0.0, 100.0, 400.0)
    assert _bone_crest_for_bbox(bbox, []) == (None, None)


def test_bone_crest_vertical_segment_picks_more_coronal_endpoint():
    """A vertical polyline segment (xa == xb) at the bbox edge x → smaller y."""
    line = [(50.0, 300.0), (50.0, 100.0)]  # vertical segment at x=50
    bbox = (50.0, 50.0, 100.0, 500.0)

    mesial, _distal = _bone_crest_for_bbox(bbox, [line])

    assert mesial is not None
    assert mesial[1] == pytest.approx(100.0)  # smaller y wins


# ----- _bone_polygons_from_polylines --------------------------------------


def test_polygons_one_per_disjoint_polyline():
    """4 polylines at disjoint x ranges → 4 separate polygons."""
    polylines = [
        [(0.0, 100.0), (50.0, 100.0)],     # x ∈ [0, 50]
        [(100.0, 100.0), (150.0, 100.0)],  # x ∈ [100, 150]
        [(200.0, 100.0), (250.0, 100.0)],  # x ∈ [200, 250]
        [(300.0, 100.0), (350.0, 100.0)],  # x ∈ [300, 350]
    ]
    polys = _bone_polygons_from_polylines(polylines)
    # Strip half-width is 15, so disjoint-by-50 strips don't merge.
    assert len(polys) == 4


def test_polygons_merged_when_overlap():
    """Adjacent polylines whose buffers overlap → fewer polygons (merged)."""
    # Two polylines 20 px apart in x — within 2×15=30 buffer, should merge.
    polylines = [
        [(0.0, 100.0), (50.0, 100.0)],   # x ∈ [0, 50]
        [(60.0, 100.0), (110.0, 100.0)],  # x ∈ [60, 110], gap=10 < buffer*2=30
    ]
    polys = _bone_polygons_from_polylines(polylines)
    assert len(polys) == 1  # merged because buffers touch


def test_polygons_empty_input():
    assert _bone_polygons_from_polylines([]) == []


def test_polygons_single_point_polylines_skipped():
    """Polylines with <2 points can't form a LineString → skipped."""
    polylines = [[(10.0, 10.0)], []]
    assert _bone_polygons_from_polylines(polylines) == []


def test_polygons_strip_height_within_expected_range():
    """A horizontal polyline → strip roughly ±15 px tall around the centerline."""
    polylines = [[(0.0, 100.0), (200.0, 100.0)]]
    polys = _bone_polygons_from_polylines(polylines)
    assert len(polys) == 1
    ys = [y for _x, y in polys[0]]
    height = max(ys) - min(ys)
    # Buffer = 15 px each side → strip is ~30 px tall (rounded ends add a bit).
    assert 25.0 <= height <= 35.0
