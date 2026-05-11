"""Schema dataclass roundtrip + JSON serialization tests."""

from __future__ import annotations

import json

import pytest

from dental_rad_cli.schema import (
    AnalysisResult,
    BoneLossPerSite,
    BoneLossSite,
    CariesFinding,
    CariesSummaryEntry,
    ImageMeta,
    LowConfidenceFinding,
    Metadata,
    SCHEMA_VERSION,
    Summary,
    ToothFinding,
    ToothKeypointsFull,
    VerticalDefect,
)


def _sample_result() -> AnalysisResult:
    return AnalysisResult(
        image=ImageMeta(path="bw01.jpg", width=1280, height=960, type="bitewing"),
        teeth=[
            ToothFinding(
                fdi="30",
                universal="30",
                bbox=(10.0, 20.0, 110.0, 200.0),
                confidence=0.94,
                root_class="double",
                keypoints=ToothKeypointsFull(
                    cej_mesial=(15.0, 30.0, 0.9),
                    cej_distal=(105.0, 30.0, 0.9),
                    bone_crest_mesial=(20.0, 60.0, 0.88),
                    bone_crest_distal=(100.0, 60.0, 0.87),
                    apex=(55.0, 180.0, 0.92),
                ),
                bone_loss=BoneLossPerSite(
                    mesial=BoneLossSite(pct=18.5, tier="moderate"),
                    distal=BoneLossSite(pct=22.1, tier="moderate"),
                ),
                pattern="horizontal",
                caries=[
                    CariesFinding(
                        surface="mesial",
                        depth="D1",
                        bbox=(12.0, 25.0, 30.0, 50.0),
                        confidence=0.81,
                    ),
                ],
            ),
        ],
        summary=Summary(
            bone_loss_pattern="generalized_horizontal",
            aap_stage_estimate="II",
            jaw_classification="mandibular",
            vertical_defects=[VerticalDefect(site="mesial_19", pct=32.0, confidence=0.81)],
            caries_findings=[
                CariesSummaryEntry(
                    tooth="30", surface="mesial", depth="D1", confidence=0.81,
                ),
            ],
        ),
        low_confidence_findings=[
            LowConfidenceFinding(
                type="caries",
                tooth="30",
                surface="occlusal",
                confidence=0.62,
                reason="below_0.75_threshold",
            ),
        ],
        note_draft="example",
        metadata=Metadata(
            models={"tooth_detect": "v0.1"},
            runtime_seconds=3.7,
            device="mps",
        ),
    )


def test_to_dict_is_json_serializable():
    result = _sample_result()
    d = result.to_dict()
    # Round-trip through json.dumps to confirm all leaves are JSON-safe.
    text = json.dumps(d)
    reparsed = json.loads(text)
    assert reparsed["schema_version"] == SCHEMA_VERSION
    assert reparsed["image"]["path"] == "bw01.jpg"
    assert reparsed["image"]["type"] == "bitewing"
    assert reparsed["teeth"][0]["fdi"] == "30"
    assert reparsed["summary"]["aap_stage_estimate"] == "II"
    assert reparsed["low_confidence_findings"][0]["tooth"] == "30"
    assert reparsed["metadata"]["schema_version"] == SCHEMA_VERSION


def test_frozen_dataclass_immutability():
    result = _sample_result()
    with pytest.raises(Exception):
        result.image.path = "other.jpg"  # type: ignore[misc]


def test_optional_fields_omitted_when_none():
    # A ToothFinding with no caries and missing keypoints should still
    # serialize cleanly. None-valued fields are dropped (the consumer
    # treats absent keys as "not present").
    finding = ToothFinding(fdi="30")
    d = finding.to_dict()
    assert d["fdi"] == "30"
    # bbox was None → omitted
    assert "bbox" not in d
    # caries is empty list → present (signals "ran, found none")
    assert d["caries"] == []


def test_empty_collections_preserved():
    result = AnalysisResult(image=ImageMeta(path="x.jpg", width=10, height=10))
    d = result.to_dict()
    # Empty lists kept; None top-level fields dropped.
    assert d["teeth"] == []
    assert d["low_confidence_findings"] == []
    assert "note_draft" not in d


def test_bone_loss_site_pct_rounded():
    site = BoneLossSite(pct=18.4567, tier="moderate")
    assert site.to_dict()["pct"] == 18.5


def test_metadata_schema_version_defaults():
    md = Metadata()
    assert md.schema_version == SCHEMA_VERSION


def test_keypoints_full_drops_none_entries():
    kp = ToothKeypointsFull(
        cej_mesial=(1.0, 2.0, 0.9),
        cej_distal=None,
    )
    d = kp.to_dict()
    assert d["cej_mesial"] == [1.0, 2.0, 0.9]
    assert "cej_distal" not in d
