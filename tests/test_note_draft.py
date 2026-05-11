"""Tests for the note-draft template renderer."""

from __future__ import annotations

from dental_rad_cli.note_draft import render_note
from dental_rad_cli.schema import (
    AnalysisResult,
    CariesSummaryEntry,
    ImageMeta,
    LowConfidenceFinding,
    Summary,
    VerticalDefect,
)


def _base(image_type: str = "bitewing", **summary_kwargs) -> AnalysisResult:
    return AnalysisResult(
        image=ImageMeta(path="bw01.jpg", width=1280, height=960, type=image_type),
        summary=Summary(**summary_kwargs),
    )


def test_happy_path_horizontal_with_caries_and_vertical_defect():
    result = AnalysisResult(
        image=ImageMeta(path="bw01.jpg", width=1280, height=960, type="bitewing"),
        summary=Summary(
            bone_loss_pattern="generalized_horizontal",
            aap_stage_estimate="II",
            vertical_defects=[VerticalDefect(site="mesial_19", pct=32.0, confidence=0.8)],
            caries_findings=[
                CariesSummaryEntry(tooth="3", surface="MO", depth="D2", confidence=0.9),
                CariesSummaryEntry(tooth="14", surface="M", depth="D1", confidence=0.8),
            ],
        ),
    )
    note = render_note(result)
    assert "Bitewings demonstrate generalized horizontal bone loss" in note
    assert "moderate" in note
    assert "AAP Stage II" in note
    assert "vertical defect mesial of #19" in note
    assert "~32%" in note
    assert "Interproximal caries: #3 MO (D2), #14 M (D1)." in note


def test_no_caries_clause_when_summary_empty():
    result = _base(
        bone_loss_pattern="localized_horizontal",
        aap_stage_estimate="I",
    )
    note = render_note(result)
    assert "Interproximal caries" not in note
    assert "mild" in note
    assert "AAP Stage I" in note


def test_low_confidence_clause_present_when_findings_listed():
    result = AnalysisResult(
        image=ImageMeta(path="bw01.jpg", width=1280, height=960, type="bitewing"),
        summary=Summary(
            bone_loss_pattern="localized_horizontal",
            aap_stage_estimate="II",
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
    )
    note = render_note(result)
    assert "Low-confidence findings to verify:" in note
    assert "#30" in note
    assert "occlusal" in note
    assert "caries" in note
    assert "62% confidence" in note


def test_vertical_defects_multiple_sites():
    result = _base(
        bone_loss_pattern="generalized_horizontal",
        aap_stage_estimate="III",
        vertical_defects=[
            VerticalDefect(site="mesial_19", pct=32.0),
            VerticalDefect(site="distal_30", pct=40.0),
        ],
    )
    note = render_note(result)
    assert "vertical defects" in note
    assert "mesial of #19" in note
    assert "distal of #30" in note


def test_periapical_image_label():
    result = _base(image_type="periapical", bone_loss_pattern="localized_horizontal")
    note = render_note(result)
    assert note.startswith("Periapical demonstrate")


def test_unknown_image_type_uses_generic_label():
    result = _base(image_type="unknown", bone_loss_pattern="localized_horizontal")
    note = render_note(result)
    assert note.startswith("Radiograph demonstrate")


def test_no_stage_keeps_string_clean():
    result = _base(bone_loss_pattern="unknown", aap_stage_estimate=None)
    note = render_note(result)
    # No "AAP Stage None" leak.
    assert "None" not in note
    assert "AAP Stage" not in note


def test_paragraphs_separated_by_blank_line():
    result = AnalysisResult(
        image=ImageMeta(path="bw01.jpg", width=1280, height=960, type="bitewing"),
        summary=Summary(
            bone_loss_pattern="generalized_horizontal",
            aap_stage_estimate="II",
            caries_findings=[
                CariesSummaryEntry(tooth="3", surface="MO", depth="D2"),
            ],
        ),
        low_confidence_findings=[
            LowConfidenceFinding(type="caries", tooth="30", confidence=0.62),
        ],
    )
    note = render_note(result)
    paragraphs = note.split("\n\n")
    assert len(paragraphs) == 3
