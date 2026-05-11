"""Template-rendered clinical note draft.

The doctor copy-pastes (or auto-pastes via a follow-on integration) the
output into the chart note. v0 uses a deterministic template with no
LLM call; per-doctor voice customization is deferred to v0.5+ (see the
design doc §"Note-draft text generation" for the v1 LLM routing plan).

Per the code-quality review, the renderer consumes the ``AnalysisResult``
dataclass directly rather than a flat key-value mapping — that prevents
"missing key" silent failures and lets future template variants pull
nested fields without flattening upstream.

Template surface
----------------

The output has up to three paragraphs separated by blank lines:

1. **Bone-loss line.** Image type + overall pattern + AAP stage, with
   an optional clause naming any vertical defects.
2. **Caries clause.** "Interproximal caries: ..." or nothing if the
   summary lists no caries findings.
3. **Low-confidence flags.** "Low-confidence findings to verify: ..."
   or nothing if the result has no low-confidence findings.

The renderer NEVER inserts "no findings" filler — empty paragraphs are
dropped to keep the output clinically readable.
"""

from __future__ import annotations

from typing import List

from dental_rad_cli.schema import (
    AnalysisResult,
    CariesSummaryEntry,
    LowConfidenceFinding,
    VerticalDefect,
)

# Image-type → human label for the leading clause.
_IMAGE_LABELS = {
    "bitewing": "Bitewings",
    "periapical": "Periapical",
    "panoramic": "Panoramic",
    "unknown": "Radiograph",
}

# AAP stage → severity adjective for the leading clause.
_STAGE_SEVERITY = {
    "I": "mild",
    "II": "moderate",
    "III": "severe",
    "IV": "severe",
}


def _vertical_defects_clause(defects: List[VerticalDefect]) -> str:
    """Compose ", with a vertical defect ..." clause (empty if no defects)."""
    if not defects:
        return ""
    if len(defects) == 1:
        d = defects[0]
        # "mesial_19" → "mesial of #19"
        site_phrase = _site_phrase(d.site)
        return f", with a vertical defect {site_phrase} (~{d.pct:.0f}%)"
    site_phrases = [f"{_site_phrase(d.site)} (~{d.pct:.0f}%)" for d in defects]
    return ", with vertical defects " + ", ".join(site_phrases)


def _site_phrase(site_token: str) -> str:
    """Convert ``"mesial_19"`` → ``"mesial of #19"``. Pass-through for unknown shapes."""
    if "_" in site_token:
        surface, tooth = site_token.split("_", 1)
        return f"{surface} of #{tooth}"
    return site_token


def _caries_clause(findings: List[CariesSummaryEntry]) -> str:
    """Compose the caries paragraph, or empty string if none."""
    if not findings:
        return ""
    parts = [f"#{c.tooth} {c.surface} ({c.depth})" for c in findings]
    return "Interproximal caries: " + ", ".join(parts) + "."


def _low_confidence_clause(findings: List[LowConfidenceFinding]) -> str:
    """Compose the verification-needed paragraph, or empty string if none."""
    if not findings:
        return ""
    parts: List[str] = []
    for f in findings:
        tooth = f"#{f.tooth}" if f.tooth else "(unspecified tooth)"
        surface = f" {f.surface}" if f.surface else ""
        type_label = f.type if f.type else "finding"
        conf = ""
        if f.confidence is not None:
            conf = f" ({int(round(f.confidence * 100))}% confidence)"
        parts.append(f"{tooth}{surface} {type_label}{conf}")
    return "Low-confidence findings to verify: " + ", ".join(parts) + "."


def render_note(result: AnalysisResult) -> str:
    """Render the note-draft text for an analysis result.

    Returns a string with up to three paragraphs (bone-loss line,
    caries clause, low-confidence flags), separated by blank lines.
    Empty paragraphs are omitted.
    """
    image_label = _IMAGE_LABELS.get(result.image.type, "Radiograph")
    pattern = (result.summary.bone_loss_pattern or "unknown").replace("_", " ")
    stage = result.summary.aap_stage_estimate
    severity = _STAGE_SEVERITY.get(stage, None) if stage else None

    # Bone-loss line.
    if severity and stage:
        leading = (
            f"{image_label} demonstrate {pattern} bone loss, {severity} "
            f"(AAP Stage {stage})"
        )
    elif stage:
        leading = f"{image_label} demonstrate {pattern} bone loss (AAP Stage {stage})"
    else:
        leading = f"{image_label} demonstrate {pattern} bone loss"
    leading += _vertical_defects_clause(result.summary.vertical_defects)
    leading += "."

    paragraphs: List[str] = [leading]

    caries = _caries_clause(result.summary.caries_findings)
    if caries:
        paragraphs.append(caries)

    low_conf = _low_confidence_clause(result.low_confidence_findings)
    if low_conf:
        paragraphs.append(low_conf)

    return "\n\n".join(paragraphs)
