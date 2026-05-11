"""Matplotlib-based annotated PNG renderer.

Produces a side-by-side image: clean original on the left, annotation
overlay on the right. The annotated half carries tooth bounding boxes,
CEJ + bone-crest landmarks, CEJ→bone-crest severity-tier-colored
segments, % bone-loss labels, defect-pattern shading (hatched for
vertical, solid for horizontal), and a summary banner at top. Low-
confidence findings are drawn with dashed outlines.

Design source: see ``output/proposals/2026-05-11-dental-rad-cli-v0-
design.md`` §"Annotated PNG output design". The implementation here is
fresh code; no upstream paper repo content is reused.

Color discipline
----------------

We use a deliberately small palette so the artifact reads well at
chairside-monitor color profiles:

- CEJ landmarks   → green dot
- Bone-crest      → red dot
- Mild segment    → green
- Moderate        → goldenrod (amber)
- Severe         → firebrick (red)
- Vertical fill   → hatched ("/" or "\\\\")
- Horizontal fill → solid translucent
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # headless, must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dental_rad_cli.schema import (
    AnalysisResult,
    BoneLossSite,
    LowConfidenceFinding,
    SeverityTier,
    ToothFinding,
    ToothKeypointsFull,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_BBOX_EDGE = "#888888"
_CEJ_DOT = "#2ca02c"   # green
_CREST_DOT = "#d62728"  # red

_TIER_COLORS = {
    "mild": "#2ca02c",       # green
    "moderate": "#daa520",   # goldenrod
    "severe": "#b22222",     # firebrick
}

_LABEL_BG = dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none")
_LABEL_FONT = dict(color="white", fontsize=6, family="monospace")


def _midpoint(
    p1: Optional[Tuple[float, float, float]],
    p2: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float]]:
    """Return the midpoint of two confidence-tagged keypoints, or one if only one is present."""
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return (float(p2[0]), float(p2[1]))
    if p2 is None:
        return (float(p1[0]), float(p1[1]))
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _low_conf_set(findings: Iterable[LowConfidenceFinding]) -> Set[Tuple[str, Optional[str]]]:
    """Build ``{(tooth, surface)}`` lookup set for dashed-outline trigger."""
    return {(f.tooth or "", f.surface) for f in findings}


def _site_is_low_confidence(
    tooth_fdi: str,
    surface: str,
    low_conf: Set[Tuple[str, Optional[str]]],
) -> bool:
    return (tooth_fdi, surface) in low_conf or (tooth_fdi, None) in low_conf


def _draw_tooth(
    ax: plt.Axes,
    tooth: ToothFinding,
    low_conf: Set[Tuple[str, Optional[str]]],
) -> None:
    """Draw all annotations for one tooth on the given axes."""
    # Bounding box
    if tooth.bbox is not None:
        x1, y1, x2, y2 = tooth.bbox
        is_low = _site_is_low_confidence(tooth.fdi, None, low_conf)
        ax.add_patch(Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=_BBOX_EDGE,
            linewidth=0.8,
            linestyle="--" if is_low else "-",
        ))
        ax.text(
            x1 + 2,
            y1 + 8,
            tooth.fdi,
            color="white",
            fontsize=6,
            family="monospace",
            bbox=_LABEL_BG,
        )

    kp = tooth.keypoints

    # CEJ landmarks (green)
    for cej in (kp.cej_mesial, kp.cej_distal):
        if cej is not None:
            ax.plot(cej[0], cej[1], "o", color=_CEJ_DOT, markersize=3)

    # Bone-crest landmarks (red)
    for crest in (kp.bone_crest_mesial, kp.bone_crest_distal):
        if crest is not None:
            ax.plot(crest[0], crest[1], "o", color=_CREST_DOT, markersize=3)

    # CEJ→bone-crest segments per site, color-coded by tier
    _draw_site_segment(ax, tooth, "mesial", kp.cej_mesial, kp.bone_crest_mesial, low_conf)
    _draw_site_segment(ax, tooth, "distal", kp.cej_distal, kp.bone_crest_distal, low_conf)


def _draw_site_segment(
    ax: plt.Axes,
    tooth: ToothFinding,
    surface: str,
    cej: Optional[Tuple[float, float, float]],
    crest: Optional[Tuple[float, float, float]],
    low_conf: Set[Tuple[str, Optional[str]]],
) -> None:
    """Draw the CEJ→bone-crest segment + % label for one site."""
    if cej is None or crest is None:
        return
    site: Optional[BoneLossSite] = getattr(tooth.bone_loss, surface, None)
    tier: Optional[SeverityTier] = site.tier if site else None
    color = _TIER_COLORS.get(tier, "#999999") if tier else "#999999"

    is_low = _site_is_low_confidence(tooth.fdi, surface, low_conf)
    linestyle = "--" if is_low else "-"

    ax.plot(
        [cej[0], crest[0]],
        [cej[1], crest[1]],
        color=color,
        linewidth=1.5,
        linestyle=linestyle,
    )

    if site and site.pct is not None:
        # Vertical defect → hatched fill banner; horizontal → solid alpha.
        if tooth.pattern == "angular_vertical":
            hatch = "//"
        else:
            hatch = None
        # Small filled marker behind the % label so doctors get a
        # pattern cue even on tiny segments.
        midx, midy = (cej[0] + crest[0]) / 2.0, (cej[1] + crest[1]) / 2.0
        pad = 6
        ax.add_patch(Rectangle(
            (midx - pad, midy - pad),
            pad * 2,
            pad * 2,
            facecolor=color,
            alpha=0.18,
            hatch=hatch,
            edgecolor="none",
        ))
        ax.text(
            midx + 7,
            midy,
            f"{site.pct:.0f}%",
            bbox=_LABEL_BG,
            **_LABEL_FONT,
        )


def _summary_banner(result: AnalysisResult) -> str:
    """Compose the top-of-image summary banner text."""
    s = result.summary
    parts: list[str] = []
    pattern = s.bone_loss_pattern.replace("_", " ")
    stage = f"AAP Stage {s.aap_stage_estimate}" if s.aap_stage_estimate else "stage unknown"
    parts.append(f"{pattern} bone loss  ·  {stage}")
    if s.vertical_defects:
        defects = ", ".join(
            f"{d.site.replace('_', ' #')} ({d.pct:.0f}%)" for d in s.vertical_defects
        )
        parts.append(f"Vertical defects: {defects}")
    if s.caries_findings:
        cs = ", ".join(
            f"#{c.tooth} {c.surface} ({c.depth})" for c in s.caries_findings
        )
        parts.append(f"Caries: {cs}")
    return "\n".join(parts)


def render_annotated(
    image,  # np.ndarray RGB; typed as Any to keep numpy import light
    result: AnalysisResult,
    out_path: Path,
) -> Path:
    """Render a side-by-side annotated PNG to ``out_path``.

    Parameters
    ----------
    image
        RGB numpy array as returned by ``analyze._load_image_rgb``.
    result
        The ``AnalysisResult`` whose findings are overlaid.
    out_path
        Where to write the PNG. Parent directory is created.

    Returns
    -------
    Path
        ``out_path`` on success.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[0], image.shape[1]
    low_conf = _low_conf_set(result.low_confidence_findings)

    # Two-panel figure: clean | annotated.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # image coords: y grows downward

    axes[0].set_title("original", fontsize=8)
    axes[1].set_title("annotated", fontsize=8)

    # Summary banner above the annotated panel.
    banner = _summary_banner(result)
    axes[1].text(
        4,
        12,
        banner,
        bbox=_LABEL_BG,
        color="white",
        fontsize=7,
        family="monospace",
        verticalalignment="top",
    )

    # Draw all teeth.
    for tooth in result.teeth:
        _draw_tooth(axes[1], tooth, low_conf)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
