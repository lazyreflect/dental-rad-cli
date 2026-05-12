"""Matplotlib-based annotated PNG renderer.

Produces a single full-resolution annotated PNG matching the input image
dimensions. Overlays include tooth bounding boxes, CEJ + bone-crest
landmarks, CEJ→bone-crest severity-tier-colored segments, % bone-loss
labels, caries bounding boxes with depth labels, defect-pattern shading
(hatched for vertical, solid for horizontal), and a summary banner at
top. Low-confidence findings are drawn with dashed outlines.

Output is annotated-only at native input resolution — no original
reference panel. Doctors verifying findings already see the radiograph
in their PMS; the AI overlay is what they need from this tool.

Color discipline
----------------

We use a deliberately small palette so the artifact reads well at
chairside-monitor color profiles:

- CEJ landmarks   → green dot
- Bone-crest      → red dot
- Apex            → purple dot
- Mild segment    → green
- Moderate        → goldenrod (amber)
- Severe          → firebrick (red)
- Caries bbox     → cyan (distinct from severity palette)
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
    CariesFinding,
    LowConfidenceFinding,
    SeverityTier,
    ToothFinding,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_BBOX_EDGE = "#888888"
_CEJ_DOT = "#2ca02c"   # green
_CREST_DOT = "#d62728"  # red
_APEX_DOT = "#9467bd"   # purple
_CARIES_EDGE = "#17becf"  # cyan — distinct from severity palette

_TIER_COLORS = {
    "mild": "#2ca02c",       # green
    "moderate": "#daa520",   # goldenrod
    "severe": "#b22222",     # firebrick
}

# Render at native input resolution. DPI=100 means figure-inches map
# 1:1 to image-pixels at 100 px/inch — convenient round number; the
# absolute value doesn't matter as long as figsize × dpi == image_size.
_RENDER_DPI = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _low_conf_set(findings: Iterable[LowConfidenceFinding]) -> Set[Tuple[str, Optional[str]]]:
    """Build ``{(tooth, surface)}`` lookup set for dashed-outline trigger."""
    return {(f.tooth or "", f.surface) for f in findings}


def _site_is_low_confidence(
    tooth_fdi: str,
    surface: Optional[str],
    low_conf: Set[Tuple[str, Optional[str]]],
) -> bool:
    return (tooth_fdi, surface) in low_conf or (tooth_fdi, None) in low_conf


def _font_scale(image_h: int) -> float:
    """Return a font-size multiplier so labels remain readable across image sizes.

    Calibrated so a 1000-px-tall image renders at 1.0 (the historical
    default), with linear scaling for larger/smaller inputs.
    """
    return max(0.6, image_h / 1000.0)


def _draw_tooth(
    ax: plt.Axes,
    tooth: ToothFinding,
    low_conf: Set[Tuple[str, Optional[str]]],
    font_scale: float,
) -> None:
    """Draw all annotations for one tooth on the given axes."""
    fs_label = 8 * font_scale
    fs_pct = 8 * font_scale
    marker_size = 5 * font_scale
    line_w = 2.0 * font_scale

    label_bg = dict(facecolor="black", alpha=0.65, pad=2.0, edgecolor="none")

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
            linewidth=1.0 * font_scale,
            linestyle="--" if is_low else "-",
        ))
        ax.text(
            x1 + 3,
            y1 + 12 * font_scale,
            tooth.fdi,
            color="white",
            fontsize=fs_label,
            family="monospace",
            bbox=label_bg,
        )

    kp = tooth.keypoints

    # CEJ landmarks (green)
    for cej in (kp.cej_mesial, kp.cej_distal):
        if cej is not None:
            ax.plot(cej[0], cej[1], "o", color=_CEJ_DOT, markersize=marker_size,
                    markeredgecolor="black", markeredgewidth=0.5)

    # Bone-crest landmarks (red)
    for crest in (kp.bone_crest_mesial, kp.bone_crest_distal):
        if crest is not None:
            ax.plot(crest[0], crest[1], "o", color=_CREST_DOT, markersize=marker_size,
                    markeredgecolor="black", markeredgewidth=0.5)

    # Apex landmark (purple)
    if kp.apex is not None:
        ax.plot(kp.apex[0], kp.apex[1], "o", color=_APEX_DOT, markersize=marker_size,
                markeredgecolor="black", markeredgewidth=0.5)

    # CEJ→bone-crest segments per site, color-coded by tier
    _draw_site_segment(ax, tooth, "mesial", kp.cej_mesial, kp.bone_crest_mesial,
                       low_conf, font_scale, fs_pct, line_w, label_bg)
    _draw_site_segment(ax, tooth, "distal", kp.cej_distal, kp.bone_crest_distal,
                       low_conf, font_scale, fs_pct, line_w, label_bg)

    # Caries lesions (cyan)
    for car in tooth.caries:
        _draw_caries(ax, tooth.fdi, car, low_conf, font_scale, fs_label, label_bg)


def _draw_site_segment(
    ax: plt.Axes,
    tooth: ToothFinding,
    surface: str,
    cej,
    crest,
    low_conf: Set[Tuple[str, Optional[str]]],
    font_scale: float,
    fs_pct: float,
    line_w: float,
    label_bg: dict,
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
        linewidth=line_w,
        linestyle=linestyle,
    )

    if site and site.pct is not None:
        if tooth.pattern == "angular_vertical":
            hatch = "//"
        else:
            hatch = None
        midx, midy = (cej[0] + crest[0]) / 2.0, (cej[1] + crest[1]) / 2.0
        pad = 8 * font_scale
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
            midx + 10 * font_scale,
            midy,
            f"{site.pct:.0f}%",
            color="white",
            fontsize=fs_pct,
            family="monospace",
            bbox=label_bg,
        )


def _draw_caries(
    ax: plt.Axes,
    tooth_fdi: str,
    caries: CariesFinding,
    low_conf: Set[Tuple[str, Optional[str]]],
    font_scale: float,
    fs_label: float,
    label_bg: dict,
) -> None:
    """Draw a caries lesion bbox + depth/surface label."""
    if caries.bbox is None:
        return
    x1, y1, x2, y2 = caries.bbox
    is_low = _site_is_low_confidence(tooth_fdi, caries.surface, low_conf)
    linestyle = "--" if is_low else "-"

    ax.add_patch(Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=_CARIES_EDGE,
        linewidth=1.5 * font_scale,
        linestyle=linestyle,
    ))
    label = f"#{tooth_fdi} {caries.surface} {caries.depth}"
    if caries.confidence is not None and caries.confidence > 0:
        label += f" ({caries.confidence:.0%})"
    ax.text(
        x1 + 3,
        y2 - 4,
        label,
        color="white",
        fontsize=fs_label,
        family="monospace",
        bbox=dict(facecolor=_CARIES_EDGE, alpha=0.85, pad=2.0, edgecolor="none"),
        verticalalignment="bottom",
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
    """Render a full-resolution annotated PNG to ``out_path``.

    The output PNG matches the input image's pixel dimensions exactly,
    with overlays drawn on top. No side-by-side original reference panel
    — doctors verifying findings already see the source radiograph in
    their PMS, and the previous side-by-side rendering halved the usable
    resolution for evaluating keypoint placement.

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
    font_scale = _font_scale(height)

    # Figure sized so figsize × DPI == image pixel dimensions.
    fig = plt.figure(figsize=(width / _RENDER_DPI, height / _RENDER_DPI), dpi=_RENDER_DPI)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed; no margin
    ax.imshow(image, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image coords: y grows downward
    ax.set_axis_off()

    # Summary banner at top of image.
    banner = _summary_banner(result)
    ax.text(
        6,
        6,
        banner,
        bbox=dict(facecolor="black", alpha=0.7, pad=3.0, edgecolor="none"),
        color="white",
        fontsize=9 * font_scale,
        family="monospace",
        verticalalignment="top",
    )

    # Draw all teeth (bbox + keypoints + severity segments + caries).
    for tooth in result.teeth:
        _draw_tooth(ax, tooth, low_conf, font_scale)

    fig.savefig(out_path, dpi=_RENDER_DPI, pad_inches=0)
    plt.close(fig)
    return out_path
