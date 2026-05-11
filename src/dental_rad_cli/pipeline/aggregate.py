"""Per-quadrant and whole-mouth AAP staging rollup.

Pure functions. No I/O.

AAP-staging interpretation (the methodology brief documents tier
thresholds but does not specify whole-mouth staging — clinical AAP 2018
staging combines per-tooth severity into one of I, II, III, IV with
extent and complexity modifiers. The simplification used here
intentionally drops modifiers we can't observe from a single radiograph
and rolls four buckets out of the per-site tier list):

- Stage I:  no site more severe than mild.
- Stage II: at least one moderate, no severe.
- Stage III: at least one severe (``< 3`` severe sites).
- Stage IV: ``>= 3`` severe sites (proxy for "extensive bone loss"
  modifier — Stage IV usually requires extraction-imminent / complex
  rehabilitation criteria the radiograph alone can't tell us; we use
  the count proxy and document the limitation).

The Stage IV count proxy is the load-bearing interpretive choice — see
the open question in the subagent report.

FDI quadrant rule (ISO 3950): the first digit of the FDI two-digit
number encodes quadrant: ``1=UR, 2=UL, 3=LL, 4=LR`` for permanent teeth
(and ``5..8`` for primary; we only handle permanent here).

Per-tooth tier reduction: a tooth has two sites (mesial + distal). The
"tooth tier" for quadrant rollup is the worst of the two; a tooth with
one site missing uses the other site's tier; a tooth with both sites
missing is excluded from quadrant counts.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from dental_rad_cli.schema import (
    AAPStage,
    BoneLossSite,
    Quadrant,
    QuadrantSummary,
    SeverityTier,
    ToothFinding,
)

# Stage IV threshold — number of severe sites required.
_STAGE_IV_SEVERE_COUNT: int = 3

_QUADRANT_BY_FDI_FIRST_DIGIT: Dict[int, Quadrant] = {
    1: "UR",
    2: "UL",
    3: "LL",
    4: "LR",
}

_TIER_RANK: Dict[SeverityTier, int] = {"mild": 0, "moderate": 1, "severe": 2}


def _quadrant_of_fdi(fdi: str) -> Optional[Quadrant]:
    """Return the quadrant for a permanent-tooth FDI string ("11"..."48"),
    or None for primary / malformed / out-of-range.
    """
    if not fdi or len(fdi) != 2 or not fdi.isdigit():
        return None
    n = int(fdi)
    if not (11 <= n <= 48):
        return None
    return _QUADRANT_BY_FDI_FIRST_DIGIT.get(n // 10)


def aap_stage(per_site_findings: Iterable[BoneLossSite]) -> AAPStage:
    """Aggregate per-site bone-loss findings into a single AAP stage.

    See module docstring for the staging rules. Sites with no tier
    (``tier is None``) are ignored. An empty list returns Stage I (no
    evidence of bone loss → healthy presumption).
    """
    severe = 0
    moderate = 0
    for site in per_site_findings:
        if site is None:
            continue
        if site.tier == "severe":
            severe += 1
        elif site.tier == "moderate":
            moderate += 1
    if severe >= _STAGE_IV_SEVERE_COUNT:
        return "IV"
    if severe >= 1:
        return "III"
    if moderate >= 1:
        return "II"
    return "I"


def _tooth_sites(tooth: ToothFinding) -> List[BoneLossSite]:
    """Return the non-None per-site measurements on a tooth."""
    sites: List[BoneLossSite] = []
    if tooth.bone_loss.mesial is not None:
        sites.append(tooth.bone_loss.mesial)
    if tooth.bone_loss.distal is not None:
        sites.append(tooth.bone_loss.distal)
    return sites


def _tooth_worst(
    tooth: ToothFinding,
) -> Tuple[Optional[SeverityTier], Optional[float]]:
    """Return ``(worst_tier, worst_pct)`` across the tooth's sites."""
    best_rank = -1
    best_tier: Optional[SeverityTier] = None
    best_pct: Optional[float] = None
    for site in _tooth_sites(tooth):
        if site.tier is None:
            continue
        r = _TIER_RANK[site.tier]
        if r > best_rank or (r == best_rank and site.pct is not None and (best_pct is None or site.pct > best_pct)):
            best_rank = r
            best_tier = site.tier
            best_pct = site.pct
    return best_tier, best_pct


def quadrant_summary(findings: List[ToothFinding]) -> Dict[Quadrant, QuadrantSummary]:
    """Group per-tooth findings into UR / UL / LL / LR quadrant summaries.

    Teeth whose FDI string does not map to a permanent quadrant
    (e.g. primary teeth, malformed inputs) are silently dropped.
    """
    buckets: Dict[Quadrant, List[ToothFinding]] = {
        "UR": [],
        "UL": [],
        "LL": [],
        "LR": [],
    }
    for f in findings:
        q = _quadrant_of_fdi(f.fdi)
        if q is None:
            continue
        buckets[q].append(f)

    result: Dict[Quadrant, QuadrantSummary] = {}
    for quadrant, teeth in buckets.items():
        n_teeth = len(teeth)
        n_mild = 0
        n_moderate = 0
        n_severe = 0
        worst_rank = -1
        worst_tier: Optional[SeverityTier] = None
        worst_pct: Optional[float] = None
        for t in teeth:
            tier, pct = _tooth_worst(t)
            if tier == "mild":
                n_mild += 1
            elif tier == "moderate":
                n_moderate += 1
            elif tier == "severe":
                n_severe += 1
            if tier is not None:
                r = _TIER_RANK[tier]
                if r > worst_rank or (
                    r == worst_rank
                    and pct is not None
                    and (worst_pct is None or pct > worst_pct)
                ):
                    worst_rank = r
                    worst_tier = tier
                    worst_pct = pct
        result[quadrant] = QuadrantSummary(
            quadrant=quadrant,
            n_teeth=n_teeth,
            n_mild=n_mild,
            n_moderate=n_moderate,
            n_severe=n_severe,
            worst_tier=worst_tier,
            worst_pct=worst_pct,
        )
    return result
