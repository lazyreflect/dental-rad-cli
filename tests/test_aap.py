"""Tests for `dental_rad_cli.pipeline.aggregate`.

Validates aap_stage rule (I/II/III/IV) and quadrant_summary FDI routing.
"""

from __future__ import annotations

from dental_rad_cli.pipeline.aggregate import aap_stage, quadrant_summary
from dental_rad_cli.schema import (
    BoneLossPerSite,
    BoneLossSite,
    ToothFinding,
)


def _site(tier: str | None, pct: float | None = None) -> BoneLossSite:
    return BoneLossSite(pct=pct, tier=tier)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# aap_stage
# ---------------------------------------------------------------------------


def test_aap_stage_empty_is_stage_i() -> None:
    assert aap_stage([]) == "I"


def test_aap_stage_all_mild_is_stage_i() -> None:
    sites = [_site("mild", 5.0), _site("mild", 10.0), _site("mild", 14.0)]
    assert aap_stage(sites) == "I"


def test_aap_stage_one_moderate_is_stage_ii() -> None:
    sites = [_site("mild", 5.0), _site("moderate", 20.0), _site("mild", 8.0)]
    assert aap_stage(sites) == "II"


def test_aap_stage_one_severe_is_stage_iii() -> None:
    sites = [_site("mild", 5.0), _site("moderate", 20.0), _site("severe", 50.0)]
    assert aap_stage(sites) == "III"


def test_aap_stage_two_severe_still_stage_iii() -> None:
    sites = [_site("severe", 40.0), _site("severe", 60.0), _site("mild", 5.0)]
    assert aap_stage(sites) == "III"


def test_aap_stage_three_severe_is_stage_iv() -> None:
    # 3+ severe sites trigger the Stage IV count proxy.
    sites = [_site("severe", 40.0), _site("severe", 50.0), _site("severe", 60.0)]
    assert aap_stage(sites) == "IV"


def test_aap_stage_ignores_none_tiers() -> None:
    sites = [_site(None), _site(None), _site("moderate", 20.0)]
    assert aap_stage(sites) == "II"


# ---------------------------------------------------------------------------
# quadrant_summary
# ---------------------------------------------------------------------------


def _tooth(fdi: str, mesial: BoneLossSite | None, distal: BoneLossSite | None = None) -> ToothFinding:
    return ToothFinding(
        fdi=fdi,
        bone_loss=BoneLossPerSite(mesial=mesial, distal=distal),
    )


def test_quadrant_summary_routes_by_first_digit() -> None:
    findings = [
        _tooth("11", _site("mild", 5.0)),  # UR
        _tooth("23", _site("moderate", 20.0)),  # UL
        _tooth("36", _site("severe", 50.0)),  # LL
        _tooth("47", _site("mild", 8.0)),  # LR
    ]
    summary = quadrant_summary(findings)
    assert set(summary.keys()) == {"UR", "UL", "LL", "LR"}
    assert summary["UR"].n_mild == 1
    assert summary["UL"].n_moderate == 1
    assert summary["LL"].n_severe == 1
    assert summary["LR"].n_mild == 1
    assert summary["LL"].worst_tier == "severe"
    assert summary["LL"].worst_pct == 50.0


def test_quadrant_summary_worst_of_mesial_and_distal() -> None:
    # Same tooth, mesial=mild, distal=severe → tooth-tier is severe.
    findings = [
        _tooth("16", mesial=_site("mild", 5.0), distal=_site("severe", 40.0)),
    ]
    summary = quadrant_summary(findings)
    assert summary["UR"].n_severe == 1
    assert summary["UR"].n_mild == 0
    assert summary["UR"].worst_tier == "severe"
    assert summary["UR"].worst_pct == 40.0


def test_quadrant_summary_skips_primary_and_malformed() -> None:
    findings = [
        _tooth("51", _site("severe", 50.0)),  # primary — skip
        _tooth("99", _site("severe", 50.0)),  # out of range — skip
        _tooth("", _site("severe", 50.0)),  # malformed — skip
        _tooth("21", _site("mild", 5.0)),  # UL
    ]
    summary = quadrant_summary(findings)
    assert summary["UL"].n_teeth == 1
    assert summary["UR"].n_teeth == 0
    assert summary["LL"].n_teeth == 0
    assert summary["LR"].n_teeth == 0


def test_quadrant_summary_empty_quadrant_has_zero_counts() -> None:
    findings = [_tooth("11", _site("mild", 5.0))]
    summary = quadrant_summary(findings)
    assert summary["LL"].n_teeth == 0
    assert summary["LL"].worst_tier is None
    assert summary["LL"].worst_pct is None
