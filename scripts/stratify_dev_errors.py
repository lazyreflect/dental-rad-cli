"""Stratify per-site dev errors by GT-severity bucket + tooth-position proxy.

The mean MAE (0.723 mm) on the dev split mixes:
  - Healthy/mild cases the model handles well
  - Severe-visible bone-loss cases the algorithm may under-predict
  - Severe-hidden cases (1/2-walled defects) at the radiographic
    detection ceiling (sensitivity ~0.22 per literature)
  - Anterior teeth where GT may be mislabeled at the incisal edge

Stratifying separates the addressable buckets from the irreducible
ones, so we know which bucket to spend architecture effort on.

Buckets:
  GT-severity (from gt_mm):
    healthy   gt_mm < 2 mm
    mild      2 <= gt_mm < 4
    moderate  4 <= gt_mm < 6
    severe    6 <= gt_mm < 8
    extreme   gt_mm >= 8  (likely 1/2-walled hidden defect OR GT issue)

  Tooth-position proxy (from bbox aspect ratio + relative size):
    anterior_likely  bbox_h/bbox_w > 2.5 AND bbox_w/image_w < 0.18
    posterior_likely bbox_h/bbox_w < 1.6 AND bbox_w/image_w > 0.18
    ambiguous        else

Usage::
    python scripts/stratify_dev_errors.py [--json path/to/benchmark.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def _bucket_gt(gt_mm: float) -> str:
    if gt_mm < 2:
        return "healthy"
    if gt_mm < 4:
        return "mild"
    if gt_mm < 6:
        return "moderate"
    if gt_mm < 8:
        return "severe"
    return "extreme"


def _bucket_position(record: dict) -> str:
    bbox = record.get("bbox")
    image_w = record.get("image_w")
    if bbox is None or image_w is None:
        return "ambiguous"
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return "ambiguous"
    aspect = h / w
    rel_w = w / image_w
    if aspect > 2.5 and rel_w < 0.18:
        return "anterior_likely"
    if aspect < 1.6 and rel_w > 0.18:
        return "posterior_likely"
    return "ambiguous"


def _stats(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0}
    a = np.asarray(errs)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def _print_bucket_table(title: str, buckets: dict[str, list[float]],
                        order: list[str]) -> None:
    print(f"\n{title}")
    print(f"  {'bucket':<22} {'n':>4} {'mean':>8} {'median':>8} "
          f"{'p90':>8} {'max':>8}")
    print("  " + "-" * 64)
    total_errs: list[float] = []
    for k in order:
        errs = buckets.get(k, [])
        total_errs.extend(errs)
        s = _stats(errs)
        if s["n"] == 0:
            print(f"  {k:<22} {0:>4} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
        else:
            print(f"  {k:<22} {s['n']:>4} {s['mean']:>8.3f} "
                  f"{s['median']:>8.3f} {s['p90']:>8.3f} {s['max']:>8.3f}")
    s = _stats(total_errs)
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<22} {s['n']:>4} {s['mean']:>8.3f} "
          f"{s['median']:>8.3f} {s['p90']:>8.3f} {s['max']:>8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None,
                    help="Benchmark JSON (latest dev by default)")
    ap.add_argument(
        "--evidence-dir", type=Path,
        default=Path("output/training-evidence"),
    )
    args = ap.parse_args()

    if args.json is None:
        candidates = sorted(args.evidence_dir.glob("benchmark-eval-dev-*.json"))
        if not candidates:
            print("ERROR: no dev benchmark JSONs found", file=sys.stderr)
            return 1
        args.json = candidates[-1]

    print(f"Reading {args.json}")
    payload = json.loads(args.json.read_text())
    records = payload.get("per_site_records") or []
    n_total_sites = payload.get("n_gt_sites")
    n_matched = payload.get("n_predicted_matched")
    print(f"records loaded: {len(records)}  "
          f"(benchmark reports n_gt_sites={n_total_sites}, "
          f"matched={n_matched})")

    if records and records[0].get("bbox") is None:
        print(
            "WARN: records have no bbox field. Re-run benchmark_eval.py with "
            "the updated SiteError schema to enable position bucketing.",
            file=sys.stderr,
        )

    # GT-severity bucketing.
    by_gt: dict[str, list[float]] = {}
    for r in records:
        gt = r.get("gt_mm")
        err = r.get("abs_err_mm")
        if gt is None or err is None:
            continue
        by_gt.setdefault(_bucket_gt(gt), []).append(err)
    _print_bucket_table(
        "MAE by GT severity bucket",
        by_gt,
        ["healthy", "mild", "moderate", "severe", "extreme"],
    )

    # Tooth-position bucketing.
    by_pos: dict[str, list[float]] = {}
    for r in records:
        err = r.get("abs_err_mm")
        if err is None:
            continue
        by_pos.setdefault(_bucket_position(r), []).append(err)
    _print_bucket_table(
        "MAE by tooth-position proxy",
        by_pos,
        ["anterior_likely", "posterior_likely", "ambiguous"],
    )

    # Cross-tab: GT severity × position. Surfaces e.g. anterior+extreme
    # (likely incisal-edge mislabel) vs. posterior+extreme (likely real
    # severe perio).
    print("\nCross-tab: GT severity × tooth position (mean MAE / n)")
    pos_order = ["anterior_likely", "posterior_likely", "ambiguous"]
    sev_order = ["healthy", "mild", "moderate", "severe", "extreme"]
    cross: dict[tuple, list[float]] = {}
    for r in records:
        gt = r.get("gt_mm")
        err = r.get("abs_err_mm")
        if gt is None or err is None:
            continue
        cross.setdefault((_bucket_gt(gt), _bucket_position(r)), []).append(err)
    print(f"  {'severity':<12}" + "".join(f"{p:>22}" for p in pos_order))
    for sev in sev_order:
        row = f"  {sev:<12}"
        for p in pos_order:
            errs = cross.get((sev, p), [])
            if errs:
                row += f"{np.mean(errs):>14.3f} (n={len(errs):>3})"
            else:
                row += f"{'-':>22}"
        print(row)

    # Standalone Q: what is MAE on "honest visible" cases?
    # Define: gt_mm < 6 (excludes severe-hidden suspects) AND position
    # not anterior_likely (excludes incisal-edge mislabel suspects).
    honest_errs: list[float] = []
    for r in records:
        gt = r.get("gt_mm")
        err = r.get("abs_err_mm")
        if gt is None or err is None:
            continue
        if gt >= 6:
            continue
        if _bucket_position(r) == "anterior_likely" and gt >= 4:
            continue
        honest_errs.append(err)
    s = _stats(honest_errs)
    print(f"\n'Honest visible' subset (gt_mm < 6 AND NOT "
          f"[anterior_likely AND gt_mm >= 4]):")
    print(f"  n={s['n']}  mean={s['mean']:.3f}  median={s['median']:.3f}  "
          f"p90={s['p90']:.3f}")
    print(f"  vs. Overjet 0.300 mm: Δ {s['mean']-0.300:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
