"""BR4: dumb baseline floor for the architecture ladder.

The honest question: what MAE does a constant prediction give? Without
this anchor, "we got 0.723 mean MAE" is unanchored — maybe a trivial
baseline already gets 1.0 and the whole pipeline only buys 0.3.

This script computes several no-segmentation, no-learning baselines
from the existing dev benchmark JSON's GT mm distribution:

  - constant_0       : predict every site = 0 mm ("no bone loss")
                       → MAE = mean(gt_mm). The "model is useless" floor.
  - constant_median  : predict every site = median(gt_mm)
                       → minimum constant-prediction MAE (median is
                          optimal for MAE).
  - constant_mean    : predict every site = mean(gt_mm)
                       → minimizes RMSE, suboptimal for MAE.
  - constant_3mm     : predict every site = 3 mm (mild-defect default)
  - bbox_only        : use tooth-bbox geometry only. Pred bone-crest =
                       bbox top + 10 px. Pred CEJ = bbox top.
                       Pred mm = 10/px_per_mm ≈ uniform mild.

We don't have full per-site bbox+px_per_mm without re-running inference,
so bbox_only uses the GT bbox (assumes a perfect tooth detector). This
strips away segmentation entirely.

Usage::

    python scripts/dumb_baseline_floor.py [--json <path>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def _mae(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(gt - pred)))


def _median(a: np.ndarray) -> float:
    return float(np.median(a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None,
                    help="dev benchmark JSON (latest by default)")
    ap.add_argument("--evidence-dir", type=Path,
                    default=Path("output/training-evidence"))
    args = ap.parse_args()

    if args.json is None:
        candidates = sorted(
            args.evidence_dir.glob("benchmark-eval-dev-*.json")
        )
        if not candidates:
            print("ERROR: no dev benchmark JSONs", file=sys.stderr)
            return 1
        args.json = candidates[-1]

    print(f"Reading {args.json}\n")
    payload = json.loads(args.json.read_text())
    records = payload.get("per_site_records") or []
    if not records:
        print("ERROR: no per_site_records", file=sys.stderr)
        return 1

    gt = np.array([r["gt_mm"] for r in records if r.get("gt_mm") is not None])
    actual_pred = np.array([
        r["pred_mm"] for r in records
        if r.get("pred_mm") is not None and r.get("gt_mm") is not None
    ])

    # Baselines.
    baselines = {
        "constant_0":      np.zeros_like(gt),
        "constant_1.0":    np.full_like(gt, 1.0),
        "constant_2.0":    np.full_like(gt, 2.0),
        "constant_3.0":    np.full_like(gt, 3.0),
        "constant_median": np.full_like(gt, _median(gt)),
        "constant_mean":   np.full_like(gt, gt.mean()),
    }

    # Production line (from JSON) for reference.
    actual_mae = payload.get("mm_mae_stats", {}).get("mean")

    print(f"GT mm distribution (n={gt.size}):")
    print(f"  mean   {gt.mean():.3f}  median {_median(gt):.3f}  "
          f"p90 {np.percentile(gt, 90):.3f}  max {gt.max():.3f}\n")

    print(f"Production pipeline mean MAE: {actual_mae:.3f}\n"
          if actual_mae else "")

    print("Dumb baseline MAE (predict same number for every site):")
    print(f"  {'baseline':<22} {'pred':>8} {'mae':>8} "
          f"{'vs.prod':>10}")
    print("  " + "-" * 52)
    for name, pred in baselines.items():
        m = _mae(gt, pred)
        delta = m - actual_mae if actual_mae else None
        print(f"  {name:<22} {pred[0]:>8.3f} {m:>8.3f} "
              f"{(f'{delta:+.3f}' if delta is not None else '-'):>10}")

    # Stratified by GT severity — what does each baseline give per bucket?
    def _bucket(g):
        if g < 2: return "healthy"
        if g < 4: return "mild"
        if g < 6: return "moderate"
        if g < 8: return "severe"
        return "extreme"
    bucket_names = ["healthy", "mild", "moderate", "severe", "extreme"]
    by_bucket: dict[str, list[float]] = {b: [] for b in bucket_names}
    for r in records:
        if r.get("gt_mm") is None:
            continue
        by_bucket[_bucket(r["gt_mm"])].append(r["gt_mm"])

    print("\nDumb 'predict the constant K' MAE by GT bucket:")
    print(f"  {'bucket':<10} {'n':>4} {'mean(gt)':>10} "
          f"{'pred=0':>10} {'pred=1':>10} {'pred=median':>13}")
    print("  " + "-" * 65)
    for b in bucket_names:
        gts = np.array(by_bucket[b])
        if gts.size == 0:
            print(f"  {b:<10} {0:>4}")
            continue
        print(f"  {b:<10} {gts.size:>4} {gts.mean():>10.3f} "
              f"{_mae(gts, np.zeros_like(gts)):>10.3f} "
              f"{_mae(gts, np.ones_like(gts)):>10.3f} "
              f"{_mae(gts, np.full_like(gts, _median(gt))):>13.3f}")

    if actual_pred.size:
        print(f"\nProduction pipeline pred-mm distribution (n={actual_pred.size}):")
        print(f"  mean   {actual_pred.mean():.3f}  "
              f"median {_median(actual_pred):.3f}  "
              f"p90 {np.percentile(actual_pred, 90):.3f}  "
              f"max {actual_pred.max():.3f}")

    print("\nInterpretation:")
    median_mae = _mae(gt, np.full_like(gt, _median(gt)))
    if actual_mae and actual_mae < median_mae:
        print(f"  Production ({actual_mae:.3f}) beats constant-median floor "
              f"({median_mae:.3f}). Pipeline buys "
              f"{median_mae - actual_mae:.3f} mm vs the trivial floor.")
    elif actual_mae:
        print(f"  Production ({actual_mae:.3f}) does NOT beat the "
              f"constant-median floor ({median_mae:.3f}). The pipeline "
              f"is failing to do better than a constant prediction.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
