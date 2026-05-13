"""BR6 (partial): stratify dev errors by DenPAR Arch + Site metadata.

DenPAR ships an XLSX with per-image metadata:
  id | Arch (Lower/Upper) | Site (Right/Left/Anterior) | FDI list

This joins the dev benchmark per-site records with that metadata so we
can see error patterns by:
  - arch: Lower vs Upper
  - site: Right / Left / Anterior
  - is_anterior view (Site == "Anterior")

Full per-FDI stratification needs a tooth-numbering model to map bbox
to specific FDI (parallel dental-tooth-numbering session is building
that). For now arch+site is the coarse-but-useful cut.

Usage::

    python scripts/stratify_by_denpar_meta.py [--json <path>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl required. pip install openpyxl", file=sys.stderr)
    raise SystemExit(1)


def _load_metadata(xlsx_path: Path) -> dict[str, dict]:
    """Return {stem_str: {arch, site, fdis}}."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    out: dict[str, dict] = {}
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    # Find columns.
    id_col = arch_col = site_col = fdi_col = None
    for i, h in enumerate(header):
        if h is None:
            continue
        h_low = str(h).strip().lower()
        if h_low == "id":
            id_col = i
        elif h_low == "arch":
            arch_col = i
        elif h_low == "site":
            site_col = i
        elif "fdi" in h_low:
            fdi_col = i
    for row in rows:
        if id_col is None or row[id_col] is None:
            continue
        stem = str(int(row[id_col])) if isinstance(row[id_col], float) else str(row[id_col]).strip()
        out[stem] = {
            "arch": str(row[arch_col]).strip() if arch_col is not None and row[arch_col] else None,
            "site": str(row[site_col]).strip() if site_col is not None and row[site_col] else None,
            "fdis": (str(row[fdi_col]).strip() if fdi_col is not None and row[fdi_col] else ""),
        }
    return out


def _print_table(title: str, buckets: dict, order: list) -> None:
    print(f"\n{title}")
    print(f"  {'bucket':<24} {'n':>4} {'mean':>8} {'median':>8} "
          f"{'p90':>8} {'max':>8}")
    print("  " + "-" * 66)
    for k in order:
        errs = buckets.get(k, [])
        if not errs:
            print(f"  {k:<24} {0:>4} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue
        a = np.array(errs)
        print(f"  {k:<24} {a.size:>4} {a.mean():>8.3f} "
              f"{np.median(a):>8.3f} {np.percentile(a, 90):>8.3f} "
              f"{a.max():>8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--xlsx", type=Path,
        default=Path("data/denpar/Dataset/Characteristics of radiographs included.xlsx"),
    )
    ap.add_argument(
        "--evidence-dir", type=Path,
        default=Path("output/training-evidence"),
    )
    args = ap.parse_args()

    if args.json is None:
        candidates = sorted(args.evidence_dir.glob("benchmark-eval-dev-*.json"))
        if not candidates:
            print("ERROR: no dev benchmark JSONs", file=sys.stderr)
            return 1
        args.json = candidates[-1]

    print(f"Reading {args.json}")
    print(f"Reading metadata from {args.xlsx}\n")
    payload = json.loads(args.json.read_text())
    records = payload.get("per_site_records") or []

    metadata = _load_metadata(args.xlsx)
    print(f"metadata rows: {len(metadata)}")

    # Join + stratify.
    by_arch: dict[str, list[float]] = {}
    by_site: dict[str, list[float]] = {}
    by_anterior: dict[str, list[float]] = {}
    by_arch_severity: dict[tuple, list[float]] = {}

    def _sev(g):
        if g < 2: return "healthy"
        if g < 4: return "mild"
        if g < 6: return "moderate"
        if g < 8: return "severe"
        return "extreme"

    n_matched = 0
    n_missing = 0
    for r in records:
        err = r.get("abs_err_mm")
        gt = r.get("gt_mm")
        if err is None or gt is None:
            continue
        stem = r.get("stem")
        meta = metadata.get(stem)
        if meta is None:
            n_missing += 1
            continue
        n_matched += 1
        arch = meta["arch"] or "Unknown"
        site = meta["site"] or "Unknown"
        is_ant = site == "Anterior"
        by_arch.setdefault(arch, []).append(err)
        by_site.setdefault(site, []).append(err)
        by_anterior.setdefault(
            "Anterior view" if is_ant else "Posterior view", []
        ).append(err)
        by_arch_severity.setdefault((arch, _sev(gt)), []).append(err)

    print(f"records joined: {n_matched}  missing metadata: {n_missing}\n")

    _print_table("MAE by Arch", by_arch, ["Lower", "Upper", "Unknown"])
    _print_table("MAE by Site", by_site,
                 ["Right", "Left", "Anterior", "Unknown"])
    _print_table("MAE by view (Anterior vs Posterior)", by_anterior,
                 ["Anterior view", "Posterior view"])

    print("\nArch × GT severity cross-tab (mean MAE / n)")
    arches = ["Lower", "Upper", "Unknown"]
    sevs = ["healthy", "mild", "moderate", "severe", "extreme"]
    print(f"  {'severity':<12}" + "".join(f"{a:>16}" for a in arches))
    for s in sevs:
        row = f"  {s:<12}"
        for a in arches:
            errs = by_arch_severity.get((a, s), [])
            if errs:
                row += f"{np.mean(errs):>10.3f} (n={len(errs):>3})"
            else:
                row += f"{'-':>16}"
        print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
