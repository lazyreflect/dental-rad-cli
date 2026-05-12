"""Stratified per-image diagnostics on the trained CEJ polyline model.

For every test image:
- Run the trained CEJ polyline model, capture n_predicted_masks,
  total predicted band pixel count, max prediction confidence.
- Pull per-image GT metadata: n_CEJ_points, n_bboxes (teeth),
  Arch (Upper/Lower), Site (Anterior/Left/Right), FDI notation, image
  dimensions, mean intensity, restoration_proxy (fraction of pixels
  in the top intensity quartile, a coarse stand-in for radiopaque
  restorations).

Writes a single CSV at ``output/training-evidence/karpathy-stratify.csv``
and a per-stratum failure-rate report to stdout.

The hypothesis (Karpathy thing): the polyline model overfit DenPAR's
supervision-density pattern. Sparse-supervision images (few GT CEJ
points, narrow views, heavy restorations) produce sparse predictions.
Stratifying by Arch / Site / GT-point-count / restoration-load should
expose the systematic pattern.

Usage::

    python scripts/karpathy_stratify.py --weights weights/segmentation_cej.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dental_rad_cli.data.denpar_adapter import (  # noqa: E402
    _load_keypoint_json,
    _split_dir,
)


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([cl.apply(L), a, b]), cv2.COLOR_LAB2RGB)


def _restoration_proxy(gray: np.ndarray) -> float:
    """Fraction of pixels in the top intensity quartile.

    Radiopaque restorations (metal, ceramic, root canal fills) appear
    near-white on radiographs. The top-quartile-pixel fraction is a
    coarse-but-honest stand-in for "how much white stuff is in this
    image." Higher → more restorations or denser anatomy.
    """
    threshold = np.percentile(gray, 90)  # top 10%
    return float((gray >= threshold).sum() / gray.size)


def _fdi_count_categories(fdi: str) -> dict:
    """Parse FDI notation string into per-tooth-type counts.

    DenPAR FDI is like '44,45,46,47,48' or '12,11,21,22'. Returns
    a dict with counts of incisor / canine / premolar / molar /
    third_molar.
    """
    counts = {
        "incisors": 0, "canines": 0, "premolars": 0,
        "molars": 0, "third_molars": 0,
    }
    if not isinstance(fdi, str):
        return counts
    for tok in fdi.split(","):
        tok = tok.strip()
        if len(tok) < 2 or not tok[-1].isdigit():
            continue
        pos = int(tok[-1])  # 1=central incisor through 8=third molar
        if pos in (1, 2):
            counts["incisors"] += 1
        elif pos == 3:
            counts["canines"] += 1
        elif pos in (4, 5):
            counts["premolars"] += 1
        elif pos in (6, 7):
            counts["molars"] += 1
        elif pos == 8:
            counts["third_molars"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights/segmentation_cej.pt"))
    ap.add_argument("--denpar-root", type=Path, default=Path("data/denpar"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--out-csv", type=Path,
        default=Path("output/training-evidence/karpathy-stratify.csv"),
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    # Load characteristics XLSX for view metadata.
    char_path = args.denpar_root / "Dataset" / "Characteristics of radiographs included.xlsx"
    df_char = pd.read_excel(char_path)
    df_char["id"] = df_char["id"].astype(str)
    char_by_id = df_char.set_index("id")

    testing = _split_dir(args.denpar_root, "Testing")
    images_dir = testing / "Images"
    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))

    rows = []
    for i, stem in enumerate(stems):
        if i % 25 == 0:
            print(f"... {i}/{len(stems)}", flush=True)
        img_path = images_dir / f"{stem}.jpg"
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = _apply_clahe(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Model prediction.
        results = model.predict(
            source=rgb, conf=args.conf, verbose=False, save=False
        )
        n_masks = 0
        total_pixels = 0
        max_conf = 0.0
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            masks = results[0].masks.data.cpu().numpy().astype(bool)
            n_masks = len(masks)
            band = np.zeros((h, w), dtype=bool)
            for m in masks:
                m_rs = cv2.resize(
                    m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                band |= m_rs
            total_pixels = int(band.sum())
            if results[0].boxes is not None and results[0].boxes.conf is not None:
                confs = results[0].boxes.conf.cpu().numpy()
                if confs.size > 0:
                    max_conf = float(confs.max())

        # GT.
        kp = _load_keypoint_json(testing, stem) or {}
        bboxes = kp.get("bboxes", []) or []
        cej_pts = kp.get("CEJ_Points", []) or []
        n_cej = len(cej_pts)
        n_bboxes = len(bboxes)

        # Metadata.
        if stem in char_by_id.index:
            char_row = char_by_id.loc[stem]
            arch = char_row.get("Arch", None)
            site = char_row.get("Site", None)
            fdi = char_row.get("FDI notation of fully/partially visible teeth", "")
        else:
            arch = None
            site = None
            fdi = ""
        fdi_counts = _fdi_count_categories(fdi)
        n_teeth_visible_fdi = sum(fdi_counts.values())

        # Restoration proxy.
        restoration = _restoration_proxy(gray)
        mean_intensity = float(gray.mean())

        rows.append({
            "stem": stem,
            "n_masks": n_masks,
            "total_band_px": total_pixels,
            "max_conf": round(max_conf, 4),
            "predict_success": int(n_masks > 0),
            "n_cej_gt": n_cej,
            "n_bboxes_gt": n_bboxes,
            "img_w": w,
            "img_h": h,
            "aspect_ratio": round(w / h, 3) if h > 0 else 0,
            "mean_intensity": round(mean_intensity, 2),
            "restoration_proxy": round(restoration, 4),
            "arch": arch,
            "site": site,
            "n_teeth_fdi": n_teeth_visible_fdi,
            "incisors": fdi_counts["incisors"],
            "canines": fdi_counts["canines"],
            "premolars": fdi_counts["premolars"],
            "molars": fdi_counts["molars"],
            "third_molars": fdi_counts["third_molars"],
            "fdi_notation": fdi,
        })

    # Write CSV.
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_csv}")

    # Quick analysis.
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\nKARPATHY STRATIFICATION (n={len(df)}, conf={args.conf})\n{'='*60}\n")

    total = len(df)
    n_success = int(df["predict_success"].sum())
    print(f"Overall: predict_success = {n_success}/{total} ({100*n_success/total:.1f}%)\n")

    def _show_strat(group_col, title):
        print(f"--- by {title} ---")
        g = df.groupby(group_col).agg(
            n_images=("stem", "count"),
            success_rate=("predict_success", "mean"),
            mean_n_cej_gt=("n_cej_gt", "mean"),
            mean_restoration=("restoration_proxy", "mean"),
            mean_n_masks_when_success=("n_masks", "mean"),
        ).round(3)
        print(g.to_string())
        print()

    _show_strat("arch", "Arch (Upper/Lower)")
    _show_strat("site", "Site (Anterior/Left/Right)")

    # Bin restoration_proxy.
    df["restoration_bin"] = pd.cut(
        df["restoration_proxy"], bins=[-0.001, 0.10, 0.15, 0.20, 1.0],
        labels=["low", "med", "high", "very_high"],
    )
    _show_strat("restoration_bin", "restoration_proxy (intensity-quartile)")

    # Bin n_cej_gt.
    df["n_cej_bin"] = pd.cut(
        df["n_cej_gt"], bins=[-0.5, 2.5, 4.5, 7.5, 100],
        labels=["0-2", "3-4", "5-7", "8+"],
    )
    _show_strat("n_cej_bin", "n_GT_CEJ_points")

    # Cross-stratum: arch x site.
    print("--- crosstab: arch x site ---")
    ct = df.groupby(["arch", "site"]).agg(
        n=("stem", "count"),
        success_rate=("predict_success", "mean"),
    ).round(3)
    print(ct.to_string())
    print()

    print(f"\nFull CSV at {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
