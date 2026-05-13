"""Generate full-150 dev-split contact sheet for visual inspection.

Karpathy's "become one with your data" step. Renders every image in the
dev split with GT + predicted overlays, per-image MAE in the header bar,
and writes an HTML gallery sorted by error descending. The intent is for
a human to scroll through ALL of them — not just the worst N — and build
intuition about failure modes before any more architecture tuning.

For each dev stem:
  - GT CEJ landmarks (yellow), GT bone-crest at GT CEJ x (orange),
    GT vertical line connecting them.
  - Predicted CEJ (green), predicted bone-crest (red), predicted
    landmarks per the Family A pipeline.
  - mm labels: M/D GT=X P=Y per tooth.
  - Header bar: stem, n_sites, per-image mean MAE, max site error.

Outputs:
  output/diagnostics/dev-contact-sheet/<rank>_<stem>.png
  output/diagnostics/dev-contact-sheet/index.html
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dental_rad_cli.analyze import _get_or_create_bundle, analyze  # noqa: E402
from dental_rad_cli.data.denpar_adapter import _split_dir  # noqa: E402
from benchmark_eval import _bbox_iou, _derive_gt_mm, _load_split  # noqa: E402


def _render(img: np.ndarray, gt_teeth: list, result) -> tuple[np.ndarray, dict]:
    """Draw overlays; return (overlay, per_image_stats).

    Stats: {n_sites_gt, n_sites_pred, mean_abs_err, max_abs_err, errs}.
    """
    overlay = img.copy()
    errs: list[float] = []
    n_gt_sites = 0
    n_pred_sites = 0

    for gt in gt_teeth:
        bb = [int(round(c)) for c in gt["bbox"]]
        cv2.rectangle(overlay, (bb[0], bb[1]), (bb[2], bb[3]),
                      (255, 255, 0), 2)

        for cej_pt in (gt["mesial_cej"], gt["distal_cej"]):
            if cej_pt is None:
                continue
            cv2.circle(overlay, (int(cej_pt[0]), int(cej_pt[1])), 7,
                       (0, 255, 255), -1)
            cv2.circle(overlay, (int(cej_pt[0]), int(cej_pt[1])), 7,
                       (0, 0, 0), 1)

        for cej_pt, bone_y in (
            (gt["mesial_cej"], gt.get("mesial_bone_y")),
            (gt["distal_cej"], gt.get("distal_bone_y")),
        ):
            if cej_pt is None or bone_y is None:
                continue
            cv2.circle(overlay, (int(cej_pt[0]), int(bone_y)), 7,
                       (0, 165, 255), -1)
            cv2.circle(overlay, (int(cej_pt[0]), int(bone_y)), 7,
                       (0, 0, 0), 1)
            cv2.line(overlay,
                     (int(cej_pt[0]), int(cej_pt[1])),
                     (int(cej_pt[0]), int(bone_y)),
                     (0, 165, 255), 2)

        # Match predicted tooth.
        best_iou, best_pred = 0.0, None
        for t in result.teeth:
            iou = _bbox_iou(gt["bbox"], t.bbox)
            if iou > best_iou:
                best_iou, best_pred = iou, t

        if best_pred is None or best_iou < 0.3:
            for site_mm in (gt["mesial_mm"], gt["distal_mm"]):
                if site_mm is not None:
                    n_gt_sites += 1
            continue

        kp = best_pred.keypoints
        for cej in (kp.cej_mesial, kp.cej_distal):
            if cej is not None:
                cv2.circle(overlay, (int(cej[0]), int(cej[1])), 5,
                           (0, 255, 0), -1)
                cv2.circle(overlay, (int(cej[0]), int(cej[1])), 5,
                           (0, 0, 0), 1)
        for bc in (kp.bone_crest_mesial, kp.bone_crest_distal):
            if bc is not None:
                cv2.circle(overlay, (int(bc[0]), int(bc[1])), 5,
                           (0, 0, 255), -1)
                cv2.circle(overlay, (int(bc[0]), int(bc[1])), 5,
                           (0, 0, 0), 1)

        for site_name, gt_mm, pred_site in (
            ("M", gt["mesial_mm"], best_pred.bone_loss.mesial),
            ("D", gt["distal_mm"], best_pred.bone_loss.distal),
        ):
            if gt_mm is None:
                continue
            n_gt_sites += 1
            pred_mm = pred_site.mm_estimate if pred_site else None
            if pred_mm is not None:
                n_pred_sites += 1
                err = abs(gt_mm - pred_mm)
                errs.append(err)
                txt = f"{site_name} GT={gt_mm:.1f} P={pred_mm:.1f} d={err:.2f}"
            else:
                txt = f"{site_name} GT={gt_mm:.1f} P=-"
            label_x = bb[0] if site_name == "M" else max(bb[0], bb[2] - 160)
            label_y = bb[3] + 20
            cv2.putText(overlay, txt, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(overlay, txt, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    stats = {
        "n_gt_sites": n_gt_sites,
        "n_pred_sites": n_pred_sites,
        "mean_abs_err": float(np.mean(errs)) if errs else None,
        "max_abs_err": float(max(errs)) if errs else None,
        "errs": errs,
    }
    return overlay, stats


def _add_header(img: np.ndarray, text: str) -> np.ndarray:
    """Add a 32 px black header strip with white text above the image."""
    h, w = img.shape[:2]
    header = np.zeros((32, w, 3), dtype=np.uint8)
    cv2.putText(header, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([header, img])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights"))
    ap.add_argument("--denpar-root", type=Path, default=Path("data/denpar"))
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument("--split", default="dev",
                    choices=["dev", "held-out", "all"])
    ap.add_argument("--out", type=Path,
                    default=Path("output/diagnostics/dev-contact-sheet"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.split == "held-out":
        print("ERROR: contact sheet on held-out is a held-out touch. Use "
              "benchmark_eval.py with --confirm-held-out-touch instead.",
              file=sys.stderr)
        return 2

    testing = _split_dir(args.denpar_root, "Testing")
    images_dir = testing / "Images"

    split_stems = _load_split(args.splits_dir, args.split)
    if split_stems is None:
        stems = sorted(p.stem for p in images_dir.glob("*.jpg"))
    else:
        stems = sorted(split_stems)
    if args.limit > 0:
        stems = stems[: args.limit]

    print(f"Contact sheet — split={args.split} ({len(stems)} images)")
    args.out.mkdir(parents=True, exist_ok=True)

    bundle = _get_or_create_bundle(args.weights)
    rows: list[dict] = []

    t0 = time.perf_counter()
    for idx, stem in enumerate(stems):
        if idx % 25 == 0:
            print(f"... {idx}/{len(stems)}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

        gt_teeth = _derive_gt_mm(testing, stem)
        img_path = images_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if not gt_teeth:
            out_img = _add_header(img,
                                  f"stem={stem}  NO GT (no derivable sites)")
            rows.append({
                "stem": stem, "mean_err": None, "max_err": None,
                "n_gt": 0, "n_pred": 0, "bucket": "no_gt",
                "_png": cv2.imencode(".png", out_img)[1].tobytes(),
            })
            continue

        try:
            result = analyze(img_path, weights_dir=args.weights,
                             bundle=bundle, render=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {stem}: analyze() failed: {e}", flush=True)
            continue

        overlay, stats = _render(img, gt_teeth, result)
        mean_e = stats["mean_abs_err"]
        max_e = stats["max_abs_err"]
        n_gt = stats["n_gt_sites"]
        n_pred = stats["n_pred_sites"]

        if mean_e is None:
            bucket = "no_pred"
            header_txt = (f"stem={stem}  NO PREDICTIONS  "
                          f"n_gt={n_gt}  n_pred={n_pred}")
        else:
            bucket = "scored"
            header_txt = (f"stem={stem}  mae={mean_e:.3f}  "
                          f"max={max_e:.3f}  n_gt={n_gt}  n_pred={n_pred}")

        out_img = _add_header(overlay, header_txt)
        rows.append({"stem": stem, "mean_err": mean_e, "max_err": max_e,
                     "n_gt": n_gt, "n_pred": n_pred, "bucket": bucket})

        # Don't write yet — wait until sort to embed rank in filename.
        # Cache the encoded bytes:
        rows[-1]["_png"] = cv2.imencode(".png", out_img)[1].tobytes()

    elapsed = time.perf_counter() - t0
    print(f"\nrendered {len(rows)} images in {elapsed:.0f}s")

    # Sort: scored by mean_err desc, then no_pred, then no_gt at end.
    scored = sorted(
        [r for r in rows if r["bucket"] == "scored"],
        key=lambda r: -r["mean_err"],
    )
    no_pred = sorted(
        [r for r in rows if r["bucket"] == "no_pred"],
        key=lambda r: r["stem"],
    )
    no_gt = sorted(
        [r for r in rows if r["bucket"] == "no_gt"],
        key=lambda r: r["stem"],
    )

    # Write PNGs with rank prefix in filename.
    rank = 0
    for r in scored + no_pred + no_gt:
        if "_png" not in r:
            continue
        rank += 1
        fname = f"{rank:03d}_{r['bucket']}_{r['stem']}.png"
        (args.out / fname).write_bytes(r["_png"])
        r["fname"] = fname

    # Write index.html.
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>dev contact sheet</title>",
        "<style>body{font-family:sans-serif;background:#222;color:#ddd;"
        "margin:0;padding:16px} h2{border-bottom:1px solid #555;"
        "padding-bottom:4px} .row{margin:12px 0} .row img{max-width:100%;"
        "border:1px solid #444} .meta{font-size:13px;color:#aaa;"
        "margin:4px 0}</style></head><body>",
        f"<h1>dev contact sheet — split={args.split} "
        f"({len(rows)} images, "
        f"{len(scored)} scored / {len(no_pred)} no_pred / "
        f"{len(no_gt)} no_gt)</h1>",
    ]
    if scored:
        scored_mean = float(np.mean([r["mean_err"] for r in scored]))
        scored_med = float(np.median([r["mean_err"] for r in scored]))
        lines.append(
            f"<p>scored: per-image MAE mean={scored_mean:.3f} "
            f"median={scored_med:.3f}</p>"
        )
    lines.append("<h2>scored, worst → best</h2>")
    for r in scored:
        lines.append("<div class='row'>")
        lines.append(
            f"<div class='meta'>stem={html.escape(r['stem'])} "
            f"mae={r['mean_err']:.3f} max={r['max_err']:.3f} "
            f"n_gt={r['n_gt']} n_pred={r['n_pred']}</div>"
        )
        lines.append(
            f"<img src='{html.escape(r['fname'])}' loading='lazy'/></div>"
        )
    if no_pred:
        lines.append("<h2>no predictions (model failed to fire)</h2>")
        for r in no_pred:
            lines.append("<div class='row'>")
            lines.append(
                f"<div class='meta'>stem={html.escape(r['stem'])} "
                f"n_gt={r['n_gt']}</div>"
            )
            lines.append(
                f"<img src='{html.escape(r['fname'])}' loading='lazy'/></div>"
            )
    if no_gt:
        lines.append("<h2>no derivable GT (skipped from benchmark)</h2>")
        for r in no_gt:
            lines.append("<div class='row'>")
            lines.append(
                f"<div class='meta'>stem={html.escape(r['stem'])}</div>"
            )
            lines.append(
                f"<img src='{html.escape(r['fname'])}' loading='lazy'/></div>"
            )
    lines.append("</body></html>")

    index = args.out / "index.html"
    index.write_text("\n".join(lines))
    print(f"wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
