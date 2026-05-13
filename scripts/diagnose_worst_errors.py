"""Render GT vs predicted landmark overlays for the worst-error cases.

For a stem from the benchmark, draws:
- GT mesial/distal CEJ points (yellow)
- GT mesial/distal bone-crest points at GT CEJ x (orange)
- Predicted mesial/distal CEJ (green)
- Predicted mesial/distal bone-crest (red)
- mm label at each site: "GT XmmᴬᶜᵗᵘᵃˡCloudPred Ymm"
- Tooth bbox (cyan)

If the GT landmarks don't match the actual anatomy on the image,
the GT itself is suspect and our MAE is overstated.

Usage::

    python scripts/diagnose_worst_errors.py --stem 106
    python scripts/diagnose_worst_errors.py --stem 1251 --tooth 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dental_rad_cli.analyze import _get_or_create_bundle, analyze  # noqa: E402
from dental_rad_cli.data.denpar_adapter import _split_dir  # noqa: E402
from benchmark_eval import _bbox_iou, _derive_gt_mm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--tooth", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("output/diagnostics/worst-errors"))
    ap.add_argument("--denpar-root", type=Path, default=Path("data/denpar"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    testing = _split_dir(args.denpar_root, "Testing")
    images_dir = testing / "Images"
    img_path = images_dir / f"{args.stem}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: image not found at {img_path}")
        return 1

    gt_teeth = _derive_gt_mm(testing, args.stem)
    if not gt_teeth:
        print("No GT teeth derivable for this stem.")
        return 1

    bundle = _get_or_create_bundle(Path("weights"))
    result = analyze(img_path, weights_dir=Path("weights"), bundle=bundle, render=False)

    overlay = img.copy()
    teeth_to_show = (
        [gt_teeth[args.tooth]] if args.tooth is not None else gt_teeth
    )
    for gt in teeth_to_show:
        bb = [int(round(c)) for c in gt["bbox"]]
        cv2.rectangle(overlay, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 0), 2)

        # GT CEJ landmarks (yellow circles).
        for cej_pt in (gt["mesial_cej"], gt["distal_cej"]):
            if cej_pt is None:
                continue
            cv2.circle(
                overlay, (int(cej_pt[0]), int(cej_pt[1])), 7, (0, 255, 255), -1
            )
            cv2.circle(
                overlay, (int(cej_pt[0]), int(cej_pt[1])), 7, (0, 0, 0), 1
            )

        # GT bone-crest at GT CEJ x (orange).
        for cej_pt, bone_y in (
            (gt["mesial_cej"], gt.get("mesial_bone_y")),
            (gt["distal_cej"], gt.get("distal_bone_y")),
        ):
            if cej_pt is None or bone_y is None:
                continue
            cv2.circle(
                overlay, (int(cej_pt[0]), int(bone_y)), 7, (0, 165, 255), -1
            )
            cv2.circle(
                overlay, (int(cej_pt[0]), int(bone_y)), 7, (0, 0, 0), 1
            )
            # GT vertical line.
            cv2.line(
                overlay, (int(cej_pt[0]), int(cej_pt[1])),
                (int(cej_pt[0]), int(bone_y)), (0, 165, 255), 2,
            )

        # Match predicted tooth.
        best_iou, best_pred = 0.0, None
        for t in result.teeth:
            iou = _bbox_iou(gt["bbox"], t.bbox)
            if iou > best_iou:
                best_iou, best_pred = iou, t
        if best_pred is None or best_iou < 0.3:
            continue

        # Predicted CEJ (green) + bone-crest (red).
        kp = best_pred.keypoints
        for cej in (kp.cej_mesial, kp.cej_distal):
            if cej is not None:
                cv2.circle(overlay, (int(cej[0]), int(cej[1])), 5, (0, 255, 0), -1)
                cv2.circle(overlay, (int(cej[0]), int(cej[1])), 5, (0, 0, 0), 1)
        for bc in (kp.bone_crest_mesial, kp.bone_crest_distal):
            if bc is not None:
                cv2.circle(overlay, (int(bc[0]), int(bc[1])), 5, (0, 0, 255), -1)
                cv2.circle(overlay, (int(bc[0]), int(bc[1])), 5, (0, 0, 0), 1)

        # mm labels.
        for site_name, gt_mm, pred_site in (
            ("M", gt["mesial_mm"], best_pred.bone_loss.mesial),
            ("D", gt["distal_mm"], best_pred.bone_loss.distal),
        ):
            pred_mm = pred_site.mm_estimate if pred_site else None
            txt = f"{site_name} GT={gt_mm:.1f}" if gt_mm else f"{site_name} GT=-"
            if pred_mm is not None:
                txt += f" P={pred_mm:.1f}"
            label_x = bb[0] if site_name == "M" else bb[2] - 100
            label_y = bb[3] + 20
            cv2.putText(
                overlay, txt, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
            )
            cv2.putText(
                overlay, txt, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

    out_path = args.out / f"{args.stem}.png"
    cv2.imwrite(str(out_path), overlay)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
