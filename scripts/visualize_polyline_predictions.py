"""Render predicted CEJ polyline bands over test images for diagnosis.

Loads the trained CEJ polyline model and renders the predicted band
(union of all detection masks) over a sample of test images. Overlays
GT CEJ points (yellow dots) so the visual gap between prediction and
ground truth is immediately apparent.

Helps diagnose why the polyline-degenerate rate is high (predictions
fail to cross both bbox.x1 and bbox.x2 on many teeth):

- Are predictions too short / fragmented?
- Are predictions in the wrong y-region (off-CEJ)?
- Are predictions missing entire arches?

Usage::

    python scripts/visualize_polyline_predictions.py \\
        --weights weights/segmentation_cej.pt \\
        --n 12

Output: output/diagnostics/polyline-predictions/{stem}.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights/segmentation_cej.pt"))
    ap.add_argument("--denpar-root", type=Path, default=Path("data/denpar"))
    ap.add_argument(
        "--n", type=int, default=12,
        help="number of test images to render (default 12)"
    )
    ap.add_argument(
        "--conf", type=float, default=0.25,
        help="confidence threshold for YOLO predictions (default 0.25)"
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("output/diagnostics/polyline-predictions"),
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    testing = _split_dir(args.denpar_root, "Testing")
    images_dir = testing / "Images"
    stems = sorted(p.stem for p in images_dir.glob("*.jpg"))[: args.n]

    for stem in stems:
        img_path = images_dir / f"{stem}.jpg"
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = _apply_clahe(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]

        results = model.predict(source=rgb, conf=args.conf, verbose=False, save=False)
        band = np.zeros((h, w), dtype=bool)
        if results and results[0].masks is not None and len(results[0].masks) > 0:
            masks = results[0].masks.data.cpu().numpy().astype(bool)
            for m in masks:
                m_rs = cv2.resize(
                    m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                band |= m_rs

        # Render: predicted band in green, GT CEJ points in yellow.
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        green = overlay.copy()
        green[band] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.5, green, 0.5, 0)

        # Draw bboxes (cyan) from GT.
        kp = _load_keypoint_json(testing, stem)
        if kp:
            for bb in kp.get("bboxes", []):
                x1, y1, x2, y2 = [int(round(c)) for c in bb]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)
            # GT CEJ points (yellow).
            for x, y in kp.get("CEJ_Points", []):
                cv2.circle(
                    overlay, (int(round(x)), int(round(y))), 5, (0, 255, 255), -1
                )
                cv2.circle(
                    overlay, (int(round(x)), int(round(y))), 5, (0, 0, 0), 1
                )

        cv2.putText(
            overlay, f"conf>={args.conf} | masks={len(results[0].masks) if results and results[0].masks else 0}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            overlay, f"conf>={args.conf} | masks={len(results[0].masks) if results and results[0].masks else 0}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        out_path = args.out_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
