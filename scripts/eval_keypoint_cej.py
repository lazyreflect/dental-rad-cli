"""Held-out evaluation for the CEJ keypoint head — DenPAR Testing split.

Prints a single number: cej_collapse_rate (fraction of high-confidence
predictions where mesial-distal distance < 10 px). Lower is better.

This is the baseline metric for any autoresearch loop on the CEJ head.
It runs against the 200-image DenPAR Testing split (the data the
training pipeline already creates but historically never evaluated
against).

Usage::

    python scripts/eval_keypoint_cej.py [--weights weights/keypoint_cej.pt] \\
                                        [--score-threshold 0.5] \\
                                        [--collapse-threshold 10.0]

Exit code is 0 on success; the metric is printed as the last line in
the format ``cej_collapse_rate: 0.NNNN`` so callers can grep it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dental_rad_cli.training.keypoints import _build_model  # noqa: E402


_CLAHE_CLIP = 40.0
_CLAHE_TILE = (8, 8)


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
    return cv2.cvtColor(cv2.merge([cl.apply(L), a, b]), cv2.COLOR_LAB2RGB)


def evaluate(
    weights_path: Path,
    images_dir: Path,
    score_threshold: float = 0.5,
    collapse_threshold: float = 10.0,
    device: torch.device | None = None,
) -> dict:
    """Run the CEJ head on every image; return aggregate metrics."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    payload = torch.load(str(weights_path), map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    num_keypoints = int(payload.get("num_keypoints", 2)) if isinstance(payload, dict) else 2

    model = _build_model(num_keypoints=num_keypoints)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no .jpg images in {images_dir}")

    n_inst_total = 0
    n_collapsed = 0
    distances: list[float] = []
    t0 = time.perf_counter()

    for p in images:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = _apply_clahe(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
        with torch.no_grad():
            out = model([t])[0]
        scores = out["scores"].cpu().numpy()
        kps = out["keypoints"].cpu().numpy()
        for k in range(len(scores)):
            if scores[k] < score_threshold:
                continue
            m = kps[k, 0, :2]
            d = kps[k, 1, :2]
            dist = math.hypot(m[0] - d[0], m[1] - d[1])
            distances.append(dist)
            n_inst_total += 1
            if dist < collapse_threshold:
                n_collapsed += 1

    elapsed = time.perf_counter() - t0
    rate = n_collapsed / max(n_inst_total, 1)
    arr = np.array(distances) if distances else np.array([0.0])
    return {
        "n_images": len(images),
        "n_predictions": n_inst_total,
        "n_collapsed": n_collapsed,
        "cej_collapse_rate": rate,
        "median_md_distance": float(np.median(arr)),
        "mean_md_distance": float(arr.mean()),
        "p10_md_distance": float(np.percentile(arr, 10)),
        "p25_md_distance": float(np.percentile(arr, 25)),
        "p75_md_distance": float(np.percentile(arr, 75)),
        "p90_md_distance": float(np.percentile(arr, 90)),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "score_threshold": score_threshold,
        "collapse_threshold": collapse_threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights/keypoint_cej.pt"))
    ap.add_argument("--images", type=Path, default=Path("data/denpar/Dataset/Testing/Images"))
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--collapse-threshold", type=float, default=10.0)
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.images.is_dir():
        print(f"ERROR: images dir not found: {args.images}", file=sys.stderr)
        return 1

    result = evaluate(
        args.weights, args.images,
        score_threshold=args.score_threshold,
        collapse_threshold=args.collapse_threshold,
    )

    print(f"images:              {result['n_images']}")
    print(f"predictions:         {result['n_predictions']}")
    print(f"collapsed:           {result['n_collapsed']}")
    print(f"elapsed_seconds:     {result['elapsed_seconds']:.1f}")
    print(f"device:              {result['device']}")
    print(f"score_threshold:     {result['score_threshold']}")
    print(f"collapse_threshold:  {result['collapse_threshold']}")
    print(f"median_md_distance:  {result['median_md_distance']:.2f} px")
    print(f"mean_md_distance:    {result['mean_md_distance']:.2f} px")
    print(f"p10/25/75/90:        "
          f"{result['p10_md_distance']:.1f} / {result['p25_md_distance']:.1f} / "
          f"{result['p75_md_distance']:.1f} / {result['p90_md_distance']:.1f}")
    print()
    print(f"cej_collapse_rate: {result['cej_collapse_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
