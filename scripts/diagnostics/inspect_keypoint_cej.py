#!/usr/bin/env python3
"""Inspect raw output of ``weights/keypoint_cej.pt`` for the keypoint-
collapse failure mode.

Context (2026-05-12 smoke test, results/inference-real-mac-2026-05-12/):
bw03 tooth #2 shows ``cej_mesial=(365, 111)`` and ``cej_distal=(365,
112)`` — the two CEJ keypoints collapsed to essentially the same point.
That collapse, paired with the bone-crest keypoints landing at bbox
edges (per the polygon-interpolation adapter fix at 0f7de0f), produced
clinically-impossible bone-loss percentages >100% via the old 2D
Euclidean severity formula.

The severity formula has been corrected (project onto tooth long
axis), but the keypoint collapse is still a real failure mode worth
understanding before the next training run.

This script is read-only. It does NOT modify the keypoint_cej weights,
does NOT change training code, and does NOT fix anything. It just
prints what the model is actually predicting on a known-collapsed
input image.

Usage::

    python scripts/diagnostics/inspect_keypoint_cej.py \\
        [--image examples/eval/bw03.png] \\
        [--weights weights/keypoint_cej.pt] \\
        [--output output/diagnostics/keypoint_cej_inspection.png] \\
        [--collapse-threshold 10.0]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_image_rgb(image_path: Path) -> Any:
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _apply_clahe(rgb_image: Any) -> Any:
    """Mirror the CLAHE constants from training (clip=40, grid=(8,8))."""
    import cv2

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _build_model_from_payload(payload: dict, default_num_keypoints: int = 2) -> Any:
    """Reuse the training-side model builder so architecture matches."""
    from dental_rad_cli.training.keypoints import _build_model

    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        num_keypoints = int(payload.get("num_keypoints", default_num_keypoints))
    else:
        state = payload
        num_keypoints = default_num_keypoints

    model = _build_model(num_keypoints=num_keypoints)
    model.load_state_dict(state)
    model.eval()
    return model, num_keypoints


def _kp_distance(kp_a: Tuple[float, float], kp_b: Tuple[float, float]) -> float:
    return math.hypot(kp_a[0] - kp_b[0], kp_a[1] - kp_b[1])


def _save_visualization(
    rgb_clahe: Any,
    instances: List[Dict[str, Any]],
    output_path: Path,
    collapse_threshold: float,
) -> None:
    """Overlay all predicted CEJ keypoints on the CLAHE BW.

    Each instance gets a distinct color; mesial is a filled circle,
    distal is an open ring. Collapsed instances (mesial-distal
    distance < threshold) are circled in red for emphasis.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(rgb_clahe)
    ax.set_title(
        f"keypoint_cej raw predictions on {output_path.stem}\n"
        f"red rings = collapsed (mesial-distal distance < {collapse_threshold:.1f}px)"
    )

    cmap = plt.get_cmap("tab20")
    for i, inst in enumerate(instances):
        color = cmap(i % 20)
        kps = inst["keypoints"]  # (K, 3)
        if kps.shape[0] >= 2:
            mx, my = float(kps[0, 0]), float(kps[0, 1])
            dx, dy = float(kps[1, 0]), float(kps[1, 1])
            ax.plot(mx, my, "o", color=color, markersize=8, markeredgecolor="black")
            ax.plot(
                dx, dy, "o", color="none",
                markeredgecolor=color, markeredgewidth=2, markersize=10,
            )
            # Annotate instance index near mesial point.
            ax.annotate(
                f"#{i} ({inst['score']:.2f})",
                xy=(mx, my), xytext=(6, -6), textcoords="offset points",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.7),
            )
            # Bounding box.
            bb = inst["bbox"]
            rect = plt.Rectangle(
                (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                fill=False, edgecolor=color, linestyle=":", linewidth=1,
            )
            ax.add_patch(rect)
            # Collapse highlight.
            if inst["collapsed"]:
                mid_x = 0.5 * (mx + dx)
                mid_y = 0.5 * (my + dy)
                ax.plot(
                    mid_x, mid_y, "o", color="none",
                    markeredgecolor="red", markeredgewidth=2.5, markersize=22,
                )

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path,
        default=Path("examples/eval/bw03.png"),
        help="image to run keypoint_cej against (CLAHE applied internally)",
    )
    parser.add_argument(
        "--weights", type=Path,
        default=Path("weights/keypoint_cej.pt"),
        help="path to keypoint_cej.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/diagnostics/keypoint_cej_inspection.png"),
        help="where to save the visualization PNG",
    )
    parser.add_argument(
        "--collapse-threshold", type=float, default=10.0,
        help="distance (px) below which mesial+distal counts as collapsed",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.05,
        help="filter predicted instances by score before reporting",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    print(f"Loading: {args.weights}")
    payload = torch.load(str(args.weights), map_location="cpu")
    model, num_keypoints = _build_model_from_payload(payload)
    print(f"  num_keypoints={num_keypoints}")
    if isinstance(payload, dict):
        meta_keys = {k: type(v).__name__ for k, v in payload.items() if k != "state_dict"}
        print(f"  payload meta: {meta_keys}")

    print(f"\nLoading image: {args.image}")
    rgb = _load_image_rgb(args.image)
    rgb_clahe = _apply_clahe(rgb)
    print(f"  shape: {rgb.shape}")

    tensor = torch.from_numpy(rgb_clahe).permute(2, 0, 1).float() / 255.0
    batch = [tensor]

    print("\nRunning inference...")
    with torch.no_grad():
        outs = model(batch)
    out0 = outs[0]

    boxes = out0["boxes"].cpu().numpy()
    keypoints = out0["keypoints"].cpu().numpy()       # (N, K, 3)
    keypoint_scores = out0.get("keypoints_scores")
    if keypoint_scores is None:
        keypoint_scores = out0.get("keypoint_scores")
    if keypoint_scores is not None:
        keypoint_scores = keypoint_scores.cpu().numpy()
    scores = out0["scores"].cpu().numpy()
    labels = out0["labels"].cpu().numpy() if "labels" in out0 else None

    print(f"Raw instances predicted: {len(boxes)}")
    if scores.size > 0:
        print(f"  score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  score-threshold filter: {args.score_threshold:.2f}")

    # Filter by score and assemble instance dicts.
    instances: List[Dict[str, Any]] = []
    collapsed_count = 0
    image_h, image_w = rgb.shape[:2]
    near_edge_threshold = 0.05  # 5% of image dim
    near_edge_x = near_edge_threshold * image_w
    near_edge_y = near_edge_threshold * image_h

    for i in range(len(boxes)):
        if scores[i] < args.score_threshold:
            continue
        kps = keypoints[i]
        if kps.shape[0] < 2:
            continue
        mesial = (float(kps[0, 0]), float(kps[0, 1]))
        distal = (float(kps[1, 0]), float(kps[1, 1]))
        dist = _kp_distance(mesial, distal)
        collapsed = dist < args.collapse_threshold
        if collapsed:
            collapsed_count += 1

        # Spatial-correlation: is the bbox center near any image edge?
        bb = boxes[i]
        cx = 0.5 * (bb[0] + bb[2])
        cy = 0.5 * (bb[1] + bb[3])
        near_edge = (
            cx < near_edge_x or cx > image_w - near_edge_x
            or cy < near_edge_y or cy > image_h - near_edge_y
        )

        inst = {
            "idx": i,
            "score": float(scores[i]),
            "bbox": (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
            "keypoints": kps,
            "mesial": mesial,
            "distal": distal,
            "distance": dist,
            "collapsed": collapsed,
            "near_image_edge": near_edge,
            "label": int(labels[i]) if labels is not None else None,
            "kp_scores": keypoint_scores[i].tolist()
                          if keypoint_scores is not None else None,
        }
        instances.append(inst)

    print(f"\nInstances after score filter (score >= {args.score_threshold}): "
          f"{len(instances)}")
    print(f"Collapsed instances (M-D distance < {args.collapse_threshold}px): "
          f"{collapsed_count} / {len(instances)}")

    # Per-instance dump.
    if instances:
        print(f"\n{'idx':>3s} {'score':>6s} {'M-D dist':>9s} {'collapsed':>10s} "
              f"{'near_edge':>10s}  mesial -> distal")
        print("-" * 92)
        for inst in instances:
            flag_c = "YES" if inst["collapsed"] else "no"
            flag_e = "yes" if inst["near_image_edge"] else "no"
            mx, my = inst["mesial"]
            dx, dy = inst["distal"]
            print(
                f"{inst['idx']:>3d} {inst['score']:>6.3f} "
                f"{inst['distance']:>9.2f} {flag_c:>10s} {flag_e:>10s}  "
                f"({mx:6.1f},{my:6.1f}) -> ({dx:6.1f},{dy:6.1f})"
            )

    # Correlation analyses.
    if collapsed_count > 0:
        collapsed_insts = [i for i in instances if i["collapsed"]]
        clean_insts = [i for i in instances if not i["collapsed"]]
        c_scores = [i["score"] for i in collapsed_insts]
        n_scores = [i["score"] for i in clean_insts]
        c_high_conf = sum(1 for s in c_scores if s >= 0.5)
        c_near_edge = sum(1 for i in collapsed_insts if i["near_image_edge"])

        def _mean(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        print("\nCollapse correlation analysis")
        print("-" * 30)
        print(f"  Collapsed instances:           {len(collapsed_insts)}")
        print(f"  Non-collapsed instances:       {len(clean_insts)}")
        print(f"  Mean score (collapsed):        {_mean(c_scores):.3f}")
        print(f"  Mean score (non-collapsed):    {_mean(n_scores):.3f}")
        print(f"  Collapsed AND high-conf (>=0.5): {c_high_conf} / {len(collapsed_insts)}")
        print(f"  Collapsed AND near image edge:   {c_near_edge} / {len(collapsed_insts)}")

    # Save visualization.
    print(f"\nSaving visualization to {args.output} ...")
    try:
        _save_visualization(rgb_clahe, instances, args.output, args.collapse_threshold)
        print("  done.")
    except Exception as exc:  # noqa: BLE001 — diagnostic must not crash on viz failure
        print(f"  WARNING: visualization failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
