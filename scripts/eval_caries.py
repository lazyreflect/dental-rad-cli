"""Held-out evaluation for the caries head — Baasils ICCMS test split.

Runs Ultralytics' standard `.val()` against the test split (or val
split via --split). Prints mAP50, mAP50-95, precision, recall, plus
per-class AP50.

Requires `data/prepared/yolo_caries/data.yaml` (run
`scripts/download_caries_data.sh` first to populate it).

Usage::

    python scripts/eval_caries.py [--weights weights/caries.pt] \\
                                  [--split test]

Exit code 0 on success. Headline metric printed as last lines in
format ``caries_map50: 0.NNNN`` and ``caries_map50_95: 0.NNNN``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=Path("weights/caries.pt"))
    ap.add_argument("--data-yaml", type=Path, default=Path("data/prepared/yolo_caries/data.yaml"))
    ap.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.data_yaml.exists():
        print(f"ERROR: data.yaml not found: {args.data_yaml}", file=sys.stderr)
        print("       Run scripts/download_caries_data.sh to populate.", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    result = model.val(data=str(args.data_yaml), split=args.split, verbose=False)

    print()
    print(f"=== Caries head — {args.split} split ===")
    print(f"  mAP50:     {result.box.map50:.4f}")
    print(f"  mAP50-95:  {result.box.map:.4f}")
    print(f"  Precision: {result.box.mp:.4f}")
    print(f"  Recall:    {result.box.mr:.4f}")
    print()
    print("Per class:")
    names = result.names
    ap50_arr = result.box.ap50
    ap_arr = result.box.ap
    for i, name in names.items():
        if i < len(ap50_arr):
            print(f"  {name:>10s}: AP50={ap50_arr[i]:.4f}  AP50-95={ap_arr[i]:.4f}")
    print()
    print(f"caries_map50: {result.box.map50:.4f}")
    print(f"caries_map50_95: {result.box.map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
