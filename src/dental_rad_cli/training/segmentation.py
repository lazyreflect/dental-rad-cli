"""YOLOv8x-seg training — single-class checkpoints per landmark.

Per methodology brief §1.3, the upstream pipeline trains the
segmentation model with identical hyperparameters per target:

- target="tooth"  → produces `weights/segmentation_tooth.pt`
- target="bone"   → produces `weights/segmentation_bone.pt`
- target="cej"    → produces `weights/segmentation_cej.pt`
                    (v2 polyline pivot — supervision built via y-band
                    clustering of DenPAR CEJ_Points; see
                    `dental_rad_cli.data.denpar_adapter`)

All downstream into the rule layer for pattern classification and per-
tooth bone-loss measurement. The CEJ target replaces the Keypoint
R-CNN CEJ head in the v2 architecture; bone is unchanged.

Hyperparameters (brief §1.3, identical to detection except patience):
  epochs=200, imgsz=640, lr0=0.0001, optimizer='Adam', batch=4, patience=30.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from ultralytics import YOLO

_MODEL_NAME: Final[str] = "yolov8x-seg.pt"
_IMG_SIZE: Final[int] = 640
_LR0: Final[float] = 0.0001
_WEIGHT_DECAY: Final[float] = 1e-6
_OPTIMIZER: Final[str] = "Adam"
_BATCH: Final[int] = 4
_PATIENCE: Final[int] = 30


def train(
    target: Literal["tooth", "bone", "cej"],
    data_yaml: Path,
    weights_out: Path,
    epochs: int = 200,
) -> Path:
    """Train a single-class YOLOv8x-seg model.

    Args:
        target: One of "tooth", "bone", or "cej" — selects which
            single-class checkpoint we are producing. This parameter
            is informational for run naming; the actual class is
            dictated by the `data_yaml` (which must declare
            `nc: 1, names: ['<target>']`).
        data_yaml: Path to Ultralytics segmentation dataset YAML.
        weights_out: Destination `.pt` path. Parent dirs created if needed.
        epochs: Training epochs (default 200 per brief).

    Returns:
        Absolute path to saved weights.
    """
    if target not in ("tooth", "bone", "cej"):
        raise ValueError(
            f"target must be 'tooth', 'bone', or 'cej'; got {target!r}"
        )

    data_yaml = Path(data_yaml).resolve()
    weights_out = Path(weights_out).resolve()
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    project_dir = weights_out.parent / "_runs"
    run_name = f"{weights_out.stem}_yolov8x_seg_{target}"

    model = YOLO(_MODEL_NAME)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=_IMG_SIZE,
        lr0=_LR0,
        weight_decay=_WEIGHT_DECAY,
        optimizer=_OPTIMIZER,
        batch=_BATCH,
        patience=_PATIENCE,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        resume=False,
    )

    best = project_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        best = project_dir / run_name / "weights" / "last.pt"
        if not best.exists():
            raise FileNotFoundError(
                f"Training completed but no weights file found under {best.parent}"
            )

    weights_out.write_bytes(best.read_bytes())
    return weights_out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8x-seg single-class.")
    parser.add_argument(
        "--target", choices=("tooth", "bone", "cej"), required=True
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    final = train(args.target, args.data, args.out, args.epochs)
    print(f"saved: {final}")
