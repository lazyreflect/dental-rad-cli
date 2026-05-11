"""YOLOv9e training entrypoint — 3-class tooth detection.

Classes (per methodology brief §1.2 + dataset shape):
  0 = background (implicit; not a YOLO class but downstream models expect 3)
  1 = single-rooted tooth
  2 = double-rooted (multi-rooted) tooth

Hyperparameters from the brief (§1.1):
  epochs=200, imgsz=640, lr0=0.0001, optimizer='Adam', batch=4, patience=25.

The data_yaml must point to a standard Ultralytics dataset YAML with
`train`, `val`, optional `test` keys plus `names` listing two foreground
classes ("single", "double"). The DenPAR adapter (Subagent F) produces
this YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from ultralytics import YOLO

# Paper-exact defaults from the brief (§1.1 Tooth Detection).
_MODEL_NAME: Final[str] = "yolov9e.pt"
_IMG_SIZE: Final[int] = 640
_LR0: Final[float] = 0.0001
_WEIGHT_DECAY: Final[float] = 1e-6
_OPTIMIZER: Final[str] = "Adam"
_BATCH: Final[int] = 4
_PATIENCE: Final[int] = 25


def train(data_yaml: Path, weights_out: Path, epochs: int = 200) -> Path:
    """Train YOLOv9e tooth detector and save final weights.

    Args:
        data_yaml: Path to Ultralytics dataset YAML (train/val/names).
        weights_out: Destination path for the final `.pt` weights file
            (e.g., `weights/tooth_detect.pt`). Parent directories are
            created if missing.
        epochs: Training epochs. Brief specifies 200 paper-grade; callers
            may override for smoke tests.

    Returns:
        The absolute path to the saved weights file.
    """
    data_yaml = Path(data_yaml).resolve()
    weights_out = Path(weights_out).resolve()
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    project_dir = weights_out.parent / "_runs"
    run_name = f"{weights_out.stem}_yolov9e"

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

    # Ultralytics writes best.pt under <project>/<name>/weights/best.pt.
    best = project_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        # Fallback to last.pt if early-stop tracker didn't promote best.
        best = project_dir / run_name / "weights" / "last.pt"
        if not best.exists():
            raise FileNotFoundError(
                f"Training completed but no weights file found under {best.parent}"
            )

    # Copy (not move) so the run directory remains intact for diagnostics.
    weights_out.write_bytes(best.read_bytes())
    return weights_out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv9e tooth detector.")
    parser.add_argument("--data", type=Path, required=True, help="dataset YAML path")
    parser.add_argument("--out", type=Path, required=True, help="output weights path")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    final = train(args.data, args.out, args.epochs)
    print(f"saved: {final}")
