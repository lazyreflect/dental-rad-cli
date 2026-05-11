"""YOLOv8s training entrypoint — 3-class interproximal caries detection.

Classes (3-tier collapse from ICCMS 6-tier — full rationale at
``docs/caries-class-mapping.md``):

  0 = initial   (RA1 + RA2 + RA3 — enamel through EDJ)
  1 = moderate  (RB4 + RC5       — outer + middle dentin)
  2 = deep      (RC6             — inner dentin / pulp-near)

This is **not** the schema's depth field. The schema uses ICDAS-style
``E1``/``E2``/``D1``/``D2``/``D3`` (see
:class:`dental_rad_cli.schema.CariesFinding`). The inference helper
:func:`dental_rad_cli.pipeline.caries_inference.detect_caries` maps the
3-tier model output to the schema as::

    0 (initial)  → "E1"   (enamel-confined)
    1 (moderate) → "D1"   (outer dentin)
    2 (deep)     → "D3"   (deep dentin / pulp-near)

Hyperparameters (same shape as :mod:`dental_rad_cli.training.tooth_detect`):
``epochs=200``, ``imgsz=640``, ``lr0=1e-4``, ``optimizer='Adam'``,
``batch=4``, ``patience=25``, ``weight_decay=1e-6``.

The 3-class collapse is load-bearing: the Renielaz dataset is 586
images and the deepest tier (RC6) is the rarest class. Training a
direct 5- or 6-class model starves RC6 below useful accuracy. The
collapse pools enamel and middle-dentin tiers, keeps RC6 isolated as
its own class (so deep-caries recall stays auditable), and trains
against ~196 images/class on average instead of ~98.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from ultralytics import YOLO

# Paper-aligned defaults — match tooth_detect.py shape but YOLOv8s
# (smaller backbone) per task brief.
_MODEL_NAME: Final[str] = "yolov8s.pt"
_IMG_SIZE: Final[int] = 640
_LR0: Final[float] = 1e-4
_WEIGHT_DECAY: Final[float] = 1e-6
_OPTIMIZER: Final[str] = "Adam"
_BATCH: Final[int] = 4
_PATIENCE: Final[int] = 25


def train(data_yaml: Path, weights_out: Path, epochs: int = 200) -> Path:
    """Train the YOLOv8s caries detector and save final weights.

    Args:
        data_yaml: Path to the Ultralytics dataset YAML produced by
            :func:`dental_rad_cli.data.caries_adapter.build_yolo_caries_dataset`
            (3-class: initial/moderate/deep).
        weights_out: Destination path for the final ``.pt`` weights
            file (e.g. ``weights/caries.pt``). Parent directories are
            created if missing.
        epochs: Training epochs. Default 200 matches the methodology
            cadence; callers may override for smoke tests.

    Returns:
        The absolute path to the saved weights file.

    Notes:
        CLAHE preprocessing is applied at dataset-load time via
        :class:`dental_rad_cli.data.caries_dataset.CariesDataset` AND
        at inference time via
        :func:`dental_rad_cli.pipeline.caries_inference.detect_caries` —
        identical conditioning at both stages is mandatory (same
        discipline as the keypoint pipeline).
    """
    data_yaml = Path(data_yaml).resolve()
    weights_out = Path(weights_out).resolve()
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    project_dir = weights_out.parent / "_runs"
    run_name = f"{weights_out.stem}_yolov8s"

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

    parser = argparse.ArgumentParser(description="Train YOLOv8s caries detector.")
    parser.add_argument("--data", type=Path, required=True, help="dataset YAML path")
    parser.add_argument("--out", type=Path, required=True, help="output weights path")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    final = train(args.data, args.out, args.epochs)
    print(f"saved: {final}")
