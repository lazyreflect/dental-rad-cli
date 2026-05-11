"""PyTorch ``Dataset`` for the Renielaz-derived caries detection split.

Provided for symmetry with :class:`dental_rad_cli.data.denpar_dataset.DenParDetectionDataset`
— Ultralytics' YOLO trainer reads ``data.yaml`` + ``.txt`` label files
directly, so this class is **not** consumed by ``training/caries.py``.
It exists so non-Ultralytics callers (sanity checks, diagnostic
scripts, future torch-native experiments) can iterate over the
caries split with CLAHE preprocessing applied — matching the
inference-time preprocessing in
:func:`dental_rad_cli.pipeline.caries_inference.detect_caries`.

CLAHE constants match the rest of the pipeline:
``clip_limit=40.0``, ``tile_grid_size=(8, 8)`` (see
:mod:`dental_rad_cli.training.preprocess`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import torch
import torch.utils.data

from ..training.preprocess import apply_clahe

Split = Literal["train", "val", "test"]


class CariesDataset(torch.utils.data.Dataset):
    """Iterate over the internal-format caries YOLO split.

    Reads from the layout produced by
    :func:`dental_rad_cli.data.caries_adapter.build_yolo_caries_dataset`:
    ``<root>/images/<split>/*.{jpg,png}`` +
    ``<root>/labels/<split>/*.txt`` where each label row is
    ``class cx cy w h`` (normalized YOLOv8 bbox).

    ``__getitem__`` returns ``(image_tensor, target_dict)`` where:

    - ``image_tensor``: CHW float32 in [0, 1], CLAHE-enhanced (the
      caries detector trains on CLAHE-preprocessed crops to match
      inference-time conditioning).
    - ``target_dict``:
        - ``boxes``:  FloatTensor[N, 4] absolute xyxy pixels
        - ``labels``: Int64Tensor[N] internal class ids
          (0=initial, 1=moderate, 2=deep)
        - ``image_id``: Int64Tensor[1]
    """

    def __init__(self, root: Path, split: Split) -> None:
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"missing images dir: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"missing labels dir: {self.labels_dir}")
        self._stems: list[str] = sorted(
            p.stem for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self._stems)

    def _read_image(self, stem: str) -> tuple["torch.Tensor", int, int]:
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.images_dir / f"{stem}{ext}"
            if p.is_file():
                img_path = p
                break
        else:
            raise FileNotFoundError(f"no image found for stem {stem!r}")

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = apply_clahe(rgb)
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor, w, h

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", dict]:
        stem = self._stems[idx]
        img_tensor, img_w, img_h = self._read_image(stem)

        boxes: list[list[float]] = []
        labels: list[int] = []

        label_path = self.labels_dir / f"{stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                x1 = (cx - 0.5 * w) * img_w
                y1 = (cy - 0.5 * h) * img_h
                x2 = (cx + 0.5 * w) * img_w
                y2 = (cy + 0.5 * h) * img_h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_tensor, target
